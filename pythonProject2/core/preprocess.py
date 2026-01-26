"""
图像预处理模块 - 反光抑制、噪声去除、边缘优化等
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """图像预处理器类"""

    def __init__(self):
        """初始化预处理器"""
        self.params = {
            'retinex_scales': [15, 80, 250],  # Retinex算法尺度
            'gamma': 1.0,  # Gamma校正值
            'clip_limit': 2.0,  # CLAHE限制
            'grid_size': 8,  # CLAHE网格大小
            'median_kernel': 3,  # 中值滤波核大小
            'gaussian_kernel': 5,  # 高斯滤波核大小
            'gaussian_sigma': 1.5,  # 高斯滤波sigma
            'morph_kernel': 3,  # 形态学操作核大小
            'canny_threshold1': 50,  # Canny边缘检测低阈值
            'canny_threshold2': 150,  # Canny边缘检测高阈值
        }

    def preprocess_pipeline(self, image: np.ndarray,
                            steps: list = None) -> np.ndarray:
        """
        完整的图像预处理流程

        Args:
            image: 输入图像
            steps: 预处理步骤列表，可选

        Returns:
            np.ndarray: 预处理后的图像
        """
        if steps is None:
            steps = ['light_compensation', 'denoise', 'edge_enhance']

        processed = image.copy()

        for step in steps:
            if step == 'light_compensation':
                processed = self.adaptive_light_compensation(processed)
            elif step == 'denoise':
                processed = self.denoise_image(processed)
            elif step == 'edge_enhance':
                processed = self.edge_enhancement(processed)
            elif step == 'contrast_enhance':
                processed = self.contrast_enhancement(processed)
            elif step == 'grayscale':
                processed = self.to_grayscale(processed)
            else:
                logger.warning(f"未知的预处理步骤: {step}")

        return processed

    def adaptive_light_compensation(self, image: np.ndarray,
                                    method: str = 'msrcr') -> np.ndarray:
        """
        自适应光照补偿（抑制金属反光）

        Args:
            image: 输入图像
            method: 方法选择 ('msrcr', 'clahe', 'hist_equal')

        Returns:
            np.ndarray: 光照补偿后的图像
        """
        if len(image.shape) == 2:
            # 灰度图像
            if method == 'msrcr':
                return self._msrcr_gray(image)
            elif method == 'clahe':
                return self._clahe_processing(image)
            elif method == 'hist_equal':
                return cv2.equalizeHist(image)
            else:
                return image

        # 彩色图像
        if method == 'msrcr':
            return self._multi_scale_retinex(image)
        elif method == 'clahe':
            return self._clahe_color(image)
        elif method == 'hist_equal':
            # 分离通道处理
            channels = cv2.split(image)
            eq_channels = []
            for ch in channels:
                eq_channels.append(cv2.equalizeHist(ch))
            return cv2.merge(eq_channels)
        else:
            return image

    def _multi_scale_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        多尺度Retinex算法（抑制反光）

        Args:
            image: 输入彩色图像

        Returns:
            np.ndarray: 处理后的图像
        """
        # 转换为浮点型
        img_float = image.astype(np.float32) / 255.0

        # 多尺度高斯滤波
        scales = self.params['retinex_scales']
        retinex = np.zeros_like(img_float)

        for scale in scales:
            # 高斯模糊
            blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
            # 避免除零
            blurred = np.maximum(blurred, 1e-6)
            # Retinex计算
            retinex += np.log(img_float + 1e-6) - np.log(blurred)

        # 平均
        retinex = retinex / len(scales)

        # 归一化到0-255
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        retinex = (retinex * 255).astype(np.uint8)

        return retinex

    def _msrcr_gray(self, image: np.ndarray) -> np.ndarray:
        """
        灰度图像的MSRCR算法

        Args:
            image: 输入灰度图像

        Returns:
            np.ndarray: 处理后的图像
        """
        img_float = image.astype(np.float32) / 255.0
        scales = self.params['retinex_scales']

        msrcr = np.zeros_like(img_float)
        for scale in scales:
            blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
            blurred = np.maximum(blurred, 1e-6)
            msrcr += np.log(img_float + 1e-6) - np.log(blurred)

        msrcr = msrcr / len(scales)

        # 颜色恢复（简化版）
        msrcr = 255 * (np.tanh(msrcr * 0.5) + 1) / 2

        return msrcr.astype(np.uint8)

    def _clahe_color(self, image: np.ndarray) -> np.ndarray:
        """
        彩色图像的CLAHE处理

        Args:
            image: 输入彩色图像

        Returns:
            np.ndarray: 处理后的图像
        """
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 对L通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.params['clip_limit'],
                                tileGridSize=(self.params['grid_size'], self.params['grid_size']))
        l_clahe = clahe.apply(l)

        # 合并通道并转换回RGB
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return result

    def _clahe_processing(self, image: np.ndarray) -> np.ndarray:
        """
        灰度图像的CLAHE处理

        Args:
            image: 输入灰度图像

        Returns:
            np.ndarray: 处理后的图像
        """
        clahe = cv2.createCLAHE(clipLimit=self.params['clip_limit'],
                                tileGridSize=(self.params['grid_size'], self.params['grid_size']))
        return clahe.apply(image)

    def denoise_image(self, image: np.ndarray,
                      method: str = 'median_gaussian') -> np.ndarray:
        """
        图像去噪

        Args:
            image: 输入图像
            method: 去噪方法 ('median', 'gaussian', 'median_gaussian', 'bilateral')

        Returns:
            np.ndarray: 去噪后的图像
        """
        if method == 'median':
            kernel = self.params['median_kernel']
            if kernel % 2 == 0:
                kernel += 1
            return cv2.medianBlur(image, kernel)

        elif method == 'gaussian':
            kernel = self.params['gaussian_kernel']
            sigma = self.params['gaussian_sigma']
            if kernel % 2 == 0:
                kernel += 1
            return cv2.GaussianBlur(image, (kernel, kernel), sigma)

        elif method == 'median_gaussian':
            # 先中值后高斯（组合滤波）
            kernel_m = self.params['median_kernel']
            if kernel_m % 2 == 0:
                kernel_m += 1
            median = cv2.medianBlur(image, kernel_m)

            kernel_g = self.params['gaussian_kernel']
            sigma = self.params['gaussian_sigma']
            if kernel_g % 2 == 0:
                kernel_g += 1
            return cv2.GaussianBlur(median, (kernel_g, kernel_g), sigma)

        elif method == 'bilateral':
            # 双边滤波（保边去噪）
            return cv2.bilateralFilter(image, 9, 75, 75)

        else:
            return image

    def edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        边缘增强

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 边缘增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 使用Sobel算子增强边缘
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 归一化
        sobel = np.uint8(255 * sobel / np.max(sobel))

        # 与原始图像融合
        enhanced = cv2.addWeighted(gray, 0.7, sobel, 0.3, 0)

        if len(image.shape) == 3:
            # 如果是彩色图像，只增强亮度通道
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = cv2.addWeighted(l, 0.7, sobel, 0.3, 0)
            result = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

        return enhanced

    def contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        对比度增强

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 对比度增强后的图像
        """
        # Gamma校正
        gamma = self.params['gamma']
        inv_gamma = 1.0 / gamma

        # 构建Gamma查找表
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype(np.uint8)

        if len(image.shape) == 3:
            # 彩色图像分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []
            for ch in channels:
                enhanced_channels.append(cv2.LUT(ch, table))
            return cv2.merge(enhanced_channels)
        else:
            # 灰度图像
            return cv2.LUT(image, table)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        转换为灰度图像

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 灰度图像
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def morphological_operations(self, image: np.ndarray,
                                 operation: str = 'close') -> np.ndarray:
        """
        形态学操作

        Args:
            image: 输入二值图像
            operation: 操作类型 ('open', 'close', 'erode', 'dilate')

        Returns:
            np.ndarray: 处理后的图像
        """
        kernel_size = self.params['morph_kernel']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'erode':
            return cv2.erode(image, kernel, iterations=1)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=1)
        else:
            return image

    def canny_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Canny边缘检测

        Args:
            image: 输入灰度图像

        Returns:
            np.ndarray: 边缘图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Canny边缘检测
        edges = cv2.Canny(blurred,
                          self.params['canny_threshold1'],
                          self.params['canny_threshold2'])

        return edges

    def update_parameters(self, **kwargs):
        """
        更新预处理参数

        Args:
            **kwargs: 参数键值对
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                logger.debug(f"更新参数: {key} = {value}")
            else:
                logger.warning(f"未知参数: {key}")

    def process(self, image, **kwargs):
        """
        兼容旧接口的process方法
        参数映射：
        - brightness_compensation -> light_compensation步骤
        - retinex_enabled -> 包含在light_compensation中
        - median_filter_size -> 更新median_kernel参数
        - gaussian_filter_size -> 更新gaussian_kernel参数
        - canny_low/canny_high -> 更新canny_threshold1/2参数
        """
        # 默认步骤
        steps = ['light_compensation', 'denoise', 'edge_enhance']

        # 更新参数
        if 'median_filter_size' in kwargs:
            self.update_parameters(median_kernel=kwargs['median_filter_size'])

        if 'gaussian_filter_size' in kwargs:
            self.update_parameters(gaussian_kernel=kwargs['gaussian_filter_size'])

        if 'canny_low' in kwargs and 'canny_high' in kwargs:
            self.update_parameters(
                canny_threshold1=kwargs['canny_low'],
                canny_threshold2=kwargs['canny_high']
            )

        # 如果不需要亮度补偿，移除该步骤
        if not kwargs.get('brightness_compensation', True):
            steps.remove('light_compensation')

        logger.info(f"使用预处理步骤: {steps}, 参数: {kwargs}")
        return self.preprocess_pipeline(image, steps=steps)