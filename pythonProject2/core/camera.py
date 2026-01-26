"""
图像采集模块 - 支持摄像头实时采集和本地图像导入
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CameraManager:
    """摄像头管理器类"""

    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1920, 1080)):
        """
        初始化摄像头管理器

        Args:
            camera_id: 摄像头ID（默认0）
            resolution: 分辨率 (宽, 高)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.is_capturing = False
        self.capture_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()

    def initialize(self) -> bool:
        """
        初始化摄像头

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_id}")
                return False

            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # 设置自动曝光为手动模式（减少反光）
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动模式
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # 降低曝光

            logger.info(f"摄像头 {self.camera_id} 初始化成功，分辨率: {self.resolution}")
            return True

        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False

    def start_capture(self) -> bool:
        """
        开始实时采集

        Returns:
            bool: 启动是否成功
        """
        if self.cap is None:
            if not self.initialize():
                return False

        if self.is_capturing:
            logger.warning("摄像头已经在采集状态")
            return True

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        logger.info("摄像头采集已启动")
        return True

    def _capture_loop(self):
        """摄像头采集循环"""
        while self.is_capturing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                logger.warning("摄像头采集失败")
                time.sleep(0.1)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取当前帧

        Returns:
            Optional[np.ndarray]: 当前帧图像，失败返回None
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        捕获单帧图像

        Returns:
            Optional[np.ndarray]: 单帧图像，失败返回None
        """
        if self.cap is None:
            if not self.initialize():
                return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def stop_capture(self):
        """停止采集"""
        self.is_capturing = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        logger.info("摄像头采集已停止")

    def set_camera_parameters(self, brightness: float = 0, contrast: float = 1.0,
                              saturation: float = 1.0, exposure: float = -6):
        """
        设置摄像头参数

        Args:
            brightness: 亮度
            contrast: 对比度
            saturation: 饱和度
            exposure: 曝光值
        """
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, saturation)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            logger.info(f"摄像头参数已设置: brightness={brightness}, contrast={contrast}, "
                        f"saturation={saturation}, exposure={exposure}")

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        加载本地图像

        Args:
            image_path: 图像路径

        Returns:
            Optional[np.ndarray]: 加载的图像，失败返回None
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法加载图像: {image_path}")
                return None

            # 转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"图像加载成功: {image_path}, 尺寸: {image.shape}")
            return image

        except Exception as e:
            logger.error(f"加载图像失败: {e}")
            return None

    def save_image(self, image: np.ndarray, save_path: str) -> bool:
        """
        保存图像到本地

        Args:
            image: 要保存的图像
            save_path: 保存路径

        Returns:
            bool: 保存是否成功
        """
        try:
            # 转换为BGR格式保存
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            cv2.imwrite(save_path, image_bgr)
            logger.info(f"图像保存成功: {save_path}")
            return True

        except Exception as e:
            logger.error(f"保存图像失败: {e}")
            return False

    def get_camera_info(self) -> dict:
        """
        获取摄像头信息

        Returns:
            dict: 摄像头信息字典
        """
        if self.cap is None:
            return {"status": "未初始化"}

        info = {
            "camera_id": self.camera_id,
            "is_opened": self.cap.isOpened(),
            "resolution": (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
            "gain": self.cap.get(cv2.CAP_PROP_GAIN),
        }
        return info

    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop_capture()



class CameraController:
    """兼容性包装类，用于适配UI代码"""

    def __init__(self, camera_id=0):
        self.camera = CameraManager(camera_id)
        self.is_opened_flag = False

    def is_opened(self):
        """检查相机是否打开"""
        return self.camera.is_capturing and self.is_opened_flag

    def open(self, camera_index=0):
        """打开相机"""
        self.camera = CameraManager(camera_index)
        if self.camera.initialize():
            self.is_opened_flag = True
            return True
        return False

    def close(self):
        """关闭相机"""
        self.camera.stop_capture()
        self.is_opened_flag = False

    def get_frame(self):
        """获取帧"""
        return self.camera.get_frame()

    def get_camera_info(self):
        """获取相机信息"""
        return self.camera.get_camera_info()