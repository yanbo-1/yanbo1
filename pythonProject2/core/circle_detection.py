"""
圆心检测模块 - 改进霍夫变换、最小二乘法等算法
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from scipy import optimize
import math

logger = logging.getLogger(__name__)


class CircleDetector:
    """圆心检测器类"""

    def __init__(self):
        """初始化圆心检测器"""
        self.params = {
            'hough_dp': 1.2,  # 霍夫变换累加器分辨率
            'hough_min_dist': 50,  # 圆心间最小距离
            'hough_param1': 100,  # Canny边缘检测高阈值
            'hough_param2': 30,  # 累加器阈值
            'hough_min_radius': 10,  # 最小半径
            'hough_max_radius': 200,  # 最大半径
            'lsq_max_iter': 100,  # 最小二乘法最大迭代次数
            'lsq_tolerance': 1e-6,  # 最小二乘法容忍度
            'edge_threshold': 0.1,  # 边缘检测阈值
            'contour_min_area': 100,  # 轮廓最小面积
        }

        # === 添加同心度检测专用参数 ===
        self.concentricity_params = {
            'outer_min_radius': 100,  # 外筒最小半径
            'outer_max_radius': 500,  # 外筒最大半径
            'outer_min_dist': 300,  # 外筒圆心最小距离
            'thread_min_area': 500,  # 螺纹杆最小面积
            'thread_roi_expand': 20,  # ROI扩展像素
            'thread_clahe_clip': 3.0,  # CLAHE clip limit
            'hexagon_vertex_count': 6,  # 六边形顶点数
            'hexagon_tolerance': 1,  # 顶点数容差
            'quality_threshold_mm': 0.2,  # 合格阈值（毫米）
            'pixel_to_mm_ratio': 0.1,  # 像素到毫米转换系数（默认值）
        }

        # 检测结果缓存
        self.last_detection = None
        self.last_concentricity_result = None  # 新增缓存

    def detect_circles(self, image: np.ndarray,
                       method: str = 'hough_improved',
                       target_count: int = 2) -> List[Dict]:
        """
        检测图像中的圆

        Args:
            image: 输入图像（最好是边缘图像或二值图像）
            method: 检测方法 ('hough', 'hough_improved', 'lsq', 'contour')
            target_count: 目标检测圆的数量

        Returns:
            List[Dict]: 检测到的圆列表，每个字典包含(x, y, radius, confidence)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        circles = []

        if method == 'hough':
            circles = self._hough_transform(gray)
        elif method == 'hough_improved':
            circles = self._improved_hough_transform(gray)
        elif method == 'lsq':
            circles = self._least_squares_fitting(gray)
        elif method == 'contour':
            circles = self._contour_based_detection(gray)
        else:
            logger.error(f"未知的检测方法: {method}")
            return []

        # 按置信度排序
        circles.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        # 限制返回数量
        if target_count > 0 and len(circles) > target_count:
            circles = circles[:target_count]

        self.last_detection = circles
        return circles

    def _hough_transform(self, gray_image: np.ndarray) -> List[Dict]:
        """
        标准霍夫变换圆检测

        Args:
            gray_image: 灰度图像

        Returns:
            List[Dict]: 检测到的圆
        """
        # 使用高斯模糊降噪
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 2)

        # 霍夫变换检测圆
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.params['hough_dp'],
            minDist=self.params['hough_min_dist'],
            param1=self.params['hough_param1'],
            param2=self.params['hough_param2'],
            minRadius=self.params['hough_min_radius'],
            maxRadius=self.params['hough_max_radius']
        )

        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, circle in enumerate(circles[0, :]):
                x, y, r = circle
                # 计算置信度（基于边缘强度）
                confidence = self._calculate_circle_confidence(gray_image, x, y, r)
                result.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': float(r),
                    'confidence': confidence,
                    'method': 'hough'
                })

        return result

    def _improved_hough_transform(self, gray_image: np.ndarray) -> List[Dict]:
        """
        改进的霍夫变换（多尺度检测）

        Args:
            gray_image: 灰度图像

        Returns:
            List[Dict]: 检测到的圆
        """
        # 多尺度边缘检测
        edges_multi = self._multi_scale_edge_detection(gray_image)

        # 在不同尺度上检测圆
        all_circles = []

        # 尺度参数
        dp_values = [1.0, 1.2, 1.5]
        param2_values = [20, 30, 40]

        for edges in edges_multi:
            for dp in dp_values:
                for param2 in param2_values:
                    circles = cv2.HoughCircles(
                        edges,
                        cv2.HOUGH_GRADIENT,
                        dp=dp,
                        minDist=self.params['hough_min_dist'],
                        param1=self.params['hough_param1'],
                        param2=param2,
                        minRadius=self.params['hough_min_radius'],
                        maxRadius=self.params['hough_max_radius']
                    )

                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for circle in circles[0, :]:
                            x, y, r = circle
                            # 去重检查
                            if not self._is_duplicate_circle(all_circles, x, y, r):
                                confidence = self._calculate_circle_confidence(gray_image, x, y, r)
                                all_circles.append({
                                    'x': float(x),
                                    'y': float(y),
                                    'radius': float(r),
                                    'confidence': confidence,
                                    'method': 'hough_improved',
                                    'dp': dp,
                                    'param2': param2
                                })

        # 合并相似圆
        merged_circles = self._merge_similar_circles(all_circles)

        return merged_circles

    def _least_squares_fitting(self, gray_image: np.ndarray) -> List[Dict]:
        """
        最小二乘法圆拟合

        Args:
            gray_image: 灰度图像

        Returns:
            List[Dict]: 拟合的圆
        """
        # 边缘检测
        edges = cv2.Canny(gray_image, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = []

        for contour in contours:
            # 过滤小轮廓
            area = cv2.contourArea(contour)
            if area < self.params['contour_min_area']:
                continue

            # 多边形逼近
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 5:  # 点太少不适合圆拟合
                continue

            # 提取轮廓点
            points = contour.reshape(-1, 2)

            try:
                # 最小二乘法拟合圆
                x, y, r = self._fit_circle_least_squares(points)

                # 计算拟合误差
                error = self._calculate_fitting_error(points, x, y, r)

                if error < 5.0:  # 误差阈值
                    confidence = 1.0 / (1.0 + error)
                    result.append({
                        'x': float(x),
                        'y': float(y),
                        'radius': float(r),
                        'confidence': confidence,
                        'method': 'lsq',
                        'fitting_error': error
                    })
            except Exception as e:
                logger.debug(f"圆拟合失败: {e}")
                continue

        return result

    def _contour_based_detection(self, gray_image: np.ndarray) -> List[Dict]:
        """
        基于轮廓的圆检测

        Args:
            gray_image: 灰度图像

        Returns:
            List[Dict]: 检测到的圆
        """
        # 二值化
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.params['contour_min_area']:
                continue

            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # 计算圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # 圆形度阈值
            if circularity > 0.7:
                confidence = circularity
                result.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': float(radius),
                    'confidence': confidence,
                    'method': 'contour',
                    'circularity': circularity,
                    'area': area
                })

        return result

    def _multi_scale_edge_detection(self, image: np.ndarray) -> List[np.ndarray]:
        """
        多尺度边缘检测

        Args:
            image: 输入图像

        Returns:
            List[np.ndarray]: 多尺度边缘图像列表
        """
        scales = [1.0, 1.5, 2.0]
        edges_list = []

        for scale in scales:
            # 缩放图像
            height, width = image.shape
            new_size = (int(width / scale), int(height / scale))
            resized = cv2.resize(image, new_size)

            # 边缘检测
            edges = cv2.Canny(resized, 30, 100)

            # 缩放回原始尺寸
            edges = cv2.resize(edges, (width, height))

            edges_list.append(edges)

        return edges_list

    def _fit_circle_least_squares(self, points: np.ndarray) -> Tuple[float, float, float]:
        """
        最小二乘法拟合圆

        Args:
            points: 点集 (N x 2)

        Returns:
            Tuple[float, float, float]: (圆心x, 圆心y, 半径)
        """

        def calc_R(xc, yc):
            """计算半径"""
            return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

        def f_2(c):
            """误差函数"""
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        # 初始猜测（点集的中心）
        x_m = points[:, 0].mean()
        y_m = points[:, 1].mean()
        center_estimate = x_m, y_m

        # 优化
        center, _ = optimize.leastsq(f_2, center_estimate,
                                     maxfev=self.params['lsq_max_iter'])

        xc, yc = center
        Ri = calc_R(xc, yc)
        R = Ri.mean()

        return xc, yc, R

    def _calculate_circle_confidence(self, image: np.ndarray,
                                     x: int, y: int, r: int) -> float:
        """
        计算圆的置信度

        Args:
            image: 灰度图像
            x, y: 圆心坐标
            r: 半径

        Returns:
            float: 置信度 (0-1)
        """
        # 创建圆形掩码
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), r, 255, -1)

        # 计算边缘强度
        edges = cv2.Canny(image, 50, 150)
        edge_points = np.where((edges > 0) & (mask > 0))

        if len(edge_points[0]) == 0:
            return 0.0

        # 计算圆形度
        circle_perimeter = 2 * np.pi * r
        edge_density = len(edge_points[0]) / circle_perimeter

        # 计算均匀性
        angles = np.arctan2(edge_points[0] - y, edge_points[1] - x)
        angle_hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        uniformity = 1.0 - np.std(angle_hist) / np.mean(angle_hist)

        # 综合置信度
        confidence = 0.6 * min(edge_density, 1.0) + 0.4 * max(uniformity, 0)

        return min(max(confidence, 0.0), 1.0)

    def _calculate_fitting_error(self, points: np.ndarray,
                                 x: float, y: float, r: float) -> float:
        """
        计算拟合误差

        Args:
            points: 点集
            x, y: 圆心坐标
            r: 半径

        Returns:
            float: 平均误差
        """
        distances = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
        errors = np.abs(distances - r)
        return np.mean(errors)

    def _is_duplicate_circle(self, circles: List[Dict],
                             x: int, y: int, r: int,
                             threshold: float = 10.0) -> bool:
        """
        检查是否为重复圆

        Args:
            circles: 已检测的圆列表
            x, y, r: 待检查的圆参数
            threshold: 重复阈值

        Returns:
            bool: 是否为重复圆
        """
        for circle in circles:
            dx = abs(circle['x'] - x)
            dy = abs(circle['y'] - y)
            dr = abs(circle['radius'] - r)

            if dx < threshold and dy < threshold and dr < threshold:
                return True

        return False

    def _merge_similar_circles(self, circles: List[Dict],
                               distance_threshold: float = 15.0,
                               radius_threshold: float = 10.0) -> List[Dict]:
        """
        合并相似的圆

        Args:
            circles: 圆列表
            distance_threshold: 距离阈值
            radius_threshold: 半径阈值

        Returns:
            List[Dict]: 合并后的圆列表
        """
        if not circles:
            return []

        # 按置信度排序
        circles.sort(key=lambda x: x['confidence'], reverse=True)

        merged = []
        used = [False] * len(circles)

        for i, circle in enumerate(circles):
            if used[i]:
                continue

            # 找到相似的圆
            similar_indices = [i]
            for j in range(i + 1, len(circles)):
                if used[j]:
                    continue

                dx = abs(circle['x'] - circles[j]['x'])
                dy = abs(circle['y'] - circles[j]['y'])
                dr = abs(circle['radius'] - circles[j]['radius'])

                if dx < distance_threshold and dy < distance_threshold and dr < radius_threshold:
                    similar_indices.append(j)
                    used[j] = True

            # 合并相似圆（加权平均）
        if len(similar_indices) == 1:
            merged.append(circle)
        else:
            # 计算加权平均值
            total_confidence = sum(circles[idx]['confidence'] for idx in similar_indices)

            # 添加零值检查
            if total_confidence > 0:
                x_weighted = sum(circles[idx]['x'] * circles[idx]['confidence']
                                 for idx in similar_indices) / total_confidence
                y_weighted = sum(circles[idx]['y'] * circles[idx]['confidence']
                                 for idx in similar_indices) / total_confidence
                r_weighted = sum(circles[idx]['radius'] * circles[idx]['confidence']
                                 for idx in similar_indices) / total_confidence
            else:
                # 如果总置信度为0，使用算术平均
                logger.warning(f"合并圆时总置信度为0，使用算术平均")
                x_weighted = np.mean([circles[idx]['x'] for idx in similar_indices])
                y_weighted = np.mean([circles[idx]['y'] for idx in similar_indices])
                r_weighted = np.mean([circles[idx]['radius'] for idx in similar_indices])

            # 使用最高置信度
            max_confidence = max(circles[idx]['confidence'] for idx in similar_indices)

            merged.append({
                'x': x_weighted,
                'y': y_weighted,
                'radius': r_weighted,
                'confidence': max_confidence,
                'method': 'merged',
                'merged_count': len(similar_indices)
            })

        return merged

    def calculate_actual_error(self, pixel_error: float,
                               pixel_to_mm: float = None) -> float:
        """
        将像素偏差转换为实际毫米偏差
        Args:
            pixel_error: 像素偏差
            pixel_to_mm: 像素到毫米转换系数（如果为None则使用默认值）

        Returns:
            float: 实际偏差（毫米）
        """
        if pixel_to_mm is None:
            pixel_to_mm = self.concentricity_params['pixel_to_mm_ratio']

        if pixel_error < 0:
            logger.warning(f"像素偏差为负值: {pixel_error}")
            pixel_error = abs(pixel_error)

        return pixel_error * pixel_to_mm

    def set_calibration_ratio(self, pixel_to_mm: float):
        """
        设置像素到毫米的转换系数
        Args:
            pixel_to_mm: 像素到毫米转换系数
        """
        if pixel_to_mm <= 0:
            logger.error(f"无效的像素到毫米系数: {pixel_to_mm}")
            raise ValueError(f"像素到毫米系数必须大于0: {pixel_to_mm}")

        self.concentricity_params['pixel_to_mm_ratio'] = pixel_to_mm
        logger.info(f"设置像素到毫米转换系数: {pixel_to_mm}")

    def validate_concentricity_result(self, result: Dict) -> bool:
        """
        验证同心度检测结果的合理性
        Args:
            result: 检测结果

        Returns:
            bool: 结果是否合理
        """
        if not result.get('success', False):
            return False

        # 检查外筒信息
        outer_circle = result.get('outer_circle')
        if not outer_circle:
            return False

        # 检查螺纹杆中心
        inner_center = result.get('inner_center')
        if not inner_center:
            return False

        # 检查半径合理性
        radius = outer_circle.get('radius', 0)
        if radius < self.concentricity_params['outer_min_radius'] or \
                radius > self.concentricity_params['outer_max_radius']:
            logger.warning(f"外筒半径超出合理范围: {radius}")
            return False

        # 检查偏差是否过大（超过半径的50%）
        pixel_error = result.get('pixel_error', 0)
        if pixel_error > radius * 0.5:
            logger.warning(f"同心度偏差过大: {pixel_error} > {radius * 0.5}")
            return False

        return True

    def get_detection_statistics(self) -> Dict:
        """
        获取检测统计信息
        Returns:
            Dict: 统计信息
        """
        stats = {
            'last_detection_time': None,
            'detection_count': 0,
            'concentricity_detection_count': 0,
            'last_concentricity_error': None,
            'average_error': None
        }

        if self.last_detection:
            stats['detection_count'] = len(self.last_detection)

        if self.last_concentricity_result:
            stats['last_detection_time'] = self.last_concentricity_result.get('timestamp')
            stats['concentricity_detection_count'] = 1
            stats['last_concentricity_error'] = self.last_concentricity_result.get('pixel_error')

        return stats

    def detect_outer_tube(self, gray_image: np.ndarray) -> List[Dict]:
        """
        专门检测外筒（大圆）
        Args:
            gray_image: 灰度图像

        Returns:
            List[Dict]: 外筒圆信息
        """
        # 调整参数以检测大圆
        params = {
            'dp': 1.0,
            'minDist': 300,  # 外筒较大，需要更大的最小距离
            'param1': 100,
            'param2': 25,  # 降低阈值以检测较弱的圆
            'minRadius': 100,
            'maxRadius': 500
        }

        # 预处理：使用更强的模糊
        blurred = cv2.GaussianBlur(gray_image, (7, 7), 3)

        # 边缘检测
        edges = cv2.Canny(blurred, 30, 100)

        # 霍夫变换检测大圆
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=params['dp'],
            minDist=params['minDist'],
            param1=params['param1'],
            param2=params['param2'],
            minRadius=params['minRadius'],
            maxRadius=params['maxRadius']
        )

        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # 为大圆设置更高的初始置信度
                confidence = self._calculate_circle_confidence(gray_image, x, y, r) * 1.2
                confidence = min(confidence, 1.0)  # 限制在0-1之间

                result.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': float(r),
                    'confidence': confidence,
                    'method': 'outer_tube_hough',
                    'is_large_circle': r > 150  # 标记为大圆
                })

        return result

    def detect_threaded_rod_center(self, gray_image: np.ndarray,
                                   outer_circle: Dict = None,
                                   roi_expand: int = 20) -> Optional[Dict]:
        """
        检测螺纹杆（六边形）的中心
        Args:
            gray_image: 灰度图像
            outer_circle: 外筒圆信息 (如果有的话，用于ROI裁剪)
            roi_expand: ROI扩展像素

        Returns:
            Dict: 螺纹杆中心信息
        """
        try:
            # 如果有外筒信息，创建ROI掩码
            if outer_circle:
                mask = np.zeros_like(gray_image)
                cx, cy, r = int(outer_circle['x']), int(outer_circle['y']), int(outer_circle['radius'])
                # 创建外筒内部的ROI（向内缩小一点避免边缘）
                cv2.circle(mask, (cx, cy), r - roi_expand, 255, -1)
                roi_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            else:
                roi_image = gray_image

            # 增强对比度，突出六边形结构
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(roi_image)

            # 二值化
            _, binary = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 形态学操作去除中间小孔干扰
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("未检测到螺纹杆轮廓")
                return None

            # 选择面积最大的轮廓（六边形）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < self.concentricity_params['thread_min_area']:  # 使用配置参数
                logger.warning(f"螺纹杆轮廓面积太小: {area} < {self.concentricity_params['thread_min_area']}")
                return None

            # 计算形心（中心点）
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 计算多边形逼近，确定是六边形
            perimeter = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

            # 计算六边形的特征
            is_hexagon = len(approx) == 6

            return {
                'x': float(cx),
                'y': float(cy),
                'area': area,
                'is_hexagon': is_hexagon,
                'vertex_count': len(approx),
                'confidence': 0.9 if is_hexagon else 0.6,
                'method': 'threaded_rod_contour'
            }

        except Exception as e:
            logger.error(f"螺纹杆中心检测失败: {e}")
            return None

    def detect_concentricity_pair(self, image: np.ndarray) -> Dict:
        """
        检测外筒与螺纹杆的同心度对
        Args:
            image: 输入图像

        Returns:
            Dict: 包含外筒和螺纹杆的检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 1. 检测外筒（大圆）
        outer_circles = self.detect_outer_tube(gray)

        if not outer_circles:
            logger.error("未检测到外筒圆")
            return {'success': False, 'error': '未检测到外筒圆'}

        # 选择置信度最高的外筒圆
        outer_circle = max(outer_circles, key=lambda x: x['confidence'])

        # 2. 检测螺纹杆中心
        inner_center = self.detect_threaded_rod_center(gray, outer_circle)

        if not inner_center:
            logger.error("未检测到螺纹杆中心")
            return {'success': False, 'error': '未检测到螺纹杆中心'}

        # 3. 计算同心度偏差（像素）
        dx = outer_circle['x'] - inner_center['x']
        dy = outer_circle['y'] - inner_center['y']
        pixel_error = math.sqrt(dx ** 2 + dy ** 2)

        # 4. 计算实际偏差（毫米）← 添加这一行
        error_mm = self.calculate_actual_error(pixel_error)

        # 5. 计算相对偏差（相对于外筒半径）
        relative_error = (pixel_error / outer_circle['radius']) * 100

        # 6. 判断是否合格 ← 添加这一行
        is_qualified = error_mm <= self.concentricity_params['quality_threshold_mm']

        # 7. 构建返回结果
        result = {
            'success': True,
            'outer_circle': outer_circle,
            'inner_center': inner_center,
            'pixel_error': pixel_error,
            'error_mm': error_mm,  # ← 添加实际毫米偏差
            'relative_error_percent': relative_error,
            'dx': dx,
            'dy': dy,
            'is_qualified': is_qualified,  # ← 添加合格状态
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }

        # 缓存结果
        self.last_concentricity_result = result

        return result

    def draw_concentricity_result(self, image: np.ndarray,
                                  result: Dict) -> np.ndarray:
        """
        绘制同心度检测结果
        Args:
            image: 原始图像
            result: detect_concentricity_pair的返回结果

        Returns:
            np.ndarray: 绘制后的图像
        """
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()

        if not result.get('success', False):
            cv2.putText(output, "检测失败", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return output

        outer = result['outer_circle']
        inner = result['inner_center']

        # 绘制外筒圆（红色）
        cx, cy, r = int(outer['x']), int(outer['y']), int(outer['radius'])
        cv2.circle(output, (cx, cy), r, (0, 0, 255), 3)
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

        # 绘制螺纹杆中心（绿色）
        ix, iy = int(inner['x']), int(inner['y'])
        cv2.circle(output, (ix, iy), 10, (0, 255, 0), 3)
        cv2.circle(output, (ix, iy), 3, (255, 0, 0), -1)

        # 绘制连接线
        cv2.line(output, (cx, cy), (ix, iy), (255, 255, 0), 2)

        # 计算偏移角度
        angle = math.degrees(math.atan2(iy - cy, ix - cx))

        # 添加文字信息 - 直接从result中获取error_mm ← 修改这里
        error_mm = result.get('error_mm', 0)  # 直接从result获取
        is_qualified = result.get('is_qualified', False)  # 直接从result获取

        status_color = (0, 255, 0) if is_qualified else (0, 0, 255)
        status_text = "合格" if is_qualified else "不合格"

        # 信息列表
        info_lines = [
            f"外筒: ({cx}, {cy}), R={r}px",
            f"螺纹杆: ({ix}, {iy})",
            f"像素偏差: {result['pixel_error']:.2f}px",
            f"实际偏差: {error_mm:.3f}mm",
            f"相对偏差: {result['relative_error_percent']:.2f}%",
            f"偏移角度: {angle:.1f}°",
            f"状态: {status_text}"
        ]

        # 绘制信息
        y_offset = 30
        for line in info_lines:
            cv2.putText(output, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # 在圆心位置显示十字线
        cv2.line(output, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(output, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        return output

    def update_parameters(self, **kwargs):
        """
        更新检测参数

        Args:
            **kwargs: 参数键值对
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                logger.debug(f"更新通用参数: {key} = {value}")
            # 添加对同心度参数的更新
            elif key in self.concentricity_params:
                self.concentricity_params[key] = value
                logger.debug(f"更新同心度参数: {key} = {value}")
            else:
                logger.warning(f"未知参数: {key}")



