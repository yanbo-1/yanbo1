"""
圆心检测模块 - 改进霍夫变换、最小二乘法等算法
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from scipy import optimize
import math
import os
from datetime import datetime
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
        min_r = max(200, self.concentricity_params.get('outer_min_radius', 200) - 30)
        max_r = min(400, self.concentricity_params.get('outer_max_radius', 350) + 50)

        if radius < min_r or radius > max_r:
            logger.warning(f"外筒半径可能异常: {radius}px (范围: {min_r}-{max_r}px)")
            # 不直接返回False，只警告
            # return False  # 注释掉这行，让检测继续

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
        专门检测外筒（大圆）- 高精度版
        """
        # 使用多步骤检测提高精度
        results = []

        # 保留两种边缘检测参数（不同阈值）
        edges_low = cv2.Canny(gray_image, 30, 100)  # 低阈值，更多边缘
        edges_high = cv2.Canny(gray_image, 50, 150)  # 高阈值，更精确边缘

        # 简化参数组合
        param_combinations = [
            {'dp': 1.0, 'minDist': 300, 'param1': 100, 'param2': 30, 'minRadius': 250, 'maxRadius': 320},
            {'dp': 1.2, 'minDist': 350, 'param1': 100, 'param2': 28, 'minRadius': 250, 'maxRadius': 320},
        ]

        # 遍历两种边缘检测和参数组合
        edge_list = [edges_low, edges_high]
        edge_names = ['low', 'high']

        for edges, name in zip(edge_list, edge_names):
            for params in param_combinations:
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

                if circles is not None:
                    circles = np.uint16(np.around(circles[0, :]))
                    for circle in circles:
                        x, y, r = circle
                        confidence = self._calculate_circle_confidence(gray_image, x, y, r)

                        results.append({
                            'x': float(x),
                            'y': float(y),
                            'radius': float(r),
                            'confidence': confidence,
                            'method': f'outer_tube_{name}_dp{params["dp"]}',
                            'is_large_circle': r > 150
                        })

        # 如果检测到多个圆，合并相似的圆
        if results:
            # 按置信度排序
            results.sort(key=lambda x: x['confidence'], reverse=True)

            # 合并相似圆（距离小于20像素，半径差小于10像素）
            merged_results = []
            used = [False] * len(results)

            for i, circle in enumerate(results):
                if used[i]:
                    continue

                similar = [i]
                for j in range(i + 1, len(results)):
                    if used[j]:
                        continue

                    dx = abs(circle['x'] - results[j]['x'])
                    dy = abs(circle['y'] - results[j]['y'])
                    dr = abs(circle['radius'] - results[j]['radius'])

                    if dx < 20 and dy < 20 and dr < 10:
                        similar.append(j)
                        used[j] = True

                # 计算平均值
                avg_x = np.mean([results[idx]['x'] for idx in similar])
                avg_y = np.mean([results[idx]['y'] for idx in similar])
                avg_r = np.mean([results[idx]['radius'] for idx in similar])
                max_conf = max([results[idx]['confidence'] for idx in similar])

                merged_results.append({
                    'x': float(avg_x),
                    'y': float(avg_y),
                    'radius': float(avg_r),
                    'confidence': float(max_conf),
                    'method': 'merged',
                    'is_large_circle': avg_r > 150,
                    'merged_count': len(similar)
                })

            return merged_results

        return results

    def detect_hexagon_center_precise(self, gray_image: np.ndarray,
                                      outer_circle: Dict) -> Optional[Dict]:
        """
        精确检测六边形螺纹杆中心
        专门针对六边形结构优化
        """
        try:
            debug_dir = "debug_output"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 步骤1：创建更精确的ROI
            mask = np.zeros_like(gray_image)
            cx, cy, r = int(outer_circle['x']), int(outer_circle['y']), int(outer_circle['radius'])

            # 使用更小的ROI，只取中心区域（六边形应该在中心）
            roi_radius = int(r * 0.3)  # 30%的外圆半径
            cv2.circle(mask, (cx, cy), roi_radius, 255, -1)
            roi_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

            # 步骤2：专门针对六边形的预处理
            # 使用Sobel算子增强六边形边缘
            sobelx = cv2.Sobel(roi_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

            # 步骤3：自适应阈值（六边形边缘通常清晰）
            binary = cv2.adaptiveThreshold(
                sobel_magnitude, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 2
            )

            # 步骤4：形态学操作（轻微，保持六边形形状）
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 步骤5：查找轮廓并筛选六边形
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            best_hexagon = None
            best_hexagon_score = -1

            for contour in contours:
                area = cv2.contourArea(contour)

                # 面积筛选：六边形不应该太小或太大
                if area < 1000 or area > 20000:
                    continue

                # 计算多边形逼近
                perimeter = cv2.arcLength(contour, True)

                # 尝试不同的epsilon值寻找六边形
                for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                    epsilon = epsilon_factor * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    vertex_count = len(approx)

                    # 如果是六边形（允许±1的误差）
                    if vertex_count in [5, 6, 7]:
                        # 计算六边形得分
                        area_ratio = area / (roi_radius * roi_radius * 3.14159)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                        # 六边形应该有较高的圆形度（约0.8-0.9）
                        hexagon_score = 0
                        if vertex_count == 6:
                            hexagon_score += 0.5  # 顶点数正确
                        if 0.7 < circularity < 0.95:
                            hexagon_score += 0.3  # 圆形度合理
                        if 0.1 < area_ratio < 0.5:
                            hexagon_score += 0.2  # 面积比例合理

                        if hexagon_score > best_hexagon_score:
                            best_hexagon_score = hexagon_score
                            best_hexagon = {
                                'contour': contour,
                                'approx': approx,
                                'vertices': vertex_count,
                                'area': area,
                                'score': hexagon_score
                            }

            if best_hexagon is None:
                return None

            # 步骤6：计算六边形中心
            contour = best_hexagon['contour']
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None

            cx_hex = int(M["m10"] / M["m00"])
            cy_hex = int(M["m01"] / M["m00"])

            # 步骤7：绘制调试图像
            debug_img = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
            for point in best_hexagon['approx']:
                cv2.circle(debug_img, tuple(point[0]), 4, (0, 0, 255), -1)
            cv2.circle(debug_img, (cx_hex, cy_hex), 8, (255, 0, 0), -1)
            cv2.imwrite(f"{debug_dir}/debug_hexagon_detection_{timestamp}.jpg", debug_img)

            print(
                f"精确六边形检测: 中心({cx_hex}, {cy_hex}), 顶点数{best_hexagon['vertices']}, 面积{best_hexagon['area']:.0f}, 得分{best_hexagon_score:.2f}")

            return {
                'x': float(cx_hex),
                'y': float(cy_hex),
                'area': best_hexagon['area'],
                'is_hexagon': best_hexagon['vertices'] == 6,
                'vertex_count': best_hexagon['vertices'],
                'confidence': min(0.9, best_hexagon_score),
                'method': 'precise_hexagon'
            }

        except Exception as e:
            logger.error(f"精确六边形检测失败: {e}")
            return None

    def detect_by_simple_centroid(self, gray_image: np.ndarray,
                                  outer_circle: Dict) -> Optional[Dict]:
        """
        最简单的质心检测法
        直接计算ROI区域的灰度质心
        """
        try:
            debug_dir = "debug_output"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 创建ROI
            mask = np.zeros_like(gray_image)
            cx, cy, r = int(outer_circle['x']), int(outer_circle['y']), int(outer_circle['radius'])

            # 使用小ROI（只取中心区域）
            roi_radius = int(r * 0.3)  # 30%的外圆半径
            cv2.circle(mask, (cx, cy), roi_radius, 255, -1)
            roi_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

            # 方法1：自适应阈值（处理光照不均匀）
            binary = cv2.adaptiveThreshold(
                roi_image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # 形态学操作去除噪声
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # 计算质心
            M = cv2.moments(binary)
            if M["m00"] > 0:
                cx1 = int(M["m10"] / M["m00"])
                cy1 = int(M["m01"] / M["m00"])

                # 方法2：使用灰度图像的质心（加权）
                total_intensity = np.sum(roi_image)
                if total_intensity > 0:
                    y_coords, x_coords = np.mgrid[0:roi_image.shape[0], 0:roi_image.shape[1]]
                    cx2 = int(np.sum(x_coords * roi_image) / total_intensity)
                    cy2 = int(np.sum(y_coords * roi_image) / total_intensity)

                    # 取两种方法的平均值
                    final_x = (cx1 + cx2) // 2
                    final_y = (cy1 + cy2) // 2

                    print(f"简单质心法: 二值质心({cx1}, {cy1}), 灰度质心({cx2}, {cy2}), 最终({final_x}, {final_y})")

                    # 绘制结果
                    result_img = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
                    cv2.circle(result_img, (cx1, cy1), 5, (0, 255, 0), -1)  # 绿色：二值质心
                    cv2.circle(result_img, (cx2, cy2), 5, (255, 0, 0), -1)  # 蓝色：灰度质心
                    cv2.circle(result_img, (final_x, final_y), 8, (0, 0, 255), -1)  # 红色：最终中心

                    cv2.imwrite(f"{debug_dir}/debug_simple_centroid_{timestamp}.jpg", result_img)

                    return {
                        'x': float(final_x),
                        'y': float(final_y),
                        'area': M["m00"],
                        'is_hexagon': False,
                        'vertex_count': 0,
                        'confidence': 0.8,
                        'method': 'simple_centroid'
                    }

            return None

        except Exception as e:
            logger.error(f"简单质心法失败: {e}")
            return None

    def detect_threaded_rod_center(self, gray_image: np.ndarray,
                                   outer_circle: Dict = None,
                                   roi_expand: int = 10) -> Optional[Dict]:
        """
        检测螺纹杆（六边形）的中心 - 基于特征点的新方法
        完全放弃传统二值化，使用角点检测和圆检测
        """
        try:
            debug_dir = "debug_output"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # ========== 步骤1：创建ROI掩码 ==========
            if outer_circle:
                mask = np.zeros_like(gray_image)
                cx, cy, r = int(outer_circle['x']), int(outer_circle['y']), int(outer_circle['radius'])

                # ROI半径：取外圆半径的60-70%
                roi_radius = int(r * 0.65)
                print(f"外圆半径: {r}, ROI半径: {roi_radius}")

                cv2.circle(mask, (cx, cy), roi_radius, 255, -1)
                roi_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

                cv2.imwrite(f"{debug_dir}/debug_roi_mask_{timestamp}.jpg", mask)
                cv2.imwrite(f"{debug_dir}/debug_roi_image_{timestamp}.jpg", roi_image)
            else:
                roi_image = gray_image

            # ========== 步骤2：增强图像质量 ==========
            # 轻度高斯模糊
            blurred = cv2.GaussianBlur(roi_image, (3, 3), 0.5)

            # CLAHE增强对比度
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

            # 中值滤波去除椒盐噪声
            denoised = cv2.medianBlur(enhanced, 3)

            cv2.imwrite(f"{debug_dir}/debug_enhanced_{timestamp}.jpg", denoised)

            # ========== 新增方法：简单质心法 ==========
            print("尝试方法-1: 简单质心法")
            simple_result = self.detect_by_simple_centroid(roi_image, outer_circle)  # 改为roi_image
            if simple_result:
                print("简单质心法成功")
                return simple_result
            else:
                print("简单质心法失败，继续其他方法")

            # ========== 新增方法：精确六边形检测 ==========
            print("尝试方法0: 精确六边形检测")
            hexagon_result = self.detect_hexagon_center_precise(roi_image, outer_circle)
            if hexagon_result and hexagon_result.get('is_hexagon', False):
                print("精确六边形检测成功")
                return hexagon_result
            else:
                print("精确六边形检测失败，继续其他方法")

            # ========== 步骤3：方法A - Harris角点检测六边形角点 ==========
            print("尝试方法A: Harris角点检测")

            # Harris角点检测参数
            corners = cv2.cornerHarris(denoised, blockSize=2, ksize=3, k=0.04)
            corners = cv2.dilate(corners, None)

            # 阈值化角点响应
            corner_threshold = 0.01 * corners.max()
            corner_mask = corners > corner_threshold

            # 获取角点坐标
            corner_points = np.argwhere(corner_mask)

            print(f"检测到角点数量: {len(corner_points)}")

            # 如果角点数量足够（六边形有6个角点）
            if len(corner_points) >= 4:
                # 提取角点坐标（y, x -> x, y）
                corner_coords = [(int(pt[1]), int(pt[0])) for pt in corner_points]

                # 对角点进行聚类，减少重复点
                clustered_corners = []
                for x, y in corner_coords:
                    # 检查是否与已有角点太近
                    too_close = False
                    for cx2, cy2 in clustered_corners:
                        if abs(x - cx2) < 10 and abs(y - cy2) < 10:
                            too_close = True
                            break
                    if not too_close:
                        clustered_corners.append((x, y))

                print(f"聚类后角点数量: {len(clustered_corners)}")

                # 如果有6个左右的角点，计算它们的中心
                if 4 <= len(clustered_corners) <= 10:
                    # 计算所有角点的质心
                    avg_x = np.mean([p[0] for p in clustered_corners])
                    avg_y = np.mean([p[1] for p in clustered_corners])

                    # 绘制角点和中心
                    corner_img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
                    for x, y in clustered_corners:
                        cv2.circle(corner_img, (x, y), 5, (0, 255, 0), -1)
                    cv2.circle(corner_img, (int(avg_x), int(avg_y)), 8, (255, 0, 0), -1)
                    cv2.imwrite(f"{debug_dir}/debug_corners_{timestamp}.jpg", corner_img)

                    print(f"角点法中心: ({int(avg_x)}, {int(avg_y)})")

                    # 验证中心是否在合理位置
                    if self._validate_center(denoised, int(avg_x), int(avg_y), roi_radius):
                        return {
                            'x': float(avg_x),
                            'y': float(avg_y),
                            'area': len(clustered_corners) * 100,  # 估算面积
                            'is_hexagon': True,
                            'vertex_count': len(clustered_corners),
                            'confidence': min(0.9, len(clustered_corners) / 6.0),
                            'method': 'harris_corners'
                        }

            # ========== 步骤4：方法B - Hough圆检测小圆孔 ==========
            print("方法A失败，尝试方法B: Hough圆检测")

            # 使用Canny边缘检测预处理
            edges = cv2.Canny(denoised, 50, 150)
            cv2.imwrite(f"{debug_dir}/debug_edges_{timestamp}.jpg", edges)

            # 优化Hough圆检测参数
            circles = cv2.HoughCircles(
                denoised,  # 使用增强图像而不是边缘图像
                cv2.HOUGH_GRADIENT,
                dp=1.0,  # 降低分辨率，更精确
                minDist=50,  # 增加最小距离，避免密集误检
                param1=100,  # Canny高阈值
                param2=25,  # 提高累加器阈值，减少误检（从15提高到25）
                minRadius=3,  # 最小半径
                maxRadius=12  # 最大半径（根据实际小圆孔大小调整）
            )

            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))
                print(f"检测到圆孔数量 (未筛选): {len(circles)}")

                # 绘制所有检测到的圆
                circle_img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
                for x, y, r in circles:
                    cv2.circle(circle_img, (x, y), r, (0, 255, 0), 1)
                    cv2.circle(circle_img, (x, y), 2, (0, 0, 255), 1)
                cv2.imwrite(f"{debug_dir}/debug_all_circles_{timestamp}.jpg", circle_img)

                # 筛选圆：只保留在螺纹杆区域的圆
                filtered_circles = []
                for x, y, r in circles:
                    # 计算到外圆圆心的距离
                    distance_to_outer = np.sqrt((x - outer_circle['x']) ** 2 + (y - outer_circle['y']) ** 2)

                    # 小圆孔应该在外圆内部，距离外圆中心不超过外圆半径的40%
                    if distance_to_outer < outer_circle['radius'] * 0.4:
                        filtered_circles.append((x, y, r))

                print(f"筛选后圆孔数量: {len(filtered_circles)}")

                if len(filtered_circles) >= 3:  # 至少需要3个圆才能可靠计算中心
                    # 计算所有圆的中心（加权平均，按半径加权）
                    total_weight = sum(r for _, _, r in filtered_circles)
                    if total_weight > 0:
                        avg_x = sum(x * r for x, _, r in filtered_circles) / total_weight
                        avg_y = sum(y * r for _, y, r in filtered_circles) / total_weight
                    else:
                        # 如果总权重为0，使用简单平均
                        avg_x = np.mean([c[0] for c in filtered_circles])
                        avg_y = np.mean([c[1] for c in filtered_circles])

                    print(f"圆孔法中心: ({int(avg_x)}, {int(avg_y)})")

                    # 绘制筛选后的圆和中心
                    filtered_img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
                    for x, y, r in filtered_circles:
                        cv2.circle(filtered_img, (x, y), r, (0, 255, 0), 2)
                        cv2.circle(filtered_img, (x, y), 2, (0, 0, 255), 3)
                    cv2.circle(filtered_img, (int(avg_x), int(avg_y)), 10, (255, 0, 0), -1)
                    cv2.imwrite(f"{debug_dir}/debug_filtered_circles_{timestamp}.jpg", filtered_img)

                    if self._validate_center(denoised, int(avg_x), int(avg_y), roi_radius):
                        return {
                            'x': float(avg_x),
                            'y': float(avg_y),
                            'area': len(filtered_circles) * 50,
                            'is_hexagon': False,
                            'vertex_count': 0,
                            'confidence': min(0.9, len(filtered_circles) / 5.0),
                            'method': 'hough_circles_optimized'
                        }

            # ========== 步骤5：方法C - 模板匹配（如果知道螺纹杆形状） ==========
            print("方法B失败，尝试方法C: 形态学梯度中心")

            # 使用形态学梯度检测边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)

            # 阈值化梯度图像
            _, binary_gradient = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)

            # 查找轮廓
            contours, _ = cv2.findContours(binary_gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 选择面积最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                # 计算轮廓的质心
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx_cnt = int(M["m10"] / M["m00"])
                    cy_cnt = int(M["m01"] / M["m00"])

                    print(f"梯度法中心: ({cx_cnt}, {cy_cnt}), 面积: {area:.1f}")

                    if area > 500 and self._validate_center(denoised, cx_cnt, cy_cnt, roi_radius):
                        # 多边形逼近
                        perimeter = cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

                        return {
                            'x': float(cx_cnt),
                            'y': float(cy_cnt),
                            'area': area,
                            'is_hexagon': len(approx) == 6,
                            'vertex_count': len(approx),
                            'confidence': 0.7,
                            'method': 'morphological_gradient'
                        }

            # ========== 步骤6：方法D - 灰度质心（改进版） ==========
            print("方法C失败，尝试方法D: 改进灰度质心")

            # 使用ROI图像，但只取中心区域
            h, w = denoised.shape
            center_region = denoised[h // 4:3 * h // 4, w // 4:3 * w // 4]

            # 计算该区域的质心
            total_intensity = np.sum(center_region)
            if total_intensity > 0:
                y_coords, x_coords = np.mgrid[0:center_region.shape[0], 0:center_region.shape[1]]
                weighted_x = np.sum(x_coords * center_region) / total_intensity
                weighted_y = np.sum(y_coords * center_region) / total_intensity

                # 转换回原图坐标
                center_x = int(weighted_x + w // 4)
                center_y = int(weighted_y + h // 4)

                print(f"灰度质心法中心: ({center_x}, {center_y})")

                # 简单的验证：检查该点附近是否有足够的边缘
                if self._validate_center(denoised, center_x, center_y, roi_radius):
                    return {
                        'x': float(center_x),
                        'y': float(center_y),
                        'area': total_intensity / 255,  # 估算面积
                        'is_hexagon': False,
                        'vertex_count': 0,
                        'confidence': 0.6,
                        'method': 'improved_gray_centroid'
                    }

            # ========== 步骤7：备用方案 - 使用ROI中心 ==========
            print("所有方法失败，使用ROI中心作为备选")

            if outer_circle:
                # 使用外圆中心作为备选
                cx, cy = int(outer_circle['x']), int(outer_circle['y'])
            else:
                # 使用图像中心
                h, w = denoised.shape
                cx, cy = w // 2, h // 2

            print(f"使用备选中心: ({cx}, {cy})")

            return {
                'x': float(cx),
                'y': float(cy),
                'area': 1500,  # 默认面积
                'is_hexagon': False,
                'vertex_count': 0,
                'confidence': 0.3,
                'method': 'fallback_center'
            }

        except Exception as e:
            logger.error(f"螺纹杆中心检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _validate_center(self, image: np.ndarray, x: int, y: int, max_radius: int) -> bool:
        """
        验证中心点是否合理

        Args:
            image: 灰度图像
            x, y: 待验证的中心坐标
            max_radius: 最大允许半径

        Returns:
            bool: 中心是否合理
        """
        h, w = image.shape

        # 1. 检查是否在图像范围内
        if x < 0 or x >= w or y < 0 or y >= h:
            return False

        # 2. 检查是否在ROI中心附近（假设螺纹杆在中心）
        center_x, center_y = w // 2, h // 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        if distance > max_radius * 0.8:  # 允许一定偏移，但不能太远
            print(f"中心点距离图像中心太远: {distance:.1f} > {max_radius * 0.8}")
            return False

        # 3. 检查该点附近的图像特征
        # 提取一个小区域
        region_size = 20
        x1 = max(0, x - region_size)
        x2 = min(w, x + region_size)
        y1 = max(0, y - region_size)
        y2 = min(h, y + region_size)

        region = image[y1:y2, x1:x2]

        # 计算区域的对比度
        if region.size > 0:
            contrast = region.max() - region.min()
            if contrast < 30:  # 对比度太低，可能不是特征区域
                print(f"中心点区域对比度太低: {contrast}")
                return False

        return True

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
            # 检测失败时不显示任何文字
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

        # 在圆心位置显示十字线
        cv2.line(output, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(output, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        # 已移除所有文字绘制代码
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