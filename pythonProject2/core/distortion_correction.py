"""
透视畸变矫正模块 - 基于相机标定的图像矫正
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
import os
import yaml

logger = logging.getLogger(__name__)


class DistortionCorrector:
    """透视畸变矫正器类"""

    def __init__(self, calibration_file: str = None):
        """
        初始化畸变矫正器

        Args:
            calibration_file: 相机标定文件路径
        """
        self.camera_matrix = None  # 相机内参矩阵
        self.dist_coeffs = None  # 畸变系数
        self.map1 = None  # 矫正映射1
        self.map2 = None  # 矫正映射2
        self.is_calibrated = False  # 是否已标定

        self.calibration_params = {
            'chessboard_size': (9, 6),  # 棋盘格角点数量
            'square_size': 25.0,  # 棋盘格方格大小(mm)
            'calibration_images': 15,  # 标定图像数量
            'calibration_flags': 0,  # 标定标志
        }

        # 加载标定文件
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)

    def calibrate_camera(self, image_paths: List[str],
                         chessboard_size: Tuple[int, int] = None,
                         square_size: float = None) -> Dict:
        """
        相机标定

        Args:
            image_paths: 标定图像路径列表
            chessboard_size: 棋盘格角点数量 (列, 行)
            square_size: 方格实际尺寸(mm)

        Returns:
            Dict: 标定结果
        """
        if chessboard_size:
            self.calibration_params['chessboard_size'] = chessboard_size
        if square_size:
            self.calibration_params['square_size'] = square_size

        pattern_size = self.calibration_params['chessboard_size']

        # 准备对象点和图像点
        obj_points = []  # 3D点
        img_points = []  # 2D点
        image_size = None

        # 生成对象点 (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= self.calibration_params['square_size']

        success_count = 0

        for i, image_path in enumerate(image_paths):
            if i >= self.calibration_params['calibration_images']:
                break

            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"无法读取图像: {image_path}")
                continue

            if image_size is None:
                image_size = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )

            if ret:
                success_count += 1

                # 提高角点检测精度
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                obj_points.append(objp)
                img_points.append(corners_refined)

                # 可视化角点（可选）
                cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
                logger.debug(f"图像 {image_path} 角点检测成功")
            else:
                logger.warning(f"图像 {image_path} 未检测到棋盘格")

        if success_count < 5:
            logger.error(f"成功检测的标定图像不足: {success_count}/5")
            return {"success": False, "error": "标定图像不足"}

        # 相机标定
        logger.info(f"开始相机标定，使用 {success_count} 张图像...")

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None,
            flags=self.calibration_params['calibration_flags']
        )

        if not ret:
            logger.error("相机标定失败")
            return {"success": False, "error": "标定计算失败"}

        # 保存标定结果
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.is_calibrated = True

        # 计算重投影误差
        mean_error = self._calculate_reprojection_error(obj_points, img_points, rvecs, tvecs)

        logger.info(f"相机标定成功！平均重投影误差: {mean_error:.2f} 像素")

        # 生成矫正映射
        self._generate_undistort_maps(image_size)

        result = {
            "success": True,
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "image_size": image_size,
            "reprojection_error": mean_error,
            "calibration_images": success_count,
            "chessboard_size": pattern_size,
            "square_size": self.calibration_params['square_size']
        }

        return result

    def _calculate_reprojection_error(self, obj_points: List[np.ndarray],
                                      img_points: List[np.ndarray],
                                      rvecs: List[np.ndarray],
                                      tvecs: List[np.ndarray]) -> float:
        """
        计算重投影误差

        Args:
            obj_points: 对象点
            img_points: 图像点
            rvecs: 旋转向量
            tvecs: 平移向量

        Returns:
            float: 平均重投影误差
        """
        total_error = 0
        total_points = 0

        for i in range(len(obj_points)):
            # 投影点
            img_points2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )

            # 计算误差
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
            total_points += 1

        return total_error / total_points

    def _generate_undistort_maps(self, image_size: Tuple[int, int]):
        """生成矫正映射"""
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs,
            image_size, 1, image_size
        )

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None,
            new_camera_matrix, image_size, cv2.CV_32FC1
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        矫正图像畸变

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 矫正后的图像
        """
        if not self.is_calibrated:
            logger.warning("相机未标定，返回原始图像")
            return image

        if self.map1 is None or self.map2 is None:
            logger.error("矫正映射未生成")
            return image

        # 使用映射矫正图像
        undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

        return undistorted

    def correct_perspective(self, image: np.ndarray,
                            src_points: Optional[np.ndarray] = None,
                            dst_points: Optional[np.ndarray] = None,
                            output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        透视变换矫正

        Args:
            image: 输入图像
            src_points: 源图像中的四个点 (4x2)
            dst_points: 目标图像中的四个点 (4x2)
            output_size: 输出图像尺寸

        Returns:
            np.ndarray: 矫正后的图像
        """
        if src_points is None or dst_points is None:
            # 自动检测矫正点（示例：基于特征点）
            return self._auto_perspective_correction(image)

        # 确保点数量正确
        if len(src_points) != 4 or len(dst_points) != 4:
            logger.error("需要4个源点和4个目标点")
            return image

        # 转换为float32
        src_pts = src_points.astype(np.float32)
        dst_pts = dst_points.astype(np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        if output_size is None:
            output_size = (image.shape[1], image.shape[0])

        # 应用透视变换
        corrected = cv2.warpPerspective(image, M, output_size)

        return corrected

    def _auto_perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """
        自动透视矫正（简化版）

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 矫正后的图像
        """
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 多边形逼近
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) != 4:
            logger.warning(f"未检测到四边形，检测到 {len(approx)} 个点")
            return image

        # 排序四个点（左上，右上，右下，左下）
        points = approx.reshape(4, 2)
        rect = self._order_points(points)

        # 计算目标点
        width = max(
            np.linalg.norm(rect[1] - rect[0]),
            np.linalg.norm(rect[2] - rect[3])
        )
        height = max(
            np.linalg.norm(rect[2] - rect[1]),
            np.linalg.norm(rect[3] - rect[0])
        )

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # 透视变换
        M = cv2.getPerspectiveTransform(rect, dst)
        corrected = cv2.warpPerspective(image, M, (int(width), int(height)))

        return corrected

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        对四个点进行排序（左上，右上，右下，左下）

        Args:
            pts: 四个点的数组

        Returns:
            np.ndarray: 排序后的点
        """
        # 初始化坐标点
        rect = np.zeros((4, 2), dtype=np.float32)

        # 左上点总和最小，右下点总和最大
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # 右上点差值最小，左下点差值最大
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def save_calibration(self, filepath: str) -> bool:
        """
        保存标定结果

        Args:
            filepath: 保存路径

        Returns:
            bool: 保存是否成功
        """
        if not self.is_calibrated:
            logger.error("相机未标定，无法保存")
            return False

        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'calibration_params': self.calibration_params,
            'is_calibrated': self.is_calibrated,
            'calibration_date': np.datetime64('now').astype(str)
        }

        try:
            with open(filepath, 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)
            logger.info(f"标定数据已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存标定数据失败: {e}")
            return False

    def load_calibration(self, filepath: str) -> bool:
        """
        加载标定结果

        Args:
            filepath: 加载路径

        Returns:
            bool: 加载是否成功
        """
        try:
            with open(filepath, 'r') as f:
                calibration_data = yaml.safe_load(f)

            self.camera_matrix = np.array(calibration_data['camera_matrix'])
            self.dist_coeffs = np.array(calibration_data['dist_coeffs'])
            self.is_calibrated = calibration_data.get('is_calibrated', True)

            if 'calibration_params' in calibration_data:
                self.calibration_params.update(calibration_data['calibration_params'])

            # 生成矫正映射（需要图像尺寸）
            logger.info(f"标定数据已加载: {filepath}")
            return True

        except Exception as e:
            logger.error(f"加载标定数据失败: {e}")
            self.is_calibrated = False
            return False

    def estimate_pose(self, object_points: np.ndarray,
                      image_points: np.ndarray) -> Dict:
        """
        估计物体姿态

        Args:
            object_points: 物体3D点
            image_points: 图像2D点

        Returns:
            Dict: 姿态估计结果
        """
        if not self.is_calibrated:
            logger.error("相机未标定，无法估计姿态")
            return {"success": False}

        # 使用PnP算法估计姿态
        ret, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            self.camera_matrix, self.dist_coeffs
        )

        if not ret:
            return {"success": False}

        # 转换为欧拉角
        rmat, _ = cv2.Rodrigues(rvec)
        euler_angles = self._rotation_matrix_to_euler_angles(rmat)

        return {
            "success": True,
            "rotation_vector": rvec.flatten().tolist(),
            "translation_vector": tvec.flatten().tolist(),
            "rotation_matrix": rmat.tolist(),
            "euler_angles": euler_angles.tolist(),
            "distance": np.linalg.norm(tvec)
        }

    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> np.ndarray:
        """
        旋转矩阵转换为欧拉角

        Args:
            R: 旋转矩阵

        Returns:
            np.ndarray: 欧拉角 (rx, ry, rz) 弧度
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])