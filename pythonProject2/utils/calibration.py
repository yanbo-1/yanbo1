"""
相机标定工具模块
实现相机标定、畸变矫正等功能
"""

import os
import json
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class CalibrationResult:
    """相机标定结果"""
    camera_matrix: np.ndarray  # 相机内参矩阵 (3x3)
    dist_coeffs: np.ndarray  # 畸变系数 (1x5)
    rvecs: List[np.ndarray]  # 旋转向量列表
    tvecs: List[np.ndarray]  # 平移向量列表
    reprojection_error: float  # 重投影误差
    image_size: Tuple[int, int]  # 图像尺寸 (宽, 高)
    calibration_time: str  # 标定时间

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'reprojection_error': self.reprojection_error,
            'image_size': self.image_size,
            'calibration_time': self.calibration_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """从字典创建"""
        return cls(
            camera_matrix=np.array(data['camera_matrix']),
            dist_coeffs=np.array(data['dist_coeffs']),
            rvecs=[],  # 通常不保存
            tvecs=[],  # 通常不保存
            reprojection_error=data['reprojection_error'],
            image_size=tuple(data['image_size']),
            calibration_time=data['calibration_time']
        )


class CalibrationError(Exception):
    """相机标定错误"""
    pass


class CameraCalibrator:
    """相机标定器"""

    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0):
        """
        初始化相机标定器

        Args:
            checkerboard_size: 棋盘格内角点数量 (列数, 行数)
            square_size: 棋盘格方格大小 (毫米)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size

        # 准备对象点
        self.obj_points = []  # 3D点
        self.img_points = []  # 2D点

        # 标定结果
        self.calibration_result: Optional[CalibrationResult] = None
        self.is_calibrated = False

        logging.info(f"相机标定器初始化: 棋盘格大小={checkerboard_size}, 方格大小={square_size}mm")

    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        添加标定图像

        Args:
            image: 标定图像

        Returns:
            是否成功检测到角点
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(
                gray, self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_FAST_CHECK +
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                # 亚像素级角点检测
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # 生成对象点
                objp = generate_checkerboard_points(self.checkerboard_size, self.square_size)

                # 添加点
                self.obj_points.append(objp)
                self.img_points.append(corners2)

                logging.info(f"成功检测到角点，当前图像数量: {len(self.img_points)}")
                return True
            else:
                logging.warning("未检测到棋盘格角点")
                return False

        except Exception as e:
            logging.error(f"处理标定图像失败: {str(e)}")
            return False

    def calibrate(self, image_size: Tuple[int, int]) -> CalibrationResult:
        """
        执行相机标定

        Args:
            image_size: 图像尺寸 (宽, 高)

        Returns:
            标定结果
        """
        try:
            if len(self.img_points) < 3:
                raise CalibrationError(f"标定图像数量不足: {len(self.img_points)}，需要至少3张")

            logging.info(f"开始相机标定，使用{len(self.img_points)}张图像...")

            # 执行标定
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, image_size, None, None
            )

            if not ret:
                raise CalibrationError("相机标定失败")

            # 计算重投影误差
            reprojection_error = calculate_reprojection_error(
                self.obj_points, self.img_points, rvecs, tvecs,
                camera_matrix, dist_coeffs
            )

            # 创建标定结果
            self.calibration_result = CalibrationResult(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rvecs=rvecs,
                tvecs=tvecs,
                reprojection_error=reprojection_error,
                image_size=image_size,
                calibration_time=np.datetime64('now').astype(str)
            )

            self.is_calibrated = True

            logging.info(f"相机标定完成，重投影误差: {reprojection_error:.4f}像素")
            return self.calibration_result

        except Exception as e:
            logging.error(f"相机标定失败: {str(e)}")
            raise CalibrationError(f"相机标定失败: {str(e)}")

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        矫正图像畸变

        Args:
            image: 原始图像

        Returns:
            矫正后的图像
        """
        if not self.is_calibrated or self.calibration_result is None:
            raise CalibrationError("相机未标定，无法矫正畸变")

        try:
            h, w = image.shape[:2]

            # 优化相机矩阵
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.calibration_result.camera_matrix,
                self.calibration_result.dist_coeffs,
                (w, h), 1, (w, h)
            )

            # 矫正图像
            undistorted = cv2.undistort(
                image,
                self.calibration_result.camera_matrix,
                self.calibration_result.dist_coeffs,
                None,
                new_camera_matrix
            )

            # 裁剪ROI区域
            x, y, w_roi, h_roi = roi
            if w_roi > 0 and h_roi > 0:
                undistorted = undistorted[y:y + h_roi, x:x + w_roi]

            return undistorted

        except Exception as e:
            logging.error(f"图像畸变矫正失败: {str(e)}")
            raise

    def save_calibration(self, filepath: str) -> str:
        """
        保存标定结果

        Args:
            filepath: 保存路径

        Returns:
            保存的文件路径
        """
        if not self.is_calibrated or self.calibration_result is None:
            raise CalibrationError("相机未标定，无法保存结果")

        return save_calibration_results(self.calibration_result, filepath)

    def load_calibration(self, filepath: str) -> CalibrationResult:
        """
        加载标定结果

        Args:
            filepath: 标定文件路径

        Returns:
            标定结果
        """
        self.calibration_result = load_calibration_results(filepath)
        self.is_calibrated = True
        return self.calibration_result

    def get_calibration_info(self) -> Dict[str, Any]:
        """
        获取标定信息

        Returns:
            标定信息字典
        """
        if not self.is_calibrated or self.calibration_result is None:
            return {"is_calibrated": False}

        result = self.calibration_result.to_dict()
        result["is_calibrated"] = True
        result["num_calibration_images"] = len(self.img_points)

        # 添加相机内参详细信息
        camera_matrix = self.calibration_result.camera_matrix
        result["focal_length_x"] = camera_matrix[0, 0]
        result["focal_length_y"] = camera_matrix[1, 1]
        result["principal_point_x"] = camera_matrix[0, 2]
        result["principal_point_y"] = camera_matrix[1, 2]

        return result


def generate_checkerboard_points(checkerboard_size: Tuple[int, int],
                                 square_size: float) -> np.ndarray:
    """
    生成棋盘格角点的3D坐标

    Args:
        checkerboard_size: 棋盘格内角点数量 (列数, 行数)
        square_size: 棋盘格方格大小

    Returns:
        3D点坐标数组 (n, 3)
    """
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                  0:checkerboard_size[1]].T.reshape(-1, 2) * square_size
    return objp


def detect_checkerboard_corners(image: np.ndarray,
                                checkerboard_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
    """
    检测棋盘格角点

    Args:
        image: 输入图像
        checkerboard_size: 棋盘格内角点数量

    Returns:
        (是否成功, 角点坐标)
    """
    try:
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # 亚像素级角点检测
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return ret, corners if ret else None

    except Exception as e:
        logging.error(f"检测棋盘格角点失败: {str(e)}")
        return False, None


def calculate_reprojection_error(obj_points: List[np.ndarray],
                                 img_points: List[np.ndarray],
                                 rvecs: List[np.ndarray],
                                 tvecs: List[np.ndarray],
                                 camera_matrix: np.ndarray,
                                 dist_coeffs: np.ndarray) -> float:
    """
    计算重投影误差

    Args:
        obj_points: 对象点列表
        img_points: 图像点列表
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数

    Returns:
        平均重投影误差（像素）
    """
    try:
        total_error = 0
        total_points = 0

        for i in range(len(obj_points)):
            # 投影点
            img_points2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )

            # 计算误差
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)
            total_error += error * error
            total_points += len(obj_points[i])

        if total_points > 0:
            mean_error = np.sqrt(total_error / total_points)
            return mean_error
        else:
            return 0.0

    except Exception as e:
        logging.error(f"计算重投影误差失败: {str(e)}")
        return float('inf')


def save_calibration_results(calibration_result: CalibrationResult,
                             filepath: str) -> str:
    """
    保存标定结果到文件

    Args:
        calibration_result: 标定结果
        filepath: 保存路径

    Returns:
        保存的文件路径
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 转换为字典
        data = calibration_result.to_dict()

        # 保存为JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logging.info(f"标定结果已保存: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"保存标定结果失败: {str(e)}")
        raise


def load_calibration_results(filepath: str) -> CalibrationResult:
    """
    从文件加载标定结果

    Args:
        filepath: 标定文件路径

    Returns:
        标定结果
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"标定文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        calibration_result = CalibrationResult.from_dict(data)

        logging.info(f"标定结果已加载: {filepath}")
        return calibration_result

    except Exception as e:
        logging.error(f"加载标定结果失败: {str(e)}")
        raise


def undistort_single_image(image: np.ndarray,
                           calibration_file: str) -> np.ndarray:
    """
    使用标定文件矫正单张图像

    Args:
        image: 原始图像
        calibration_file: 标定文件路径

    Returns:
        矫正后的图像
    """
    try:
        # 加载标定结果
        calibration_result = load_calibration_results(calibration_file)

        # 创建标定器
        calibrator = CameraCalibrator()
        calibrator.calibration_result = calibration_result
        calibrator.is_calibrated = True

        # 矫正图像
        return calibrator.undistort_image(image)

    except Exception as e:
        logging.error(f"矫正图像失败: {str(e)}")
        raise


def estimate_pose_from_checkerboard(image: np.ndarray,
                                    checkerboard_size: Tuple[int, int],
                                    square_size: float,
                                    camera_matrix: np.ndarray,
                                    dist_coeffs: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从棋盘格估计相机姿态

    Args:
        image: 输入图像
        checkerboard_size: 棋盘格大小
        square_size: 方格大小
        camera_matrix: 相机内参
        dist_coeffs: 畸变系数

    Returns:
        (旋转向量, 平移向量) 或 None
    """
    try:
        # 检测角点
        ret, corners = detect_checkerboard_corners(image, checkerboard_size)

        if not ret:
            return None

        # 生成对象点
        obj_points = generate_checkerboard_points(checkerboard_size, square_size)

        # 计算姿态
        ret, rvec, tvec = cv2.solvePnP(
            obj_points, corners, camera_matrix, dist_coeffs
        )

        if ret:
            return rvec, tvec
        else:
            return None

    except Exception as e:
        logging.error(f"估计姿态失败: {str(e)}")
        return None