# config/__init__.py
"""
配置文件包初始化模块
提供配置加载和保存的统一接口
"""

import os
import yaml
from typing import Dict, Any, Optional

__all__ = [
    'load_config',
    'save_config',
    'get_default_params',
    'get_camera_config',
    'update_config',
    'CONFIG_DIR',
    'DEFAULT_PARAMS_PATH',
    'CAMERA_CONFIG_PATH'
]

# 配置文件路径
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS_PATH = os.path.join(CONFIG_DIR, 'default_params.yaml')
CAMERA_CONFIG_PATH = os.path.join(CONFIG_DIR, 'camera_config.yaml')


def load_config(config_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


def get_default_params() -> Dict[str, Any]:
    """
    获取默认算法参数

    Returns:
        默认参数配置字典
    """
    return load_config(DEFAULT_PARAMS_PATH)


def get_camera_config() -> Dict[str, Any]:
    """
    获取相机配置

    Returns:
        相机配置字典
    """
    return load_config(CAMERA_CONFIG_PATH)


def update_config(section: str, key: str, value: Any, config_path: str) -> Dict[str, Any]:
    """
    更新配置文件的特定键值

    Args:
        section: 配置节名称
        key: 配置键名
        value: 配置值
        config_path: 配置文件路径

    Returns:
        更新后的完整配置
    """
    config = load_config(config_path)

    if section not in config:
        config[section] = {}

    config[section][key] = value

    save_config(config, config_path)

    return config


# 创建配置文件（如果不存在）
def init_config_files():
    """初始化配置文件（如果不存在）"""
    # 创建默认参数文件
    if not os.path.exists(DEFAULT_PARAMS_PATH):
        default_params = {
            'preprocess': {
                'resize_width': 800,
                'resize_height': 600,
                'use_adaptive_threshold': True,
                'adaptive_method': 'GAUSSIAN',
                'adaptive_block_size': 11,
                'adaptive_c': 2,
                'use_median_filter': True,
                'median_kernel_size': 5,
                'use_gaussian_filter': True,
                'gaussian_kernel_size': 5,
                'gaussian_sigma': 1.0,
                'use_morphology': True,
                'morphology_operation': 'OPEN',
                'morphology_kernel_size': 3,
                'morphology_iterations': 1,
                'use_retinex': True,
                'retinex_scales': 3,
                'retinex_sigma': 15,
                'use_histogram_equalization': True,
                'clahe_clip_limit': 2.0,
                'clahe_grid_size': 8
            },
            'circle_detection': {
                'use_hough_transform': True,
                'hough_dp': 1,
                'hough_min_dist': 50,
                'hough_param1': 100,
                'hough_param2': 30,
                'hough_min_radius': 10,
                'hough_max_radius': 200,
                'use_least_squares': True,
                'ls_min_points': 10,
                'ls_max_iterations': 100,
                'ls_tolerance': 1e-3,
                'edge_detection_method': 'CANNY',
                'canny_threshold1': 50,
                'canny_threshold2': 150,
                'canny_aperture_size': 3,
                'canny_l2_gradient': False
            },
            'distortion_correction': {
                'enable_correction': True,
                'camera_matrix': {
                    'fx': 1000.0,
                    'fy': 1000.0,
                    'cx': 400.0,
                    'cy': 300.0
                },
                'dist_coeffs': [0.1, -0.2, 0.001, 0.001, 0.1],
                'calibration_file': 'data/calibration_images/calibration_data.npz',
                'use_undistort': True
            },
            'concentricity_calculation': {
                'pixel_to_mm_ratio': 0.1,
                'reference_radius_mm': 25.0,
                'tolerance_threshold': 0.2,
                'max_allowed_deviation': 1.0,
                'unit': 'mm',
                'output_format': 'both',  # 'both', 'pixel', 'mm'
                'calculate_statistics': True,
                'statistics_window': 10
            },
            'system': {
                'log_level': 'INFO',
                'save_intermediate_results': False,
                'auto_save_interval': 60,
                'max_history_size': 100,
                'default_image_format': 'png',
                'default_report_format': 'excel'
            }
        }
        save_config(default_params, DEFAULT_PARAMS_PATH)
        print(f"已创建默认参数文件: {DEFAULT_PARAMS_PATH}")

    # 创建相机配置文件
    if not os.path.exists(CAMERA_CONFIG_PATH):
        camera_config = {
            'camera_settings': {
                'camera_index': 0,
                'resolution': {
                    'width': 1920,
                    'height': 1080
                },
                'fps': 30,
                'exposure': -1,  # 自动曝光
                'gain': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'white_balance': -1,  # 自动白平衡
                'auto_focus': True,
                'focus': 0
            },
            'capture_settings': {
                'trigger_mode': 'software',  # 'software', 'hardware'
                'buffer_size': 10,
                'timeout_ms': 1000,
                'retry_count': 3,
                'save_raw_images': False,
                'raw_image_format': 'bmp',
                'compression_quality': 95
            },
            'lighting_settings': {
                'enable_lighting_control': False,
                'lighting_intensity': 50,
                'lighting_pattern': 'uniform',
                'lighting_color': 'white',
                'lighting_duration_ms': 100
            },
            'calibration_images': {
                'calibrated': False,
                'calibration_date': '',
                'calibration_valid_until': '',
                'intrinsic_matrix': None,
                'distortion_coefficients': None,
                'rotation_vectors': None,
                'translation_vectors': None,
                'reprojection_error': 0.0,
                'calibration_board': {
                    'type': 'checkerboard',
                    'rows': 9,
                    'columns': 6,
                    'square_size_mm': 25.0
                }
            },
            'network_settings': {
                'use_network_camera': False,
                'ip_address': '192.168.1.100',
                'port': 554,
                'username': '',
                'password': '',
                'stream_url': '',
                'protocol': 'rtsp'
            },
            'advanced_settings': {
                'enable_hdr': False,
                'hdr_mode': 'standard',
                'enable_denoising': True,
                'denoising_strength': 50,
                'enable_sharpening': False,
                'sharpening_strength': 25,
                'enable_color_correction': True,
                'color_profile': 'sRGB',
                'enable_lens_correction': False,
                'lens_profile': ''
            }
        }
        save_config(camera_config, CAMERA_CONFIG_PATH)
        print(f"已创建相机配置文件: {CAMERA_CONFIG_PATH}")


# 初始化配置文件
init_config_files()