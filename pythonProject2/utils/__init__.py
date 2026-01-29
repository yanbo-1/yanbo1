"""
工具函数模块
提供文件IO、相机标定、日志记录等辅助功能
"""

from .file_io import (
    save_results_to_csv,
    save_results_to_excel,
    save_image_with_annotations,
    load_image,
    save_image,
    save_parameters_to_yaml,
    load_parameters_from_yaml,
    create_directory,
    get_unique_filename,
    export_report_to_pdf,
    export_report_to_word,
    DataExporter
)

from .calibration import (
    CameraCalibrator,
    generate_checkerboard_points,
    detect_checkerboard_corners,
    calculate_reprojection_error,
    save_calibration_results,
    load_calibration_results,
    CalibrationError,
    CalibrationResult
)

from .logger import (
    setup_logger,
    get_logger,
    log_function_call,
    log_exception,
    PerformanceLogger,
    setup_file_logger,
    setup_console_logger,
    LogLevel
)

# 版本信息
__version__ = "1.0.0"
__author__ = "严波"
__email__ = "5320240755@stu.swust.edu.cn"

__all__ = [
    # file_io 模块
    'save_results_to_csv',
    'save_results_to_excel',
    'save_image_with_annotations',
    'load_image',
    'save_image',
    'save_parameters_to_yaml',
    'load_parameters_from_yaml',
    'create_directory',
    'get_unique_filename',
    'export_report_to_pdf',
    'export_report_to_word',
    'DataExporter',

    # calibration_images 模块
    'CameraCalibrator',
    'generate_checkerboard_points',
    'detect_checkerboard_corners',
    'calculate_reprojection_error',
    'save_calibration_results',
    'load_calibration_results',
    'CalibrationError',
    'CalibrationResult',

    # logger 模块
    'setup_logger',
    'get_logger',
    'log_function_call',
    'log_exception',
    'PerformanceLogger',
    'setup_file_logger',
    'setup_console_logger',
    'LogLevel',
]