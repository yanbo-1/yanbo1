"""
主窗口界面模块
"""

import sys
import os
import math

# 设置项目根目录（用于导入其他模块）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 确保core目录在路径中
core_dir = os.path.join(project_root, 'core')
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# 添加这行：确保utils目录在路径中
utils_dir = os.path.join(project_root, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QLabel, QStatusBar,
                             QMenuBar, QMenu, QAction, QMessageBox, QFileDialog,
                             QSplitter, QProgressBar, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QDialog, QInputDialog, QFrame
import cv2
import numpy as np

from .report_generator import ReportGenerator

# 从同一目录导入UI组件（相对导入）
from .camera_viewer import CameraViewer
from .parameter_panel import ParameterPanel
from .result_viewer import ResultViewer

# 从core目录导入（绝对导入）
try:
    from camera import ImageController
    from preprocess import ImagePreprocessor
    from circle_detection import CircleDetector
    from concentricity_calc import ConcentricityCalculator
except ImportError as e:
    print(f"导入core模块失败: {e}")

    # 创建日志记录器
    import logging


    # 先创建一个简单的日志类
    class SimpleLogger:
        def __init__(self, name):
            self.name = name

        def error(self, msg): print(f"[ERROR] {msg}")

        def info(self, msg): print(f"[INFO] {msg}")

        def warning(self, msg): print(f"[WARNING] {msg}")


    logger = SimpleLogger('PhoneImage')

    # 保持其他模拟类不变
    class ImagePreprocessor:
        def __init__(self):
            pass

        def process(self, image, **kwargs):
            return image


    class CircleDetector:
        def __init__(self):
            self.concentricity_params = {
                'outer_min_radius': 240,
                'outer_max_radius': 330,
                'thread_min_area': 50,
                'thread_roi_expand': 15,
                'pixel_to_mm_ratio': 0.033,
                'quality_threshold_mm': 0.2
            }

        def detect_circles(self, image, **kwargs):
            return []

        def detect_concentricity_pair(self, image):
            return {
                'success': True,
                'inner_center': {'x': 320, 'y': 240, 'area': 100, 'is_hexagon': False, 'method': 'simulated'},
                'outer_circle': {'x': 320, 'y': 240, 'radius': 280, 'confidence': 0.95},
                'pixel_error': 5.0,
                'error_mm': 0.165,
                'is_qualified': True,
                'dx': 5.0,
                'dy': 0.0,
                'relative_error_percent': 0.5
            }

        def update_parameters(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.concentricity_params:
                    self.concentricity_params[key] = value

        def validate_concentricity_result(self, result):
            return result.get('success', False)

        def draw_concentricity_result(self, image, result):
            return image


    class ConcentricityCalculator:
        def __init__(self):
            pass

        def calculate(self, inner, outer, **kwargs):
            return {'concentricity': 0.0, 'is_qualified': True}

# 从utils目录导入
try:
    from utils.logger import setup_logger
    from utils.file_io import save_results_to_csv, save_image_with_annotations
    from utils.image_loader import load_image_chinese
except ImportError:
    # 创建模拟函数
    def setup_logger(name):
        import logging
        return logging.getLogger(name)


    def save_results_to_csv(results, filepath):
        print(f"保存结果到CSV: {filepath}")


    def save_image_with_annotations(image, filepath):
        print(f"保存图像: {filepath}")


    def load_image_chinese(filepath):
        # 简单的模拟函数
        import cv2
        return cv2.imread(filepath)

class MainWindow(QMainWindow):
    """主窗口类"""

    # 信号定义
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logger = setup_logger('UI')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        # 添加数据目录路径
        self.data_dir = os.path.join(self.project_root, "data")

        self.detection_results_history = []  # 历史记录列表
        self.current_detection_result = None  # 当前结果

        self.init_ui()
        self.init_components()
        self.init_connections()
        self.current_image = None
        self.detection_results = None

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle('机械零件同心度检测系统 v1.0')
        self.setGeometry(100, 100, 1400, 800)

        # 设置图标
        if os.path.exists('assets/icon.png'):
            self.setWindowIcon(QIcon('assets/icon.png'))

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧面板（图像显示和参数设置）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 图像显示区域
        image_group = QGroupBox("图像显示")
        image_layout = QVBoxLayout()
        self.camera_viewer = CameraViewer()
        image_layout.addWidget(self.camera_viewer)
        image_group.setLayout(image_layout)

        # 控制按钮
        button_group = QGroupBox("控制面板")
        button_layout = QHBoxLayout()

        self.btn_camera = QPushButton("打开图像源")
        self.btn_camera.setFixedHeight(40)
        self.btn_load = QPushButton("加载图像")
        self.btn_load.setFixedHeight(40)
        self.btn_capture = QPushButton("拍照")  # 新增拍照按钮
        self.btn_capture.setFixedHeight(40)
        self.btn_capture.setEnabled(False)  # 初始不可用
        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setFixedHeight(40)
        self.btn_detect.setEnabled(False)
        # 删除"保存结果"按钮
        self.btn_report = QPushButton("生成报告")
        self.btn_report.setFixedHeight(40)
        self.btn_report.setEnabled(False)

        button_layout.addWidget(self.btn_camera)
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_capture)  # 添加拍照按钮
        button_layout.addWidget(self.btn_detect)
        button_layout.addWidget(self.btn_report)  # 删除后仅剩4个按钮
        button_group.setLayout(button_layout)

        # 参数设置面板
        self.parameter_panel = ParameterPanel()

        # 添加到左侧布局
        left_layout.addWidget(image_group, 4)
        left_layout.addWidget(button_group, 1)
        left_layout.addWidget(self.parameter_panel, 2)

        # 右侧面板（结果显示）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.result_viewer = ResultViewer()
        right_layout.addWidget(self.result_viewer)

        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 500])

        main_layout.addWidget(splitter)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)

        # 创建菜单栏
        self.create_menu_bar()

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        open_action = QAction('打开图像', self)
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)

        open_phone_action = QAction('打开手机图像', self)
        open_phone_action.triggered.connect(self.load_phone_image_file)
        file_menu.addAction(open_phone_action)

        connect_phone_action = QAction('连接手机摄像头', self)
        connect_phone_action.triggered.connect(self.connect_to_phone_camera)
        file_menu.addAction(connect_phone_action)

        capture_action = QAction('拍照', self)  # 新增拍照菜单项
        capture_action.triggered.connect(self.capture_image)
        file_menu.addAction(capture_action)

        file_menu.addSeparator()

        report_action = QAction('生成报告', self)
        report_action.triggered.connect(self.generate_report)
        file_menu.addAction(report_action)

        file_menu.addSeparator()

        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 工具菜单
        tools_menu = menubar.addMenu('工具')

        # 修改为手机标定
        calibrate_action = QAction('手机标定', self)
        calibrate_action.triggered.connect(self.calibrate_phone_camera)
        tools_menu.addAction(calibrate_action)

        settings_action = QAction('系统设置', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        phone_guide_action = QAction('手机使用指南', self)
        phone_guide_action.triggered.connect(self.show_phone_guide)
        help_menu.addAction(phone_guide_action)

        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        manual_action = QAction('使用手册', self)
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)

    def init_components(self):
        """初始化组件"""
        # 替换CameraController为ImageController
        self.image_controller = ImageController()
        self.preprocessor = ImagePreprocessor()
        self.circle_detector = CircleDetector()
        self.concentricity_calc = ConcentricityCalculator()

        # 添加手机摄像头定时器（用于实时预览）
        self.phone_camera_timer = QTimer()
        self.phone_camera_timer.timeout.connect(self.update_phone_camera_frame)

        # 加载默认参数
        self.load_default_parameters()

    def init_connections(self):
        """初始化信号连接"""
        # 按钮连接
        self.btn_camera.clicked.connect(self.toggle_phone_camera)  # 改为新的方法
        self.btn_load.clicked.connect(self.load_image)
        self.btn_capture.clicked.connect(self.capture_image)  # 新增拍照连接
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_report.clicked.connect(self.generate_report)

        # 参数变化连接
        self.parameter_panel.parameters_changed.connect(self.update_parameters)

        # 处理信号连接
        self.processing_started.connect(self.on_processing_started)
        self.processing_finished.connect(self.on_processing_finished)
        self.error_occurred.connect(self.on_error_occurred)

    def load_default_parameters(self):
        """加载默认参数"""
        default_params = {
            'preprocess': {
                'brightness_compensation': True,
                'retinex_enabled': True,
                'median_filter_size': 3,
                'gaussian_filter_size': 5,
                'canny_low': 50,
                'canny_high': 150
            },
            'circle_detection': {
                'hough_dp': 1,
                'hough_min_dist': 20,
                'hough_param1': 50,
                'hough_param2': 30,
                'hough_min_radius': 10,
                'hough_max_radius': 100
            },
            'concentricity': {
                'pixel_to_mm': 0.1,
                'reference_radius_mm': 50.0,
                'tolerance': 0.2
            }
        }
        self.parameter_panel.set_parameters(default_params)

    def toggle_phone_camera(self):
        """切换手机摄像头连接状态"""
        if self.image_controller.is_opened():
            # 如果已连接，则断开
            self.image_controller.close()
            self.phone_camera_timer.stop()
            self.btn_camera.setText("连接手机")
            self.btn_capture.setEnabled(False)  # 断开时禁用拍照按钮
            self.btn_detect.setEnabled(False)  # 断开时也禁用检测按钮
            self.status_label.setText("手机摄像头已断开")
        else:
            # 显示连接选项
            self.show_phone_connection_dialog()

    def show_phone_connection_dialog(self):
        """显示手机连接选项对话框 - 简化版（无高级设置）"""
        # 先询问基本URL
        camera_url, ok = QInputDialog.getText(
            self, "连接手机摄像头",
            "请输入手机摄像头URL（例如：http://192.168.1.100:8080）\n\n"
            "手机IP摄像头应用设置：\n"
            "1. 确保手机和电脑在同一WiFi\n"
            "2. 在手机应用中选择较低的编码质量\n"
            "3. 选择MJPEG编码（如果有）\n"
            "4. 分辨率设置为1280x720或更低\n\n"
            "请输入URL：",
            text="http://192.168.1.17:8080"
        )

        if ok and camera_url:
            try:
                # 确保URL有正确的格式
                if not camera_url.startswith("http://"):
                    camera_url = "http://" + camera_url

                # 添加/video路径
                if not camera_url.endswith("/video"):
                    camera_url = camera_url.rstrip("/") + "/video"

                # 尝试连接手机摄像头
                if self.image_controller.open(source_type="camera", url=camera_url):
                    # 使用固定帧率（不再询问高级设置）
                    target_fps = 10  # 默认帧率
                    timer_interval = int(1000 / target_fps)  # 毫秒

                    self.phone_camera_timer.start(timer_interval)
                    self.btn_camera.setText("断开连接")
                    self.btn_capture.setEnabled(True)
                    self.status_label.setText(f"手机摄像头已连接 (约{target_fps}FPS)")

                    QMessageBox.information(self, "成功",
                                            f"手机摄像头连接成功！\n\n"
                                            f"URL: {camera_url}\n"
                                            f"帧率: 约{target_fps}FPS\n\n"
                                            "如果延迟仍然很高，请尝试：\n"
                                            "1. 在手机应用中选择更低的分辨率\n"
                                            "2. 确保WiFi信号良好\n"
                                            "3. 关闭其他占用网络的应用")
                else:
                    QMessageBox.warning(self, "错误",
                                        "无法连接手机摄像头！\n\n请确保：\n"
                                        "1. 手机和电脑在同一WiFi网络\n"
                                        "2. 手机已启动IP摄像头应用\n"
                                        "3. 输入正确的URL\n"
                                        "4. 防火墙没有阻止连接")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"连接失败: {str(e)}")

    def load_phone_image_file(self):
        """加载手机图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择手机拍摄的图像",
            "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )

        if file_path:
            try:
                # 使用image_controller加载图像
                success = self.image_controller.load_image_file(file_path)
                if success:
                    image = self.image_controller.get_frame()
                    if image is not None:
                        self.current_image = image
                        self.camera_viewer.set_image(image)
                        self.btn_detect.setEnabled(True)
                        self.btn_camera.setText("更换图像")
                        self.status_label.setText(f"手机图像已加载: {os.path.basename(file_path)}")

                        # 显示拍照提示（只显示一次）
                        if not hasattr(self, '_phone_tips_shown'):
                            self.show_phone_camera_tips()
                            self._phone_tips_shown = True
                    else:
                        QMessageBox.warning(self, "警告", "图像加载失败！")
                else:
                    QMessageBox.warning(self, "警告", "无法加载图像文件！")

            except Exception as e:
                self.logger.error(f"加载手机图像失败: {str(e)}")
                QMessageBox.warning(self, "错误", f"加载图像失败: {str(e)}")

    def capture_image(self):
        """从手机摄像头捕获图像 - 最终版"""
        if not self.image_controller.is_opened():
            QMessageBox.warning(self, "警告", "请先连接手机摄像头！")
            return

        try:
            # 获取当前显示的最新帧
            frame = self.image_controller.capture_frame()

            if frame is not None:
                # 保存原始图像（拍照时使用原始帧）
                self.current_image = frame.copy()

                # 显示拍照瞬间的图像
                self.camera_viewer.set_image(frame)

                # 启用检测按钮
                self.btn_detect.setEnabled(True)

                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 创建保存目录
                save_dir = os.path.join(self.data_dir, "captured_images")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                # 保存图像 - 主要使用PIL
                filename = os.path.join(save_dir, f"captured_{timestamp}.jpg")

                try:
                    # 方法1: 使用PIL保存（这是主要方法）
                    from PIL import Image

                    # 确保图像是RGB格式（PIL需要的格式）
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # 转换为PIL格式
                        pil_image = Image.fromarray(frame)
                    else:
                        # 如果是灰度图，转换为RGB
                        pil_image = Image.fromarray(frame).convert('RGB')

                    # 保存为JPEG，质量为95%
                    pil_image.save(filename, 'JPEG', quality=95)

                    # 验证保存成功
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        self.status_label.setText(f"✅ 已拍照并保存 ({file_size / 1024:.1f}KB)")

                        # 显示简化的路径
                        display_path = filename
                        if len(display_path) > 80:
                            display_path = "..." + display_path[-77:]

                        QMessageBox.information(self, "拍照成功",
                                                f"✅ 图像保存成功！\n\n"
                                                f"📁 保存位置:\n{display_path}\n"
                                                f"📊 文件大小: {file_size / 1024:.1f} KB\n"
                                                f"现在可以点击'开始检测'按钮进行分析。")

                        self.logger.info(f"✅ 图像保存成功: {filename}")

                    else:
                        QMessageBox.warning(self, "警告", "图像保存失败，文件未创建！")

                except ImportError:
                    # 如果没有安装PIL，尝试使用OpenCV
                    self.logger.warning("未安装PIL，尝试使用OpenCV保存")
                    try:
                        # 转换为BGR格式
                        image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        success = cv2.imwrite(filename, image_bgr)

                        if success and os.path.exists(filename):
                            file_size = os.path.getsize(filename)
                            self.status_label.setText(f"已拍照并保存 ({file_size / 1024:.1f}KB)")
                            QMessageBox.information(self, "拍照成功", f"图像已保存到: {os.path.basename(filename)}")
                        else:
                            QMessageBox.warning(self, "警告", "无法保存图像！")

                    except Exception as cv_error:
                        self.logger.error(f"OpenCV保存也失败: {cv_error}")
                        QMessageBox.warning(self, "错误", f"保存失败: {cv_error}")

                except Exception as pil_error:
                    self.logger.error(f"PIL保存失败: {pil_error}")
                    QMessageBox.warning(self, "错误", f"保存失败: {pil_error}")

            else:
                QMessageBox.warning(self, "警告", "无法捕获图像，请检查摄像头连接！")

        except Exception as e:
            self.logger.error(f"拍照失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"拍照失败: {str(e)}")

    def capture_image_with_preview(self):
        """带预览的图像采集"""
        if not self.image_controller.is_opened():
            QMessageBox.warning(self, "警告", "请先连接手机摄像头！")
            return

        # 创建预览对话框
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtCore import QTimer

        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("图像预览")
        preview_dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(preview_dialog)

        # 预览标签
        preview_label = QLabel("正在加载图像...")
        preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(preview_label)

        # 按钮布局
        button_layout = QHBoxLayout()

        btn_capture = QPushButton("📸 拍照")
        btn_retry = QPushButton("🔄 重拍")
        btn_confirm = QPushButton("✅ 使用此图像")
        btn_cancel = QPushButton("❌ 取消")

        button_layout.addWidget(btn_capture)
        button_layout.addWidget(btn_retry)
        button_layout.addWidget(btn_confirm)
        button_layout.addWidget(btn_cancel)

        layout.addLayout(button_layout)

        captured_image = None

        def update_preview():
            nonlocal captured_image
            frame = self.image_controller.capture_frame()
            if frame is not None:
                # 转换为QPixmap显示
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                # 缩放以适合预览窗口
                scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                preview_label.setPixmap(scaled_pixmap)
                captured_image = frame

        def on_capture():
            update_preview()
            btn_confirm.setEnabled(True)
            btn_retry.setEnabled(True)
            preview_label.setText("图像已捕获")

        def on_retry():
            btn_confirm.setEnabled(False)
            btn_retry.setEnabled(False)
            preview_label.setText("重新拍照...")

        def on_confirm():
            if captured_image is not None:
                self.current_image = captured_image
                self.camera_viewer.set_image(captured_image)
                self.btn_detect.setEnabled(True)

                # 保存图像 - 使用PIL保存
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = os.path.join(self.data_dir, "captured_images")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    # 使用PIL保存
                    from PIL import Image

                    if len(captured_image.shape) == 3 and captured_image.shape[2] == 3:
                        pil_image = Image.fromarray(captured_image)
                    else:
                        pil_image = Image.fromarray(captured_image).convert('RGB')

                    filename = os.path.join(save_dir, f"captured_{timestamp}.jpg")
                    pil_image.save(filename, 'JPEG', quality=95)

                    self.status_label.setText(f"已使用图像: {filename}")
                    preview_dialog.accept()

                except Exception as e:
                    self.logger.error(f"保存预览图像失败: {e}")
                    QMessageBox.warning(self, "错误", f"保存图像失败: {e}")

        def on_cancel():
            preview_dialog.reject()

        # 连接信号
        btn_capture.clicked.connect(on_capture)
        btn_retry.clicked.connect(on_retry)
        btn_confirm.clicked.connect(on_confirm)
        btn_cancel.clicked.connect(on_cancel)

        # 初始状态
        btn_confirm.setEnabled(False)
        btn_retry.setEnabled(False)

        # 开始预览
        preview_timer = QTimer()
        preview_timer.timeout.connect(update_preview)
        preview_timer.start(100)  # 10fps预览

        # 显示对话框
        if preview_dialog.exec_() == QDialog.Accepted:
            self.status_label.setText("图像已确认，可以开始检测")
        else:
            self.status_label.setText("拍照已取消")

        # 停止预览定时器
        preview_timer.stop()

    def connect_to_phone_camera(self):
        """连接到手机摄像头"""
        # 获取摄像头URL（可以做成配置）
        camera_url, ok = QInputDialog.getText(
            self, "连接手机摄像头",
            "请输入手机摄像头URL：\n（例如：http://192.168.1.100:8080/video）",
            text="http://192.168.1.100:8080/video"
        )

        if ok and camera_url:
            try:
                # 尝试连接手机摄像头
                if self.image_controller.open("camera"):
                    self.phone_camera_timer.start(100)  # 10fps
                    self.btn_camera.setText("断开摄像头")
                    self.status_label.setText("手机摄像头已连接")
                    QMessageBox.information(self, "成功", "手机摄像头连接成功！")
                else:
                    QMessageBox.warning(self, "错误",
                                        "无法连接手机摄像头！\n\n请确保：\n1. 手机和电脑在同一WiFi网络\n2. 手机已安装IP摄像头应用\n3. 摄像头URL正确")

            except Exception as e:
                self.logger.error(f"连接手机摄像头失败: {str(e)}")
                QMessageBox.warning(self, "错误", f"连接失败: {str(e)}")

    def update_phone_camera_frame(self):
        """更新手机摄像头帧 - 进一步优化"""
        if self.image_controller.is_opened():
            frame = self.image_controller.capture_frame()
            if frame is not None:
                # 直接显示，不进行额外处理
                self.camera_viewer.set_image(frame)

                # 只在需要时才更新current_image（比如准备拍照时）
                # 这里不更新，减少拷贝开销
                # self.current_image = frame

                # 启用检测按钮（当有图像时）
                self.btn_detect.setEnabled(True)
            else:
                # 如果无法获取帧，可能是连接问题
                self.logger.warning("无法从手机摄像头获取帧")

    def update_camera_frame(self):
        """更新相机帧"""
        if self.camera_controller.is_opened():
            frame = self.camera_controller.get_frame()
            if frame is not None:
                self.current_image = frame.copy()
                self.camera_viewer.set_image(frame)
                self.btn_detect.setEnabled(True)

    def load_image(self):
        """加载图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件",
            "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )

        if file_path:
            try:
                # 读取图像
                # image = cv2.imread(file_path)
                image = load_image_chinese(file_path)
                if image is None:
                    raise ValueError("无法读取图像文件")

                # 转换为RGB格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = image_rgb

                # 显示图像
                self.camera_viewer.set_image(image_rgb)
                self.btn_detect.setEnabled(True)
                self.status_label.setText(f"已加载图像: {os.path.basename(file_path)}")

            except Exception as e:
                self.logger.error(f"加载图像失败: {str(e)}")
                QMessageBox.warning(self, "错误", f"加载图像失败: {str(e)}")

    def start_detection(self):
        """开始检测"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像！")
            return

        # 停止摄像头预览（如果正在运行）
        if self.image_controller.is_opened():
            self.phone_camera_timer.stop()

        print("=== 紧急修复：重置检测参数 ===")

        # 强制设置合理的参数范围
        self.circle_detector.concentricity_params.update({
            'outer_min_radius': 240,
            'outer_max_radius': 330,
            'thread_min_area': 50,
            'thread_roi_expand': 15,
            'pixel_to_mm_ratio': 0.033,
        })

        # 同时更新通用参数
        self.circle_detector.update_parameters(
            hough_min_radius=240,
            hough_max_radius=330,
            hough_param2=30,  # 提高阈值减少误检
        )

        # 禁用按钮
        self.btn_detect.setEnabled(False)
        self.processing_started.emit()

        try:
            # 获取当前参数
            params = self.parameter_panel.get_parameters()

            # === 第一步：图像预处理和调试 ===
            # 首先保存原始图像供调试
            if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 3:
                # 当前是RGB格式，转换为BGR用于OpenCV处理
                image_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = self.current_image.copy()

            # 保存原始图像
            debug_dir = "debug_output"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{debug_dir}/debug_original_{timestamp}.jpg", image_bgr)
            print(f"原始图像形状: {image_bgr.shape}")

            # 转换为灰度图
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{debug_dir}/debug_gray_{timestamp}.jpg", gray)

            # 调试步骤1：简单阈值处理，测试是否能提取螺纹杆
            print("=" * 50)
            print("开始图像预处理调试...")
            print("=" * 50)

            # 方法1: 全局阈值
            _, binary_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(f"{debug_dir}/debug_binary_global_{timestamp}.jpg", binary_global)

            # 方法2: 自适应阈值（处理光照不均匀）
            binary_adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            cv2.imwrite(f"{debug_dir}/debug_binary_adaptive_{timestamp}.jpg", binary_adaptive)

            # 方法3: Otsu阈值
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imwrite(f"{debug_dir}/debug_binary_otsu_{timestamp}.jpg", binary_otsu)

            # 查找并绘制轮廓，看看是否能找到螺纹杆
            best_contour_count = 0
            best_method = ""

            for binary_img, name in [(binary_global, "global"),
                                     (binary_adaptive, "adaptive"),
                                     (binary_otsu, "otsu")]:
                contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
                cv2.imwrite(f"{debug_dir}/debug_contours_{name}_{timestamp}.jpg", contour_img)

                contour_count = len(contours)
                if contour_count > best_contour_count:
                    best_contour_count = contour_count
                    best_method = name

                print(f"{name}阈值找到轮廓数: {contour_count}")

            print(f"\n最佳方法: {best_method} (找到{best_contour_count}个轮廓)")
            print("=" * 50)

            # === 第二步：更新检测器参数 ===
            print("更新检测器参数...")

            # 更新通用圆检测参数
            self.circle_detector.update_parameters(
                hough_dp=params['circle_detection']['hough_dp'],
                hough_min_dist=params['circle_detection']['hough_min_dist'],
                hough_param1=params['circle_detection']['hough_param1'],
                hough_param2=params['circle_detection']['hough_param2'],
                hough_min_radius=params['circle_detection']['hough_min_radius'],
                hough_max_radius=params['circle_detection']['hough_max_radius']
            )

            # === 关键修改：设置同心度检测参数 ===
            print("设置同心度检测参数...")

            # 设置螺纹杆检测的关键参数
            concentricity_params = {
                'thread_min_area': 80,  # 降低面积阈值
                'thread_roi_expand': 20,  # 减少ROI扩展
                'quality_threshold_mm': 0.2,
                'pixel_to_mm_ratio': 0.033,
                # params['concentricity']['pixel_to_mm']
                'outer_min_radius': 250,  # 调整外圆最小半径
                'outer_max_radius': 320,  # 调整外圆最大半径
            }

            # 直接更新检测器的同心度参数
            if hasattr(self.circle_detector, 'concentricity_params'):
                for key, value in concentricity_params.items():
                    self.circle_detector.concentricity_params[key] = value
                    print(f"设置{key} = {value}")

            # === 第三步：调用检测器 ===
            print("开始同心度检测...")
            result = self.circle_detector.detect_concentricity_pair(image_bgr)

            # ===== 第二步：数据验证和补充 =====
            print("开始数据验证和补充...")

            # 检查关键字段是否存在
            required_fields = ['inner_center', 'outer_circle', 'pixel_error', 'error_mm', 'is_qualified']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                print(f"警告：检测结果缺少字段: {missing_fields}")
                # 补充缺失字段
                for field in missing_fields:
                    if field == 'inner_center':
                        result['inner_center'] = {'x': 0, 'y': 0, 'area': 0, 'is_hexagon': False}
                    elif field == 'outer_circle':
                        result['outer_circle'] = {'x': 0, 'y': 0, 'radius': 0}
                    elif field == 'pixel_error':
                        result['pixel_error'] = 0.0
                    elif field == 'error_mm':
                        result['error_mm'] = 0.0
                    elif field == 'is_qualified':
                        result['is_qualified'] = False

            # 确保数值字段是数字类型
            def ensure_float(value, default=0.0):
                try:
                    return float(value)
                except:
                    return default

            # 验证并转换内圆坐标
            if 'inner_center' in result:
                inner = result['inner_center']
                if isinstance(inner, dict):
                    inner['x'] = ensure_float(inner.get('x', 0))
                    inner['y'] = ensure_float(inner.get('y', 0))
                    inner['area'] = ensure_float(inner.get('area', 0))
                else:
                    # 如果inner_center不是字典，创建默认值
                    result['inner_center'] = {'x': 0, 'y': 0, 'area': 0}

            # 验证并转换外圆坐标
            if 'outer_circle' in result:
                outer = result['outer_circle']
                if isinstance(outer, dict):
                    outer['x'] = ensure_float(outer.get('x', 0))
                    outer['y'] = ensure_float(outer.get('y', 0))
                    outer['radius'] = ensure_float(outer.get('radius', 0))
                else:
                    result['outer_circle'] = {'x': 0, 'y': 0, 'radius': 0}

            # 验证其他数值字段
            numeric_fields = ['pixel_error', 'error_mm', 'concentricity', 'relative_error_percent', 'dx', 'dy']
            for field in numeric_fields:
                if field in result:
                    result[field] = ensure_float(result[field])

            print(f"数据验证完成，内圆: {result.get('inner_center', {})}")
            print(f"外圆: {result.get('outer_circle', {})}")

            # ===== 关键修改：修正检测结果分析 =====
            print("\n" + "=" * 60)
            print("详细检测结果分析:")
            print("=" * 60)
            print(f"检测成功状态: {result.get('success', False)}")

            # 检查内圆信息 - 修正：检测器返回的是inner_center，不是inner_circle
            if 'inner_center' in result:
                inner = result['inner_center']
                print(f"内圆中心信息: {inner}")
                print(f"  坐标: ({inner.get('x', 'N/A')}, {inner.get('y', 'N/A')})")
                print(f"  面积: {inner.get('area', 'N/A')}")
                print(f"  是否为六边形: {inner.get('is_hexagon', 'N/A')}")
                print(f"  顶点数: {inner.get('vertex_count', 'N/A')}")
                print(f"  置信度: {inner.get('confidence', 'N/A')}")
            else:
                print("未找到inner_center信息")

            # 检查外圆信息
            if 'outer_circle' in result:
                outer = result['outer_circle']
                print(f"外圆信息: {outer}")
                print(f"  坐标: ({outer.get('x', 'N/A')}, {outer.get('y', 'N/A')})")
                print(f"  半径: {outer.get('radius', 'N/A')}")
                print(f"  置信度: {outer.get('confidence', 'N/A')}")
            else:
                print("未找到外圆信息")

            # 检查偏差信息
            if 'pixel_error' in result:
                print(f"像素偏差: {result['pixel_error']:.2f}")
            if 'error_mm' in result:
                print(f"实际偏差(mm): {result['error_mm']:.3f}")
            if 'is_qualified' in result:
                print(f"是否合格: {result['is_qualified']}")

            print("=" * 60 + "\n")
            # ===== 结束修改 =====

            # 验证检测结果
            if not result.get('success', False):
                error_msg = result.get('error', '未知错误')
                print(f"检测失败原因: {error_msg}")
                raise ValueError(f"同心度检测失败: {error_msg}")

            # === 关键修改：为UI构建inner_circle数据结构 ===
            # 检测器返回inner_center，但UI可能需要inner_circle
            if 'inner_center' in result and 'outer_circle' in result:
                inner_info = result['inner_center']
                outer_info = result['outer_circle']

                # 估算内圆半径（螺纹杆半径大约为外圆半径的1/4到1/3）
                estimated_radius = outer_info['radius'] * 0.25

                # 构建inner_circle结构供UI使用
                inner_circle = {
                    'x': inner_info['x'],
                    'y': inner_info['y'],
                    'radius': estimated_radius,
                    'confidence': inner_info.get('confidence', 0.8),
                    'method': inner_info.get('method', 'threaded_rod'),
                    'area': inner_info.get('area', 0),
                    'is_hexagon': inner_info.get('is_hexagon', False)
                }

                # 将inner_circle添加到result中
                result['inner_circle'] = inner_circle
                print(f"构建的inner_circle: 坐标({inner_circle['x']:.1f}, {inner_circle['y']:.1f}), "
                      f"半径{inner_circle['radius']:.1f}")

            # 验证结果
            if not self.circle_detector.validate_concentricity_result(result):
                self.logger.warning("检测结果验证失败，但继续处理")

            # === 手动验证同心度 ===
            print("\n" + "=" * 60)
            print("手动验证同心度:")
            print("=" * 60)

            # 从结果中获取外圆和内圆信息
            outer = result.get('outer_circle')
            inner = result.get('inner_center') or result.get('inner_circle')

            if outer and inner:
                # 提取坐标
                ox = outer.get('x', 0)
                oy = outer.get('y', 0)
                oradius = outer.get('radius', 0)

                ix = inner.get('x', 0)
                iy = inner.get('y', 0)

                # 手动计算距离
                dx = ix - ox
                dy = iy - oy
                manual_distance = math.sqrt(dx ** 2 + dy ** 2)

                # 检测器计算的距离
                detector_distance = result.get('pixel_error', 0)

                print(f"外圆中心: ({ox:.1f}, {oy:.1f}), 半径: {oradius:.1f}px")
                print(f"螺纹杆中心: ({ix:.1f}, {iy:.1f})")
                print(f"手动计算距离: {manual_distance:.2f}px")
                print(f"检测器计算距离: {detector_distance:.2f}px")
                print(f"距离差异: {abs(manual_distance - detector_distance):.2f}px")

                # 检查外圆半径是否合理
                expected_radius_range = (200, 300)  # 根据之前检测设置
                if oradius < expected_radius_range[0] or oradius > expected_radius_range[1]:
                    print(f"⚠️ 警告: 外圆半径{oradius:.1f}px超出合理范围{expected_radius_range}")

                # 计算实际毫米偏差
                pixel_to_mm = self.circle_detector.concentricity_params.get('pixel_to_mm_ratio', 0.1)
                manual_error_mm = manual_distance * pixel_to_mm
                print(f"手动计算实际偏差: {manual_error_mm:.3f}mm")

                # 判断是否合格
                quality_threshold = self.circle_detector.concentricity_params.get('quality_threshold_mm', 0.2)
                is_manual_qualified = manual_error_mm <= quality_threshold
                print(
                    f"手动验证是否合格: {'✅ 合格' if is_manual_qualified else '❌ 不合格'} (阈值: {quality_threshold}mm)")

                # 比较结果一致性
                detector_qualified = result.get('is_qualified', False)
                if is_manual_qualified != detector_qualified:
                    print(f"⚠️ 注意: 手动验证与检测器结果不一致!")

                # 绘制两个中心的连线（在原始图像上）
                debug_img = image_bgr.copy()
                cv2.circle(debug_img, (int(ox), int(oy)), int(oradius), (0, 0, 255), 3)  # 外圆
                cv2.circle(debug_img, (int(ox), int(oy)), 5, (0, 0, 255), -1)  # 外圆中心
                cv2.circle(debug_img, (int(ix), int(iy)), 10, (0, 255, 0), 3)  # 螺纹杆中心
                cv2.line(debug_img, (int(ox), int(oy)), (int(ix), int(iy)), (255, 255, 0), 2)  # 连线

                # 添加距离标签
                cv2.putText(debug_img, f"Distance: {manual_distance:.1f}px",
                            (int((ox + ix) / 2), int((oy + iy) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imwrite(f"{debug_dir}/debug_manual_verification_{timestamp}.jpg", debug_img)
                print(f"手动验证图像已保存: debug_manual_verification_{timestamp}.jpg")

            print("=" * 60 + "\n")

            # 在手动验证部分之后，添加视觉验证
            print("\n" + "=" * 60)
            print("视觉验证螺纹杆中心位置:")
            print("=" * 60)

            # 创建一个放大的ROI图像，专门显示螺纹杆区域
            roi_size = 100  # 100x100像素的ROI
            ox, oy = int(outer['x']), int(outer['y'])
            ix, iy = int(inner['x']), int(inner['y'])

            # 确保ROI在图像范围内
            roi_x1 = max(0, ix - roi_size // 2)
            roi_x2 = min(image_bgr.shape[1], ix + roi_size // 2)
            roi_y1 = max(0, iy - roi_size // 2)
            roi_y2 = min(image_bgr.shape[0], iy + roi_size // 2)

            roi_image = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2].copy()

            # 在ROI图像上标记中心
            roi_center_x = ix - roi_x1
            roi_center_y = iy - roi_y1

            cv2.circle(roi_image, (roi_center_x, roi_center_y), 5, (0, 255, 0), -1)  # 绿色中心点
            cv2.circle(roi_image, (roi_center_x, roi_center_y), 2, (255, 0, 0), -1)  # 蓝色内点

            # 添加网格线帮助定位
            grid_spacing = 10
            h, w = roi_image.shape[:2]
            for i in range(0, w, grid_spacing):
                cv2.line(roi_image, (i, 0), (i, h), (100, 100, 100), 1)
            for j in range(0, h, grid_spacing):
                cv2.line(roi_image, (0, j), (w, j), (100, 100, 100), 1)

            # 中心十字线
            cv2.line(roi_image, (roi_center_x - 10, roi_center_y), (roi_center_x + 10, roi_center_y), (255, 255, 0), 2)
            cv2.line(roi_image, (roi_center_x, roi_center_y - 10), (roi_center_x, roi_center_y + 10), (255, 255, 0), 2)

            # 保存ROI图像
            roi_path = f"{debug_dir}/debug_roi_center_{timestamp}.jpg"
            cv2.imwrite(roi_path, roi_image)
            print(f"螺纹杆中心ROI图像已保存: {roi_path}")
            print(f"ROI区域: [{roi_x1}:{roi_x2}, {roi_y1}:{roi_y2}]")
            print(f"ROI中心坐标(相对于ROI): ({roi_center_x}, {roi_center_y})")
            print("=" * 60)

            # 添加额外信息
            result['detection_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result['image_size'] = f"{image_bgr.shape[1]}x{image_bgr.shape[0]}"
            result['parameters'] = params
            result['debug_info'] = {
                'best_method': best_method,
                'contour_count': best_contour_count
            }

            # === 第五步：绘制检测结果 ===
            annotated_image = self.circle_detector.draw_concentricity_result(
                image_bgr, result
            )

            # 保存标注后的图像
            cv2.imwrite(f"{debug_dir}/debug_annotated_{timestamp}.jpg", annotated_image)

            # 将BGR转换回RGB显示
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            self.camera_viewer.set_image(annotated_image_rgb)

            # === 新增：保存检测结果到历史记录 ===
            detection_record = {
                'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'concentricity': result.get('concentricity', 0),
                'is_qualified': result.get('is_qualified', False),
                'error_mm': result.get('error_mm', 0),
                'pixel_error': result.get('pixel_error', 0),
                'inner_center': result.get('inner_center', {}),
                'outer_circle': result.get('outer_circle', {}),
                'image_size': result.get('image_size', 'N/A'),
                'debug_info': result.get('debug_info', {})
            }

            # 保存到历史记录
            self.detection_results_history.append(detection_record)
            self.current_detection_result = detection_record

            # 更新状态显示
            history_count = len(self.detection_results_history)
            self.status_label.setText(f"检测完成 (第{history_count}次检测)")

            print(f"已保存第{history_count}条检测记录到历史")

            # 发射处理完成信号
            self.processing_finished.emit(result)
            self.detection_results = result

            # 启用报告按钮（删除保存按钮相关代码）
            self.btn_report.setEnabled(True)

            self.status_label.setText("检测完成")
            print("=" * 50)
            print("检测完成！")
            print("=" * 50)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"检测过程中出错: {str(e)}\n详细错误信息:\n{error_details}")
            print(f"错误发生: {str(e)}")
            self.error_occurred.emit(f"{str(e)}\n(查看日志获取详细信息)")

        finally:
            self.btn_detect.setEnabled(True)

    def draw_detection_results(self, image, inner_circle, outer_circle, result):
        """绘制检测结果"""
        # 处理inner_center格式
        if isinstance(inner_circle, dict) and 'radius' not in inner_circle:
            # 这是inner_center格式，转换为inner_circle
            inner_circle = {
                'x': inner_circle.get('x', 0),
                'y': inner_circle.get('y', 0),
                'radius': outer_circle.get('radius', 100) * 0.25  # 估算半径
            }
        # 绘制内圆 - 从字典中获取坐标
        ix, iy = int(inner_circle['x']), int(inner_circle['y'])
        ir = int(inner_circle['radius'])
        cv2.circle(image, (ix, iy), ir, (0, 255, 0), 2)
        cv2.circle(image, (ix, iy), 2, (0, 255, 0), 3)

        # 绘制外圆 - 从字典中获取坐标
        ox, oy = int(outer_circle['x']), int(outer_circle['y'])
        oradius = int(outer_circle['radius'])
        cv2.circle(image, (ox, oy), oradius, (255, 0, 0), 2)
        cv2.circle(image, (ox, oy), 2, (255, 0, 0), 3)

        # 绘制中心连线
        cv2.line(image, (ix, iy), (ox, oy), (255, 255, 0), 2)

        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Inner: ({ix}, {iy}) R={ir}", (10, 30),
                    font, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Outer: ({ox}, {oy}) R={oradius}", (10, 60),
                    font, 0.7, (255, 0, 0), 2)

        # 从result中获取同心度值 - 处理不同的键名
        concentricity_value = result.get('concentricity', result.get('concentricity_permille', 0))
        eccentricity_value = result.get('eccentricity_mm', 0)

        cv2.putText(image, f"Concentricity: {concentricity_value:.3f}‰", (10, 90),
                    font, 0.7, (255, 255, 0), 2)
        cv2.putText(image, f"Eccentricity: {eccentricity_value:.3f}mm", (10, 120),
                    font, 0.7, (255, 255, 0), 2)

        # 判断是否合格
        is_qualified = result.get('is_qualified', result.get('is_within_tolerance', False))
        status_color = (0, 255, 0) if is_qualified else (0, 0, 255)
        status_text = "合格" if is_qualified else "不合格"
        cv2.putText(image, f"状态: {status_text}", (10, 150),
                    font, 0.7, status_color, 2)

        return image

    def generate_report(self):
        """生成报告（使用ReportGenerator类）"""
        # 如果没有历史记录，使用当前结果
        if not self.detection_results_history and self.detection_results is None:
            QMessageBox.warning(self, "警告", "没有可生成报告的结果！")
            return

        try:
            # 如果已经有报告生成器实例，先关闭旧的
            if hasattr(self, 'report_generator') and self.report_generator is not None:
                # 断开之前的信号连接
                try:
                    self.report_generator.report_generated.disconnect()
                    self.report_generator.report_error.disconnect()
                except:
                    pass
                # 关闭窗口
                self.report_generator.close()

            # 创建新的报告生成器实例
            self.report_generator = ReportGenerator()

            # === 修改：设置报告数据，优先使用历史记录 ===
            if self.detection_results_history:
                # 有历史记录，使用所有历史记录
                report_data = {
                    'all_results': self.detection_results_history,
                    'current_result': self.current_detection_result or self.detection_results_history[
                        -1] if self.detection_results_history else None,
                    'total_count': len(self.detection_results_history),
                    'generated_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                # 只有当前结果
                report_data = {
                    'all_results': [self.detection_results],
                    'current_result': self.detection_results,
                    'total_count': 1,
                    'generated_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            self.report_generator.set_report_data(report_data)

            # 设置窗口标题和公司信息
            self.report_generator.txt_report_title.setText("机械零件同心度检测报告")
            self.report_generator.txt_company_info.setText("西南科技大学\n信息与控制工程学院\n物联网工程专业")

            # 连接信号
            def on_report_generated(file_path):
                self.status_label.setText(f"报告已生成: {os.path.basename(file_path)}")
                QMessageBox.information(
                    self,
                    "成功",
                    f"报告生成成功！\n\n包含 {len(self.detection_results_history) if self.detection_results_history else 1} 条检测记录\n文件已保存到：\n{file_path}"
                )

            def on_report_error(error_msg):
                self.status_label.setText("报告生成失败")
                QMessageBox.critical(
                    self,
                    "错误",
                    f"生成报告失败：{error_msg}\n\n请检查：\n1. 保存目录是否有写入权限\n2. 磁盘空间是否充足"
                )

            self.report_generator.report_generated.connect(on_report_generated)
            self.report_generator.report_error.connect(on_report_error)

            # 设置窗口标题和最小尺寸
            report_count = len(self.detection_results_history) if self.detection_results_history else 1
            self.report_generator.setWindowTitle(f"生成检测报告 - 共{report_count}条记录")
            self.report_generator.setMinimumSize(600, 800)
            self.report_generator.resize(800, 900)

            # 将报告生成器窗口设置为应用程序模态，但允许在窗口之间切换
            self.report_generator.setWindowModality(Qt.ApplicationModal)

            # 显示报告生成器窗口（使用show()而不是exec_()）
            self.report_generator.show()

            # 将窗口提到最前面
            self.report_generator.raise_()
            self.report_generator.activateWindow()

            self.status_label.setText(f"报告生成器已打开 (共{report_count}条记录)")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"生成报告失败: {str(e)}\n详细错误信息:\n{error_details}")
            QMessageBox.warning(
                self,
                "错误",
                f"生成报告失败: {str(e)}\n(查看日志获取详细信息)"
            )

    def update_parameters(self, parameters):
        """更新参数"""
        self.logger.info(f"参数已更新: {parameters}")

    def on_processing_started(self):
        """处理开始时的处理"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.status_label.setText("正在处理...")

    def on_processing_finished(self, results):
        """处理完成时的处理"""
        self.progress_bar.setVisible(False)

        # 添加参数信息到结果中（来自第二步）
        if hasattr(self.circle_detector, 'concentricity_params'):
            params = self.circle_detector.concentricity_params
            results['concentricity_params'] = {
                'pixel_to_mm_ratio': params.get('pixel_to_mm_ratio', 0.033),
                'quality_threshold_mm': params.get('quality_threshold_mm', 0.2),
                'outer_min_radius': params.get('outer_min_radius', 250),
                'outer_max_radius': params.get('outer_max_radius', 320),
            }

        # 构建详细结果字符串（来自第五步，增强版）
        try:
            result_text = f"""
    {'=' * 50}
    检测结果详情
    {'=' * 50}
    检测时间: {results.get('detection_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
    图像尺寸: {results.get('image_size', 'N/A')}

    外筒信息:
      中心坐标: ({results['outer_circle']['x']:.1f}, {results['outer_circle']['y']:.1f})
      半径: {results['outer_circle']['radius']:.1f}px
      置信度: {results['outer_circle'].get('confidence', 0):.2%}

    螺纹杆信息:
      中心坐标: ({results['inner_center']['x']:.1f}, {results['inner_center']['y']:.1f})
      检测方法: {results['inner_center'].get('method', 'N/A')}
      是否为六边形: {'是' if results['inner_center'].get('is_hexagon', False) else '否'}

    偏差信息:
      像素偏差: {results['pixel_error']:.2f}px
      实际偏差: {results['error_mm']:.3f}mm
      相对偏差: {results['relative_error_percent']:.2f}%
      X方向偏差: {results['dx']:.2f}px
      Y方向偏差: {results['dy']:.2f}px

    检测参数:
      像素到毫米转换系数: {self.circle_detector.concentricity_params.get('pixel_to_mm_ratio', 0.033):.3f}
      合格阈值: {self.circle_detector.concentricity_params.get('quality_threshold_mm', 0.2):.2f}mm
      外筒半径范围: {self.circle_detector.concentricity_params.get('outer_min_radius', 250)}-{self.circle_detector.concentricity_params.get('outer_max_radius', 320)}px

    最终结果:
      状态: {'✅ 合格' if results['is_qualified'] else '❌ 不合格'}
    {'=' * 50}
            """
            print(result_text)
        except Exception as e:
            print(f"构建结果字符串时出错: {e}")
            # 如果出错，使用简化的输出
            pixel_to_mm = self.circle_detector.concentricity_params.get('pixel_to_mm_ratio', 0.033)
            threshold = self.circle_detector.concentricity_params.get('quality_threshold_mm', 0.2)
            print(
                f"检测完成! 实际偏差: {results.get('error_mm', 0):.3f}mm, 状态: {'合格' if results.get('is_qualified', False) else '不合格'}")

        # 更新UI显示
        self.result_viewer.update_results(results)

        # 更新状态栏（来自第五步）
        status = "合格" if results.get('is_qualified', False) else "不合格"
        self.status_label.setText(f"检测完成 - {status}")

    def on_error_occurred(self, error_msg):
        """错误发生时的处理"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "错误", error_msg)
        self.status_label.setText("处理出错")

    def calibrate_camera(self):
        """相机标定"""
        QMessageBox.information(self, "相机标定", "相机标定功能开发中...")

    def calibrate_phone_camera(self):
        """手机摄像头标定"""
        QMessageBox.information(self, "手机标定", "手机摄像头标定功能开发中...")

    def show_phone_guide(self):
        """显示手机使用指南"""
        phone_guide_text = """
        <h2>手机摄像头使用指南</h2>
        <h3>准备工作：</h3>
        <ol>
            <li>确保手机和电脑在同一WiFi网络下</li>
            <li>在手机上安装IP摄像头应用（如"IP Webcam"）</li>
            <li>打开IP摄像头应用，启动视频流服务器</li>
            <li>记下应用显示的IP地址和端口号</li>
        </ol>

        <h3>连接步骤：</h3>
        <ol>
            <li>在本软件中点击"连接手机摄像头"</li>
            <li>输入手机摄像头URL（如：http://192.168.1.100:8080/video）</li>
            <li>点击连接按钮</li>
        </ol>

        <h3>拍照技巧：</h3>
        <ul>
            <li>使用后置摄像头拍摄</li>
            <li>关闭美颜滤镜和特效</li>
            <li>确保光线充足均匀</li>
            <li>垂直拍摄，避免角度倾斜</li>
            <li>将零件放置在纯色背景上</li>
        </ul>

        <h3>常见问题：</h3>
        <ul>
            <li><b>无法连接：</b>检查WiFi网络是否一致，防火墙设置</li>
            <li><b>画面卡顿：</b>降低分辨率，确保网络稳定</li>
            <li><b>图像模糊：</b>清洁镜头，对焦清晰</li>
        </ul>
        """
        QMessageBox.information(self, "手机使用指南", phone_guide_text)

    def show_settings(self):
        """显示系统设置"""
        QMessageBox.information(self, "系统设置", "系统设置功能开发中...")

    def show_settings(self):
        """显示系统设置"""
        QMessageBox.information(self, "系统设置", "系统设置功能开发中...")

    def show_about(self):
        """显示关于信息"""
        about_text = """
        <h2>机械零件同心度检测系统 v1.0</h2>
        <p>基于机器视觉的机械零件同心度检测系统</p>
        <p>主要功能：</p>
        <ul>
            <li>非接触式同心度检测</li>
            <li>自动图像预处理</li>
            <li>高精度圆心定位</li>
            <li>实时结果可视化</li>
            <li>检测报告生成</li>
        </ul>
        <p>开发人员：严波</p>
        <p>指导教师：刘灏霖</p>
        <p>西南科技大学 信息与控制工程学院</p>
        <p>© 2026 版权所有</p>
        """
        QMessageBox.about(self, "关于", about_text)

    def show_manual(self):
        """显示使用手册"""
        QMessageBox.information(self, "使用手册", "使用手册功能开发中...")

    def show_phone_camera_tips(self):
        """显示手机拍照提示"""
        tips = """
        <h3>手机拍照使用提示：</h3>
        <p>为了获得最佳检测效果，请确保：</p>
        <ul>
            <li>使用后置摄像头拍摄</li>
            <li>关闭美颜滤镜和特效</li>
            <li>确保光线充足均匀</li>
            <li>垂直拍摄，避免角度倾斜</li>
            <li>将零件放置在纯色背景上</li>
            <li>使用高分辨率模式拍摄（1920×1080或更高）</li>
        </ul>
        <p><b>建议：</b>开启HDR模式，固定焦距，距离零件约50cm拍摄。</p>
        """

        QMessageBox.information(self, "手机拍照提示", tips)

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # 空格键拍照
        if event.key() == Qt.Key_Space:
            if self.image_controller.is_opened():
                self.capture_image()

        # Enter键开始检测
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.btn_detect.isEnabled():
                self.start_detection()

        super().keyPressEvent(event)

    def check_network_latency(self, ip_address="192.168.1.17"):
        """检查网络延迟"""
        try:
            import subprocess
            import re
            import platform

            # 根据操作系统选择ping命令
            if platform.system().lower() == "windows":
                cmd = ['ping', '-n', '3', ip_address]
            else:
                cmd = ['ping', '-c', '3', ip_address]

            # 执行ping命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            # 解析ping结果
            match = re.search(r'min/avg/max[^=]*=\s*[\d.]+/([\d.]+)/', result.stdout)
            if match:
                avg_latency = float(match.group(1))
                return avg_latency
            else:
                return None

        except subprocess.TimeoutExpired:
            self.logger.warning("Ping命令超时")
            return None
        except Exception as e:
            self.logger.error(f"检查网络延迟失败: {e}")
            return None

    def closeEvent(self, event):
        """关闭事件"""
        # 断开图像控制器
        if hasattr(self, 'image_controller'):
            self.image_controller.close()

        # 停止所有定时器
        if hasattr(self, 'phone_camera_timer'):
            self.phone_camera_timer.stop()

        event.accept()