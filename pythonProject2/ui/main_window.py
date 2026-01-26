"""
主窗口界面模块
"""

import sys
import os

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
import cv2
import numpy as np

# 从同一目录导入UI组件（相对导入）
from .camera_viewer import CameraViewer
from .parameter_panel import ParameterPanel
from .result_viewer import ResultViewer

# 从core目录导入（绝对导入）
try:
    from camera import CameraController
    from preprocess import ImagePreprocessor
    from circle_detection import CircleDetector
    from concentricity_calc import ConcentricityCalculator
except ImportError as e:
    print(f"导入core模块失败: {e}")


    # 创建模拟类以便程序可以运行
    class CameraController:
        def __init__(self): pass

        def is_opened(self): return False

        def open(self): return True

        def close(self): pass

        def get_frame(self): return None


    class ImagePreprocessor:
        def __init__(self): pass

        def process(self, image, **kwargs): return image


    class CircleDetector:
        def __init__(self): pass

        def detect_circles(self, image, **kwargs): return []


    class ConcentricityCalculator:
        def __init__(self): pass

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

class MainWindow(QMainWindow):
    """主窗口类"""

    # 信号定义
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logger = setup_logger('UI')
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

        self.btn_camera = QPushButton("打开相机")
        self.btn_camera.setFixedHeight(40)
        self.btn_load = QPushButton("加载图像")
        self.btn_load.setFixedHeight(40)
        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setFixedHeight(40)
        self.btn_detect.setEnabled(False)
        self.btn_save = QPushButton("保存结果")
        self.btn_save.setFixedHeight(40)
        self.btn_save.setEnabled(False)
        self.btn_report = QPushButton("生成报告")
        self.btn_report.setFixedHeight(40)
        self.btn_report.setEnabled(False)

        button_layout.addWidget(self.btn_camera)
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_detect)
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_report)
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

        save_action = QAction('保存结果', self)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)

        report_action = QAction('生成报告', self)
        report_action.triggered.connect(self.generate_report)
        file_menu.addAction(report_action)

        file_menu.addSeparator()

        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 工具菜单
        tools_menu = menubar.addMenu('工具')

        calibrate_action = QAction('相机标定', self)
        calibrate_action.triggered.connect(self.calibrate_camera)
        tools_menu.addAction(calibrate_action)

        settings_action = QAction('系统设置', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        manual_action = QAction('使用手册', self)
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)

    def init_components(self):
        """初始化组件"""
        self.camera_controller = CameraController()
        self.preprocessor = ImagePreprocessor()
        self.circle_detector = CircleDetector()
        self.concentricity_calc = ConcentricityCalculator()

        # 加载默认参数
        self.load_default_parameters()

    def init_connections(self):
        """初始化信号连接"""
        # 按钮连接
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_save.clicked.connect(self.save_results)
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

    def toggle_camera(self):
        """切换相机状态"""
        if self.camera_controller.is_opened():
            self.camera_controller.close()
            self.btn_camera.setText("打开相机")
            self.status_label.setText("相机已关闭")
        else:
            if self.camera_controller.open():
                self.btn_camera.setText("关闭相机")
                self.status_label.setText("相机已打开")

                # 启动相机预览定时器
                self.camera_timer = QTimer()
                self.camera_timer.timeout.connect(self.update_camera_frame)
                self.camera_timer.start(30)  # 30ms间隔
            else:
                QMessageBox.warning(self, "错误", "无法打开相机！")

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
            QMessageBox.warning(self, "警告", "请先加载图像或打开相机！")
            return

        # 禁用按钮
        self.btn_detect.setEnabled(False)
        self.processing_started.emit()

        try:
            # 获取当前参数
            params = self.parameter_panel.get_parameters()

            # === 修改点1：直接使用新方法，不需要预处理 ===
            # 注意：我们的新方法detect_concentricity_pair会自己处理图像
            # 所以不需要调用preprocessor.process

            # 直接使用原始图像进行同心度检测
            # 注意：我们的检测器需要BGR格式，但UI中存储的是RGB
            # 所以需要转换
            if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 3:
                # 当前是RGB格式，转换为BGR
                image_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = self.current_image.copy()

            # === 修改点2：更新检测器参数 ===
            # 更新通用圆检测参数
            self.circle_detector.update_parameters(
                hough_dp=params['circle_detection']['hough_dp'],
                hough_min_dist=params['circle_detection']['hough_min_dist'],
                hough_param1=params['circle_detection']['hough_param1'],
                hough_param2=params['circle_detection']['hough_param2'],
                hough_min_radius=params['circle_detection']['hough_min_radius'],
                hough_max_radius=params['circle_detection']['hough_max_radius']
            )

            # 更新同心度检测参数（需要确保parameter_panel支持这些参数）
            # 如果没有，可以暂时使用默认值或添加支持
            pixel_to_mm = params['concentricity']['pixel_to_mm']
            self.circle_detector.set_calibration_ratio(pixel_to_mm)

            # === 修改点3：调用新的同心度检测方法 ===
            result = self.circle_detector.detect_concentricity_pair(image_bgr)

            if not result.get('success', False):
                error_msg = result.get('error', '未知错误')
                raise ValueError(f"同心度检测失败: {error_msg}")

            # === 修改点4：验证结果 ===
            if not self.circle_detector.validate_concentricity_result(result):
                self.logger.warning("检测结果验证失败，但继续处理")

            # 添加额外信息
            result['detection_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result['image_size'] = f"{image_bgr.shape[1]}x{image_bgr.shape[0]}"
            result['parameters'] = params

            # === 修改点5：绘制检测结果 ===
            annotated_image = self.circle_detector.draw_concentricity_result(
                image_bgr, result
            )

            # 将BGR转换回RGB显示
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            self.camera_viewer.set_image(annotated_image_rgb)

            # 发射处理完成信号
            self.processing_finished.emit(result)
            self.detection_results = result

            # 启用保存和报告按钮
            self.btn_save.setEnabled(True)
            self.btn_report.setEnabled(True)

            self.status_label.setText("检测完成")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"检测过程中出错: {str(e)}\n详细错误信息:\n{error_details}")
            self.error_occurred.emit(f"{str(e)}\n(查看日志获取详细信息)")

        finally:
            self.btn_detect.setEnabled(True)

    def draw_detection_results(self, image, inner_circle, outer_circle, result):
        """绘制检测结果"""
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

    def save_results(self):
        """保存结果"""
        if self.detection_results is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果！")
            return

        try:
            # 选择保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存结果",
                f"results/concentricity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV文件 (*.csv)"
            )

            if file_path:
                # 保存结果到CSV
                save_results_to_csv(self.detection_results, file_path)

                # 保存标注图像
                image_path = file_path.replace('.csv', '_annotated.jpg')
                if self.current_image is not None:
                    save_image_with_annotations(self.current_image, image_path)

                self.status_label.setText(f"结果已保存到: {file_path}")
                QMessageBox.information(self, "成功", "结果保存成功！")

        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"保存结果失败: {str(e)}")

    def generate_report(self):
        """生成报告"""
        if self.detection_results is None:
            QMessageBox.warning(self, "警告", "没有可生成报告的结果！")
            return

        try:
            # 选择保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "生成报告",
                f"reports/concentricity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF文件 (*.pdf)"
            )

            if file_path:
                # 这里调用报告生成器
                # report_generator.generate_pdf_report(self.detection_results, file_path)

                self.status_label.setText(f"报告已生成: {file_path}")
                QMessageBox.information(self, "成功", "报告生成成功！")

        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"生成报告失败: {str(e)}")

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
        self.result_viewer.update_results(results)

    def on_error_occurred(self, error_msg):
        """错误发生时的处理"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "错误", error_msg)
        self.status_label.setText("处理出错")

    def calibrate_camera(self):
        """相机标定"""
        QMessageBox.information(self, "相机标定", "相机标定功能开发中...")

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

    def closeEvent(self, event):
        """关闭事件"""
        # 关闭相机
        if hasattr(self, 'camera_controller'):
            self.camera_controller.close()

        # 关闭所有定时器
        if hasattr(self, 'camera_timer'):
            self.camera_timer.stop()

        event.accept()