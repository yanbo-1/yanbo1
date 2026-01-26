"""
参数设置面板模块
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QSlider, QComboBox, QLineEdit, QPushButton,
                             QFormLayout, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class ParameterPanel(QWidget):
    """参数设置面板"""

    # 信号定义
    parameters_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.init_ui()
        self.init_parameters()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 创建各个参数标签页
        self.create_preprocess_tab()
        self.create_detection_tab()
        self.create_concentricity_tab()
        self.create_camera_tab()

        layout.addWidget(self.tab_widget)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.btn_reset = QPushButton("重置参数")
        self.btn_reset.clicked.connect(self.reset_parameters)

        self.btn_save = QPushButton("保存参数")
        self.btn_save.clicked.connect(self.save_parameters)

        self.btn_load = QPushButton("加载参数")
        self.btn_load.clicked.connect(self.load_parameters)

        button_layout.addWidget(self.btn_reset)
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_load)

        layout.addLayout(button_layout)

    def create_preprocess_tab(self):
        """创建预处理参数标签页"""
        tab = QWidget()
        layout = QFormLayout(tab)

        # 亮度补偿
        self.chk_brightness = QCheckBox("启用")
        self.chk_brightness.setChecked(True)
        self.chk_brightness.stateChanged.connect(self.on_parameter_changed)
        layout.addRow("亮度补偿:", self.chk_brightness)

        # Retinex算法
        self.chk_retinex = QCheckBox("启用")
        self.chk_retinex.setChecked(True)
        self.chk_retinex.stateChanged.connect(self.on_parameter_changed)
        layout.addRow("Retinex算法:", self.chk_retinex)

        # 中值滤波
        self.spin_median = QSpinBox()
        self.spin_median.setRange(1, 15)
        self.spin_median.setValue(3)
        self.spin_median.setSingleStep(2)
        self.spin_median.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("中值滤波大小:", self.spin_median)

        # 高斯滤波
        self.spin_gaussian = QSpinBox()
        self.spin_gaussian.setRange(1, 15)
        self.spin_gaussian.setValue(5)
        self.spin_gaussian.setSingleStep(2)
        self.spin_gaussian.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("高斯滤波大小:", self.spin_gaussian)

        # Canny边缘检测低阈值
        self.spin_canny_low = QSpinBox()
        self.spin_canny_low.setRange(0, 255)
        self.spin_canny_low.setValue(50)
        self.spin_canny_low.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("Canny低阈值:", self.spin_canny_low)

        # Canny边缘检测高阈值
        self.spin_canny_high = QSpinBox()
        self.spin_canny_high.setRange(0, 255)
        self.spin_canny_high.setValue(150)
        self.spin_canny_high.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("Canny高阈值:", self.spin_canny_high)

        # 添加到标签页
        self.tab_widget.addTab(tab, "图像预处理")

    def create_detection_tab(self):
        """创建检测参数标签页"""
        tab = QWidget()
        layout = QFormLayout(tab)

        # 霍夫变换dp参数
        self.spin_hough_dp = QDoubleSpinBox()
        self.spin_hough_dp.setRange(0.5, 2.0)
        self.spin_hough_dp.setValue(1.0)
        self.spin_hough_dp.setSingleStep(0.1)
        self.spin_hough_dp.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("霍夫变换dp:", self.spin_hough_dp)

        # 最小圆心距离
        self.spin_min_dist = QSpinBox()
        self.spin_min_dist.setRange(1, 200)
        self.spin_min_dist.setValue(20)
        self.spin_min_dist.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("最小圆心距离:", self.spin_min_dist)

        # 参数1
        self.spin_param1 = QSpinBox()
        self.spin_param1.setRange(1, 200)
        self.spin_param1.setValue(50)
        self.spin_param1.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("参数1:", self.spin_param1)

        # 参数2
        self.spin_param2 = QSpinBox()
        self.spin_param2.setRange(1, 200)
        self.spin_param2.setValue(30)
        self.spin_param2.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("参数2:", self.spin_param2)

        # 最小半径
        self.spin_min_radius = QSpinBox()
        self.spin_min_radius.setRange(1, 500)
        self.spin_min_radius.setValue(10)
        self.spin_min_radius.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("最小半径:", self.spin_min_radius)

        # 最大半径
        self.spin_max_radius = QSpinBox()
        self.spin_max_radius.setRange(1, 1000)
        self.spin_max_radius.setValue(100)
        self.spin_max_radius.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("最大半径:", self.spin_max_radius)

        self.tab_widget.addTab(tab, "圆心检测")

    def create_concentricity_tab(self):
        """创建同心度参数标签页"""
        tab = QWidget()
        layout = QFormLayout(tab)

        # 像素到毫米转换系数
        self.spin_pixel_to_mm = QDoubleSpinBox()
        self.spin_pixel_to_mm.setRange(0.001, 1.0)
        self.spin_pixel_to_mm.setValue(0.1)
        self.spin_pixel_to_mm.setDecimals(3)
        self.spin_pixel_to_mm.setSingleStep(0.001)
        self.spin_pixel_to_mm.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("像素到毫米:", self.spin_pixel_to_mm)

        # 基准半径
        self.spin_reference_radius = QDoubleSpinBox()
        self.spin_reference_radius.setRange(0.1, 1000.0)
        self.spin_reference_radius.setValue(50.0)
        self.spin_reference_radius.setDecimals(1)
        self.spin_reference_radius.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("基准半径(mm):", self.spin_reference_radius)

        # 公差
        self.spin_tolerance = QDoubleSpinBox()
        self.spin_tolerance.setRange(0.01, 5.0)
        self.spin_tolerance.setValue(0.2)
        self.spin_tolerance.setDecimals(2)
        self.spin_tolerance.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("公差(mm):", self.spin_tolerance)

        # 同心度单位选择
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["‰ (千分之)", "% (百分比)"])
        self.combo_unit.currentIndexChanged.connect(self.on_parameter_changed)
        layout.addRow("单位:", self.combo_unit)

        self.tab_widget.addTab(tab, "同心度计算")

    def create_camera_tab(self):
        """创建相机参数标签页"""
        tab = QWidget()
        layout = QFormLayout(tab)

        # 相机选择
        self.combo_camera = QComboBox()
        self.combo_camera.addItems(["相机0", "相机1", "相机2"])
        self.combo_camera.currentIndexChanged.connect(self.on_parameter_changed)
        layout.addRow("相机选择:", self.combo_camera)

        # 分辨率
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["640x480", "1280x720", "1920x1080"])
        self.combo_resolution.currentIndexChanged.connect(self.on_parameter_changed)
        layout.addRow("分辨率:", self.combo_resolution)

        # 帧率
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(30)
        self.spin_fps.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("帧率(FPS):", self.spin_fps)

        # 曝光
        self.slider_exposure = QSlider(Qt.Horizontal)
        self.slider_exposure.setRange(-10, 10)
        self.slider_exposure.setValue(0)
        self.slider_exposure.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("曝光补偿:", self.slider_exposure)

        # 亮度
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(0, 100)
        self.slider_brightness.setValue(50)
        self.slider_brightness.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("亮度:", self.slider_brightness)

        # 对比度
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 100)
        self.slider_contrast.setValue(50)
        self.slider_contrast.valueChanged.connect(self.on_parameter_changed)
        layout.addRow("对比度:", self.slider_contrast)

        self.tab_widget.addTab(tab, "相机设置")

    def init_parameters(self):
        """初始化参数"""
        self.update_parameters()

    def update_parameters(self):
        """更新参数字典"""
        self.parameters = {
            'preprocess': {
                'brightness_compensation': self.chk_brightness.isChecked(),
                'retinex_enabled': self.chk_retinex.isChecked(),
                'median_filter_size': self.spin_median.value(),
                'gaussian_filter_size': self.spin_gaussian.value(),
                'canny_low': self.spin_canny_low.value(),
                'canny_high': self.spin_canny_high.value()
            },
            'circle_detection': {
                'hough_dp': self.spin_hough_dp.value(),
                'hough_min_dist': self.spin_min_dist.value(),
                'hough_param1': self.spin_param1.value(),
                'hough_param2': self.spin_param2.value(),
                'hough_min_radius': self.spin_min_radius.value(),
                'hough_max_radius': self.spin_max_radius.value()
            },
            'concentricity': {
                'pixel_to_mm': self.spin_pixel_to_mm.value(),
                'reference_radius_mm': self.spin_reference_radius.value(),
                'tolerance': self.spin_tolerance.value(),
                'unit': self.combo_unit.currentText()
            },
            'camera': {
                'camera_index': self.combo_camera.currentIndex(),
                'resolution': self.combo_resolution.currentText(),
                'fps': self.spin_fps.value(),
                'exposure': self.slider_exposure.value(),
                'brightness': self.slider_brightness.value(),
                'contrast': self.slider_contrast.value()
            }
        }

    def on_parameter_changed(self):
        """参数改变时的处理"""
        self.update_parameters()
        self.parameters_changed.emit(self.parameters)

    def get_parameters(self):
        """获取当前参数"""
        return self.parameters.copy()

    def set_parameters(self, parameters):
        """设置参数"""
        if 'preprocess' in parameters:
            preprocess = parameters['preprocess']
            self.chk_brightness.setChecked(preprocess.get('brightness_compensation', True))
            self.chk_retinex.setChecked(preprocess.get('retinex_enabled', True))
            self.spin_median.setValue(preprocess.get('median_filter_size', 3))
            self.spin_gaussian.setValue(preprocess.get('gaussian_filter_size', 5))
            self.spin_canny_low.setValue(preprocess.get('canny_low', 50))
            self.spin_canny_high.setValue(preprocess.get('canny_high', 150))

        if 'circle_detection' in parameters:
            detection = parameters['circle_detection']
            self.spin_hough_dp.setValue(detection.get('hough_dp', 1.0))
            self.spin_min_dist.setValue(detection.get('hough_min_dist', 20))
            self.spin_param1.setValue(detection.get('hough_param1', 50))
            self.spin_param2.setValue(detection.get('hough_param2', 30))
            self.spin_min_radius.setValue(detection.get('hough_min_radius', 10))
            self.spin_max_radius.setValue(detection.get('hough_max_radius', 100))

        if 'concentricity' in parameters:
            concentricity = parameters['concentricity']
            self.spin_pixel_to_mm.setValue(concentricity.get('pixel_to_mm', 0.1))
            self.spin_reference_radius.setValue(concentricity.get('reference_radius_mm', 50.0))
            self.spin_tolerance.setValue(concentricity.get('tolerance', 0.2))

            unit = concentricity.get('unit', '‰ (千分之)')
            index = self.combo_unit.findText(unit)
            if index >= 0:
                self.combo_unit.setCurrentIndex(index)

        self.update_parameters()

    def reset_parameters(self):
        """重置参数到默认值"""
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
                'hough_dp': 1.0,
                'hough_min_dist': 20,
                'hough_param1': 50,
                'hough_param2': 30,
                'hough_min_radius': 10,
                'hough_max_radius': 100
            },
            'concentricity': {
                'pixel_to_mm': 0.1,
                'reference_radius_mm': 50.0,
                'tolerance': 0.2,
                'unit': '‰ (千分之)'
            }
        }
        self.set_parameters(default_params)

    def save_parameters(self):
        """保存参数到文件"""
        # 这里需要实现文件保存逻辑
        print("保存参数到文件")

    def load_parameters(self):
        """从文件加载参数"""
        # 这里需要实现文件加载逻辑
        print("从文件加载参数")