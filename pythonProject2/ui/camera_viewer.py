"""
图像显示控件模块
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import numpy as np
import cv2


class CameraViewer(QWidget):
    """图像显示控件"""

    # 信号定义
    image_clicked = pyqtSignal(QPoint)
    image_loaded = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.current_image = None
        self.original_image = None
        self.scale_factor = 1.0
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        # 创建图像标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; }")

        # 设置鼠标追踪
        self.image_label.setMouseTracking(True)

        # 添加到滚动区域
        self.scroll_area.setWidget(self.image_label)

        layout.addWidget(self.scroll_area)

        # 信息标签
        self.info_label = QLabel("未加载图像")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

    def set_image(self, image):
        """设置显示的图像"""
        if image is None:
            return

        self.original_image = image.copy()
        self.current_image = image.copy()

        # 转换图像格式
        if len(image.shape) == 3:
            height, width, channels = image.shape
            bytes_per_line = channels * width

            # 确保图像是uint8类型
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # 转换为QImage
            q_image = QImage(image.data, width, height, bytes_per_line,
                             QImage.Format_RGB888 if channels == 3 else QImage.Format_Grayscale8)
        else:
            height, width = image.shape
            bytes_per_line = width

            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # 创建QPixmap并显示
        pixmap = QPixmap.fromImage(q_image)

        # 根据窗口大小调整图像大小
        scaled_pixmap = self.scale_pixmap(pixmap)
        self.image_label.setPixmap(scaled_pixmap)

        # 更新信息
        self.info_label.setText(f"图像大小: {width}x{height} | 缩放: {self.scale_factor:.2f}x")

        # 发射信号
        self.image_loaded.emit(image)

    def scale_pixmap(self, pixmap):
        """缩放图像以适应显示区域"""
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()

        if pixmap_size.width() > label_size.width() or pixmap_size.height() > label_size.height():
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.scale_factor = min(label_size.width() / pixmap_size.width(),
                                    label_size.height() / pixmap_size.height())
        else:
            scaled_pixmap = pixmap
            self.scale_factor = 1.0

        return scaled_pixmap

    def get_original_image(self):
        """获取原始图像"""
        return self.original_image.copy() if self.original_image is not None else None

    def get_current_image(self):
        """获取当前显示的图像"""
        return self.current_image.copy() if self.current_image is not None else None

    def clear(self):
        """清空显示"""
        self.image_label.clear()
        self.original_image = None
        self.current_image = None
        self.info_label.setText("未加载图像")

    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.LeftButton and self.current_image is not None:
            # 获取相对于图像标签的位置
            label_pos = self.image_label.mapFrom(self, event.pos())
            pixmap = self.image_label.pixmap()

            if pixmap is not None:
                # 计算图像在标签中的位置
                label_size = self.image_label.size()
                pixmap_size = pixmap.size()

                # 计算偏移量
                offset_x = (label_size.width() - pixmap_size.width()) // 2
                offset_y = (label_size.height() - pixmap_size.height()) // 2

                # 计算图像坐标
                image_x = label_pos.x() - offset_x
                image_y = label_pos.y() - offset_y

                # 转换为原始图像坐标
                if 0 <= image_x < pixmap_size.width() and 0 <= image_y < pixmap_size.height():
                    original_x = int(image_x / self.scale_factor)
                    original_y = int(image_y / self.scale_factor)

                    self.image_clicked.emit(QPoint(original_x, original_y))

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)

        # 重新缩放当前图像
        if self.current_image is not None:
            self.set_image(self.current_image)