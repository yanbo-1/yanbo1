# 同心度检测系统/test_minimal_window.py
"""
最小化主窗口测试 - 排除所有复杂组件
"""
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QPushButton, QStatusBar, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont


class MinimalMainWindow(QMainWindow):
    """最小化主窗口 - 只有最基本的功能"""

    def __init__(self):
        super().__init__()

        print("正在初始化最小化主窗口...")

        # 设置窗口属性
        self.setWindowTitle('同心度检测系统 - 最小化测试版')
        self.setGeometry(100, 100, 800, 600)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 简单布局
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)

        # 标题
        title = QLabel("同心度检测系统")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 说明
        info = QLabel("这是一个最小化测试版本，用于定位程序崩溃问题")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        # 测试按钮1：基本功能
        btn_test1 = QPushButton("测试按钮1 - 基本功能")
        btn_test1.clicked.connect(self.test_button1)
        layout.addWidget(btn_test1)

        # 测试按钮2：模拟摄像头
        btn_test2 = QPushButton("测试按钮2 - 模拟摄像头")
        btn_test2.clicked.connect(self.test_phone_camera)
        layout.addWidget(btn_test2)

        # 测试按钮3：加载图像
        btn_test3 = QPushButton("测试按钮3 - 加载图像")
        btn_test3.clicked.connect(self.load_image_test)
        layout.addWidget(btn_test3)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 添加空间
        layout.addStretch()

        print("最小化主窗口初始化完成")

    def test_button1(self):
        """测试按钮1"""
        self.status_bar.showMessage("按钮1被点击")
        QMessageBox.information(self, "测试", "按钮1功能正常！")

    def test_phone_camera(self):
        """测试手机摄像头连接"""
        self.status_bar.showMessage("正在测试手机摄像头...")

        try:
            import cv2

            # 使用OpenCV连接手机摄像头
            phone_url = "http://192.168.1.17:8080/video"
            cap = cv2.VideoCapture(phone_url)

            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret:
                    QMessageBox.information(self, "成功",
                                            "手机摄像头连接成功！\n"
                                            f"图像尺寸: {frame.shape}\n"
                                            f"分辨率: {frame.shape[1]}×{frame.shape[0]}")
                else:
                    QMessageBox.warning(self, "警告", "摄像头已连接但无法读取图像")
            else:
                QMessageBox.warning(self, "错误", "无法连接手机摄像头")

        except Exception as e:
            QMessageBox.warning(self, "错误", f"摄像头测试失败: {str(e)}")

        self.status_bar.showMessage("就绪")

    def load_image_test(self):
        """测试加载图像"""
        self.status_bar.showMessage("加载图像测试")
        QMessageBox.information(self, "测试", "加载图像功能正常！")


def main():
    print("=" * 50)
    print("最小化主窗口测试")
    print("=" * 50)

    app = QApplication(sys.argv)

    try:
        window = MinimalMainWindow()
        window.show()
        print("✅ 主窗口创建并显示成功")
        print("如果能看到窗口，说明基本PyQt5功能正常")
        return app.exec_()
    except Exception as e:
        print(f"❌ 创建主窗口时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())