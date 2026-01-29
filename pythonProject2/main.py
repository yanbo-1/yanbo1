#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械零件同心度检测系统 - 主程序入口
作者：严波
学号：5320240755
学院：信息与控制工程学院
"""

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow
from utils.logger import setup_logger

# 配置日志
logger = setup_logger('main_app')


class ConcentricityDetectionApp:
    """同心度检测系统主应用类"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("同心度检测系统")
        self.app.setOrganizationName("西南科技大学")
        self.app.setOrganizationDomain("swust.edu.cn")

        # 设置窗口图标（如果有）
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "app_icon.png")
            if os.path.exists(icon_path):
                self.app.setWindowIcon(QIcon(icon_path))
        except:
            pass

        # 创建主窗口
        self.main_window = MainWindow()

    def run(self):
        """运行应用程序"""
        logger.info("启动同心度检测系统...")

        # 显示主窗口
        self.main_window.show()

        # 启动定时器（用于后台任务）
        self._start_timers()

        # 运行应用
        exit_code = self.app.exec_()

        logger.info("应用程序正常退出")
        return exit_code

    def _start_timers(self):
        """启动后台定时器"""
        # 状态更新定时器（每秒更新一次）
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # 1秒

        # 内存监控定时器（每10秒一次）
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self._monitor_memory)
        self.memory_timer.start(10000)  # 10秒

    def _update_status(self):
        """更新系统状态"""
        if hasattr(self.main_window, 'update_status'):
            self.main_window.update_status()

    def _monitor_memory(self):
        """监控内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 500:  # 如果内存超过500MB
                logger.warning(f"内存使用较高: {memory_mb:.1f} MB")
        except ImportError:
            pass  # psutil未安装，忽略内存监控


def main():
    """主函数"""
    try:
        # 创建并运行应用
        app = ConcentricityDetectionApp()
        return app.run()
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}", exc_info=True)
        print(f"错误: {e}")
        print("请检查依赖是否安装完整: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    # 设置命令行参数
    import argparse

    parser = argparse.ArgumentParser(description='机械零件同心度检测系统')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--image', type=str, help='直接打开指定图像文件')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID（默认0）')

    args = parser.parse_args()

    # 设置调试模式
    if args.debug:
        os.environ['DEBUG'] = '1'
        logging.getLogger().setLevel(logging.DEBUG)

    # 运行主程序
    sys.exit(main())
