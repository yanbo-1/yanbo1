# 同心度检测系统/test_component_step.py
"""
逐步添加组件测试
"""
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_step_by_step():
    """逐步测试各个组件"""
    print("=" * 50)
    print("逐步组件测试")
    print("=" * 50)

    # 第1步：测试基本窗口
    print("\n第1步：测试基本窗口...")
    from PyQt5.QtWidgets import QMainWindow, QApplication

    # 必须创建QApplication实例
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    try:
        window = QMainWindow()
        window.setWindowTitle("测试窗口")
        window.setGeometry(100, 100, 400, 300)
        print("[OK] 基本窗口创建成功")
        window.close()
    except Exception as e:
        print(f"[FAIL] 基本窗口失败: {e}")
        return

    # 第2步：测试UI组件导入
    print("\n第2步：测试UI组件导入...")
    try:
        from ui.camera_viewer import CameraViewer
        print("[OK] CameraViewer导入成功")
    except Exception as e:
        print(f"[FAIL] CameraViewer导入失败: {e}")

    try:
        from ui.parameter_panel import ParameterPanel
        print("[OK] ParameterPanel导入成功")
    except Exception as e:
        print(f"[FAIL] ParameterPanel导入失败: {e}")

    try:
        from ui.result_viewer import ResultViewer
        print("[OK] ResultViewer导入成功")
    except Exception as e:
        print(f"[FAIL] ResultViewer导入失败: {e}")

    # 第3步：测试核心模块导入
    print("\n第3步：测试核心模块导入...")
    try:
        from core.camera import ImageController
        print("[OK] ImageController导入成功")
    except Exception as e:
        print(f"[FAIL] ImageController导入失败: {e}")

    try:
        from core.circle_detection import CircleDetector
        print("[OK] CircleDetector导入成功")
    except Exception as e:
        print(f"[FAIL] CircleDetector导入失败: {e}")

    # 第4步：测试工具模块导入
    print("\n第4步：测试工具模块导入...")
    try:
        from utils.logger import setup_logger
        logger = setup_logger('test')
        print("[OK] Logger导入成功")
    except Exception as e:
        print(f"[FAIL] Logger导入失败: {e}")

    print("\n" + "=" * 50)
    print("逐步测试完成")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(test_step_by_step())