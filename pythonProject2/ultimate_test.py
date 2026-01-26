# ultimate_test.py
import sys
import os

sys.path.append('.')


def test_imports():
    """测试所有关键导入"""
    print("测试所有关键模块导入...")
    print("=" * 60)

    imports = [
        ("utils.image_loader", "load_image_chinese"),
        ("utils.file_io", "save_results_to_csv"),
        ("utils.file_io", "save_image_with_annotations"),
        ("core.preprocess", "ImagePreprocessor"),
        ("core.circle_detection", "CircleDetector"),
        ("core.concentricity_calc", "ConcentricityCalculator"),
    ]

    all_ok = True
    for module, item in imports:
        try:
            exec(f"from {module} import {item}")
            print(f"✓ {module}.{item} 导入成功")
        except ImportError as e:
            print(f"✗ {module}.{item} 导入失败: {e}")
            all_ok = False

    return all_ok


def test_methods():
    """测试所有关键方法"""
    print("\n测试所有关键方法...")
    print("=" * 60)

    methods_to_test = [
        ("ImagePreprocessor", "process"),
        ("CircleDetector", "detect_circles"),
        ("ConcentricityCalculator", "calculate"),
    ]

    all_ok = True
    for class_name, method_name in methods_to_test:
        try:
            if class_name == "ImagePreprocessor":
                from core.preprocess import ImagePreprocessor
                obj = ImagePreprocessor()
            elif class_name == "CircleDetector":
                from core.circle_detection import CircleDetector
                obj = CircleDetector()
            elif class_name == "ConcentricityCalculator":
                from core.concentricity_calc import ConcentricityCalculator
                obj = ConcentricityCalculator()

            if hasattr(obj, method_name):
                print(f"✓ {class_name}.{method_name}() 存在")
            else:
                print(f"✗ {class_name}.{method_name}() 缺失")
                all_ok = False

        except Exception as e:
            print(f"✗ 测试 {class_name}.{method_name} 失败: {e}")
            all_ok = False

    return all_ok


def simulate_workflow():
    """模拟完整工作流程"""
    print("\n模拟完整检测流程...")
    print("=" * 60)

    try:
        import numpy as np
        import cv2

        # 1. 创建测试图像
        test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.circle(test_image, (150, 150), 50, (100, 100, 100), -1)
        cv2.circle(test_image, (155, 155), 80, (50, 50, 50), -1)

        print("✓ 创建测试图像")

        # 2. 预处理
        from core.preprocess import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process(
            test_image,
            brightness_compensation=True,
            median_filter_size=3,
            gaussian_filter_size=5,
            canny_low=50,
            canny_high=150
        )
        print(f"✓ 预处理完成: {processed.shape}")

        # 3. 圆检测
        from core.circle_detection import CircleDetector
        detector = CircleDetector()
        circles = detector.detect_circles(processed, target_count=2)
        print(f"✓ 圆检测完成: 找到 {len(circles)} 个圆")

        # 4. 同心度计算
        if len(circles) >= 2:
            from core.concentricity_calc import ConcentricityCalculator
            calculator = ConcentricityCalculator()
            result = calculator.calculate(
                inner_circle=circles[0],
                outer_circle=circles[1],
                pixel_to_mm=0.1,
                reference_radius_mm=50.0
            )

            print(f"✓ 同心度计算完成:")
            print(f"  同心度: {result.get('concentricity', 'N/A'):.2f}‰")
            print(f"  偏心距: {result.get('eccentricity_mm', 'N/A'):.3f}mm")
            print(f"  是否合格: {result.get('is_qualified', 'N/A')}")

        return True

    except Exception as e:
        print(f"✗ 流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("终极兼容性测试")
    print("=" * 60)

    # 测试1：导入
    if not test_imports():
        print("\n❌ 导入测试失败，请检查模块路径")
        return False

    # 测试2：方法
    if not test_methods():
        print("\n❌ 方法测试失败，请添加缺失的方法")
        return False

    # 测试3：工作流程
    if not simulate_workflow():
        print("\n❌ 工作流程测试失败")
        return False

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！程序应该可以正常运行了。")
    print("\n现在可以运行: python main.py")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)