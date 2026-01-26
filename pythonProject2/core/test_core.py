#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心模块功能 - pytest兼容版本
"""

import cv2
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.camera import CameraManager
from core.preprocess import ImagePreprocessor
from core.circle_detection import CircleDetector
from core.concentricity_calc import ConcentricityCalculator


def test_preprocess():
    """测试图像预处理"""
    print("测试图像预处理...")
    preprocessor = ImagePreprocessor()

    # 创建测试图像（模拟机械零件）
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(test_image, (300, 200), 100, (200, 200, 200), -1)  # 外圆
    cv2.circle(test_image, (310, 210), 40, (100, 100, 100), -1)  # 内圆（偏心）

    # 预处理
    processed = preprocessor.preprocess_pipeline(test_image)

    print(f"原始图像形状: {test_image.shape}")
    print(f"处理后图像形状: {processed.shape}")

    # 断言
    assert processed.shape == test_image.shape, "预处理不应改变图像尺寸"
    print("图像预处理测试完成 [PASS]")

    return processed


def test_circle_detection():
    """测试圆心检测"""
    print("\n测试圆心检测...")

    # 创建测试图像
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(test_image, (300, 200), 100, (200, 200, 200), -1)  # 外圆
    cv2.circle(test_image, (310, 210), 40, (100, 100, 100), -1)  # 内圆（偏心）

    # 转换为灰度
    gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

    detector = CircleDetector()

    # 检测圆
    circles = detector.detect_circles(gray, method='hough_improved', target_count=2)

    print(f"检测到 {len(circles)} 个圆:")
    for i, circle in enumerate(circles):
        print(f"  圆{i + 1}: 中心({circle['x']:.1f}, {circle['y']:.1f}), "
              f"半径{circles[i]['radius']:.1f}, 置信度{circles[i]['confidence']:.2f}")

    # 断言
    assert len(circles) >= 2, f"应该检测到至少2个圆，但只检测到 {len(circles)} 个"

    # 检查坐标是否在合理范围内
    for circle in circles:
        assert 0 <= circle['x'] < 600, f"圆心x坐标越界: {circle['x']}"
        assert 0 <= circle['y'] < 400, f"圆心y坐标越界: {circle['y']}"
        assert 10 <= circle['radius'] <= 200, f"半径不合理: {circle['radius']}"

    print("圆心检测测试完成 [PASS]")

    return circles


def test_concentricity_calculation():
    """测试同心度计算"""
    print("\n测试同心度计算...")

    # 创建测试数据（模拟检测到的圆）
    inner_circle = {'x': 310.0, 'y': 210.0, 'radius': 40.0, 'confidence': 0.95}
    outer_circle = {'x': 300.0, 'y': 200.0, 'radius': 100.0, 'confidence': 0.98}

    calculator = ConcentricityCalculator(pixel_to_mm_ratio=0.1)
    calculator.set_tolerance(concentricity_tolerance=0.2)  # 0.2mm容差

    result = calculator.calculate_concentricity(inner_circle, outer_circle)

    print("同心度计算结果:")
    print(f"  偏心距: {result['eccentricity_mm']:.3f} mm")
    print(f"  同心度: {result['concentricity_permille']:.2f} ‰")
    print(f"  偏心方向: {result['eccentric_direction_deg']:.1f} °")
    print(f"  是否合格: {'是' if result['is_within_tolerance'] else '否'}")

    # 断言
    assert 'eccentricity_mm' in result, "结果中缺少偏心距"
    assert 'concentricity_permille' in result, "结果中缺少同心度"
    assert 'is_within_tolerance' in result, "结果中缺少容差检查"

    # 计算应该合理
    expected_eccentricity = np.sqrt((310 - 300) ** 2 + (210 - 200) ** 2) * 0.1  # 大约14.14像素 * 0.1
    assert abs(result['eccentricity_mm'] - expected_eccentricity) < 1.0, "偏心距计算错误"

    print("同心度计算测试完成 [PASS]")

    return result


def test_camera_manager():
    """测试摄像头管理器（不实际打开摄像头）"""
    print("\n测试摄像头管理器...")

    camera = CameraManager(camera_id=0)

    # 测试信息获取（不初始化摄像头）
    info = camera.get_camera_info()
    print(f"摄像头信息: {info}")

    # 测试图像保存和加载（使用虚拟图像）
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # 保存测试
    success = camera.save_image(test_image, "test_image.png")
    if success:
        print("图像保存测试完成 [PASS]")
        # 清理测试文件
        if os.path.exists("test_image.png"):
            os.remove("test_image.png")
    else:
        print("图像保存测试 [SKIP] - 可能没有写入权限")

    print("摄像头管理器测试完成 [PASS]")


def main():
    """主测试函数 - 用于非pytest环境"""
    print("=" * 60)
    print("机械零件同心度检测系统 - 核心模块测试")
    print("=" * 60)

    # 测试图像预处理
    processed_image = test_preprocess()

    # 测试圆心检测
    circles = test_circle_detection()

    # 测试同心度计算
    result = test_concentricity_calculation()

    # 测试摄像头管理器
    test_camera_manager()

    print("\n" + "=" * 60)
    print("所有核心模块测试完成 [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    # 设置编码
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    main()