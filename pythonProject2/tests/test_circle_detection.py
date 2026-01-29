"""
测试圆检测模块
"""
import sys
import os

# 添加core目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

import cv2
import numpy as np
import pytest
from circle_detection import CircleDetector


class TestCircleDetector:
    """测试圆检测器类"""

    @pytest.fixture
    def detector(self):
        """创建圆检测器实例"""
        return CircleDetector()

    @pytest.fixture
    def sample_circle_image(self):
        """创建包含圆的测试图像"""
        # 创建一个黑色背景的图像
        img = np.zeros((300, 300, 3), dtype=np.uint8)

        # 在中心画一个圆
        center = (150, 150)
        radius = 80
        color = (255, 255, 255)  # 白色
        thickness = 2

        cv2.circle(img, center, radius, color, thickness)

        return img

    @pytest.fixture
    def sample_multiple_circles_image(self):
        """创建包含多个圆的测试图像"""
        img = np.zeros((400, 400, 3), dtype=np.uint8)

        # 画多个圆
        circles_to_draw = [
            ((100, 100), 30),
            ((300, 100), 40),
            ((200, 300), 50),
        ]

        for center, radius in circles_to_draw:
            cv2.circle(img, center, radius, (255, 255, 255), 2)

        return img

    @pytest.fixture
    def sample_circle_points(self):
        """创建测试圆点集"""
        # 生成一个圆的点集
        center = np.array([100.0, 100.0])
        radius = 50.0
        num_points = 100

        # 生成均匀分布在圆上的点
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = np.zeros((num_points, 2))

        for i, angle in enumerate(angles):
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points[i] = [x, y]

        return points

    def test_detect_circles_hough(self, detector, sample_circle_image):
        """测试霍夫变换圆检测"""
        # 测试标准霍夫变换
        circles = detector.detect_circles(sample_circle_image, method='hough')

        assert circles is not None
        assert isinstance(circles, list)

        # 应该至少检测到一个圆
        if len(circles) > 0:
            circle = circles[0]
            assert 'x' in circle
            assert 'y' in circle
            assert 'radius' in circle
            assert 'confidence' in circle
            assert 'method' in circle
            assert circle['method'] == 'hough'

            # 圆心应该在图像范围内
            assert 0 <= circle['x'] < sample_circle_image.shape[1]
            assert 0 <= circle['y'] < sample_circle_image.shape[0]
            assert 10 <= circle['radius'] <= 200  # 在参数范围内

    def test_detect_circles_hough_improved(self, detector, sample_circle_image):
        """测试改进的霍夫变换圆检测"""
        circles = detector.detect_circles(sample_circle_image, method='hough_improved')

        assert circles is not None
        assert isinstance(circles, list)

        if len(circles) > 0:
            circle = circles[0]
            assert circle['method'] == 'hough_improved'
            assert 'confidence' in circle
            assert 0 <= circle['confidence'] <= 1

    def test_detect_circles_least_squares(self, detector, sample_circle_image):
        """测试最小二乘法圆检测"""
        circles = detector.detect_circles(sample_circle_image, method='lsq')

        assert circles is not None
        assert isinstance(circles, list)

        # 最小二乘法可能检测不到圆，这取决于边缘检测结果
        if len(circles) > 0:
            circle = circles[0]
            assert circle['method'] == 'lsq'
            assert 'fitting_error' in circle
            assert isinstance(circle['fitting_error'], float)

    def test_detect_circles_contour(self, detector, sample_circle_image):
        """测试基于轮廓的圆检测"""
        circles = detector.detect_circles(sample_circle_image, method='contour')

        assert circles is not None
        assert isinstance(circles, list)

        if len(circles) > 0:
            circle = circles[0]
            assert circle['method'] == 'contour'
            assert 'circularity' in circle
            assert 'area' in circle
            assert 0 <= circle['circularity'] <= 1

    def test_detect_circles_target_count(self, detector, sample_multiple_circles_image):
        """测试目标数量限制"""
        # 限制检测数量
        target_count = 2
        circles = detector.detect_circles(
            sample_multiple_circles_image,
            method='hough_improved',
            target_count=target_count
        )

        assert circles is not None
        assert len(circles) <= target_count

    def test_detect_circles_gray_image(self, detector):
        """测试灰度图像输入"""
        # 创建灰度测试图像
        gray_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(gray_img, (100, 100), 50, 255, 2)

        circles = detector.detect_circles(gray_img, method='hough')

        assert circles is not None
        # 可能是空列表，这取决于检测效果

    def test_detect_circles_no_circles(self, detector):
        """测试无圆图像的检测"""
        # 创建不包含圆的图像
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        circles = detector.detect_circles(img, method='hough')

        assert circles is not None
        assert isinstance(circles, list)
        # 可能是空列表

    def test_last_detection_cache(self, detector, sample_circle_image):
        """测试检测结果缓存"""
        # 第一次检测
        circles1 = detector.detect_circles(sample_circle_image)

        # 检查缓存
        assert detector.last_detection is not None
        assert detector.last_detection == circles1

        # 第二次检测
        circles2 = detector.detect_circles(sample_circle_image, method='hough_improved')

        # 缓存应该更新
        assert detector.last_detection == circles2

    def test_update_parameters(self, detector):
        """测试参数更新"""
        original_min_dist = detector.params['hough_min_dist']
        original_min_radius = detector.params['hough_min_radius']

        # 更新参数
        detector.update_parameters(
            hough_min_dist=100,
            hough_min_radius=20
        )

        assert detector.params['hough_min_dist'] == 100
        assert detector.params['hough_min_radius'] == 20
        assert detector.params['hough_min_dist'] != original_min_dist

        # 测试无效参数（应该只记录警告）
        detector.update_parameters(invalid_param=10)

    def test_different_methods_comparison(self, detector, sample_circle_image):
        """测试不同检测方法的比较"""
        methods = ['hough', 'hough_improved', 'lsq', 'contour']

        results = {}
        for method in methods:
            circles = detector.detect_circles(sample_circle_image, method=method)
            results[method] = circles

            assert isinstance(circles, list)

            if len(circles) > 0:
                circle = circles[0]
                assert circle['method'] == method

        # 至少有一种方法应该能检测到圆
        successful_methods = [m for m in methods if len(results[m]) > 0]
        assert len(successful_methods) > 0

    def test_circle_confidence(self, detector):
        """测试置信度计算"""
        # 创建测试图像
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, 255, 2)

        # 调用内部方法（通过公共方法间接测试）
        circles = detector.detect_circles(img, method='hough')

        if len(circles) > 0:
            confidence = circles[0]['confidence']
            assert 0 <= confidence <= 1

    def test_performance_hough(self, detector, sample_circle_image):
        """测试霍夫检测性能"""
        import time

        start_time = time.time()

        runs = 5
        for _ in range(runs):
            detector.detect_circles(sample_circle_image, method='hough')

        end_time = time.time()
        avg_time = (end_time - start_time) / runs

        print(f"标准霍夫检测平均时间: {avg_time:.3f}秒")
        assert avg_time < 2.0  # 应该小于2秒

    def test_performance_hough_improved(self, detector, sample_circle_image):
        """测试改进霍夫检测性能"""
        import time

        start_time = time.time()

        runs = 3
        for _ in range(runs):
            detector.detect_circles(sample_circle_image, method='hough_improved')

        end_time = time.time()
        avg_time = (end_time - start_time) / runs

        print(f"改进霍夫检测平均时间: {avg_time:.3f}秒")
        assert avg_time < 3.0  # 改进方法可能稍慢

    def test_edge_case_small_circle(self, detector):
        """测试小圆检测"""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 5, 255, 1)  # 小圆

        circles = detector.detect_circles(img, method='hough')

        # 可能检测不到，取决于参数
        if len(circles) > 0:
            circle = circles[0]
            assert circle['radius'] >= detector.params['hough_min_radius']

    def test_edge_case_large_circle(self, detector):
        """测试大圆检测"""
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(img, (250, 250), 150, 255, 3)  # 大圆

        circles = detector.detect_circles(img, method='hough')

        if len(circles) > 0:
            circle = circles[0]
            assert circle['radius'] <= detector.params['hough_max_radius']


def test_circle_detector_integration():
    """测试圆检测器的集成功能"""
    detector = CircleDetector()

    # 创建复杂测试图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 画多个圆
    cv2.circle(img, (100, 100), 30, (255, 255, 255), 2)
    cv2.circle(img, (300, 100), 40, (255, 255, 255), 2)
    cv2.circle(img, (200, 300), 50, (255, 255, 255), 2)

    print("\n" + "="*60)
    print("圆检测器集成测试")
    print("="*60)

    # 测试各种方法
    methods = ['hough', 'hough_improved', 'lsq', 'contour']

    for method in methods:
        print(f"\n测试方法: {method}")
        circles = detector.detect_circles(img, method=method)

        print(f"  检测到 {len(circles)} 个圆")
        for i, circle in enumerate(circles[:3]):  # 只显示前3个
            print(f"  圆{i+1}: 中心({circle['x']:.1f}, {circle['y']:.1f}), "
                  f"半径{circle['radius']:.1f}, 置信度{circle['confidence']:.3f}")

    print("\n" + "="*60)
    print("[DONE] 集成测试完成")
    print("="*60)


if __name__ == "__main__":
    """直接运行测试"""
    import numpy as np
    import cv2

    print("=" * 60)
    print("圆检测模块测试")
    print("=" * 60)

    # 创建检测器和测试图像
    detector = CircleDetector()

    print("\n1. 创建测试图像...")
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (255, 255, 255), 2)

    print("\n2. 测试各种检测方法...")
    methods = ['hough', 'hough_improved', 'lsq', 'contour']

    for method in methods:
        print(f"\n  方法: {method}")
        circles = detector.detect_circles(img, method=method)

        if circles and len(circles) > 0:
            print(f"    检测到 {len(circles)} 个圆")
            for i, circle in enumerate(circles[:2]):  # 显示前2个
                print(f"    圆{i+1}: 中心({circle['x']:.1f}, {circle['y']:.1f}), "
                      f"半径{circle['radius']:.1f}")
        else:
            print("    未检测到圆")

    print("\n3. 测试参数更新...")
    detector.update_parameters(hough_min_dist=100, hough_min_radius=20)
    print(f"   新参数 - min_dist: {detector.params['hough_min_dist']}")
    print(f"   新参数 - min_radius: {detector.params['hough_min_radius']}")

    print("\n4. 测试多圆检测...")
    multi_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(multi_img, (100, 100), 30, (255, 255, 255), 2)
    cv2.circle(multi_img, (300, 100), 40, (255, 255, 255), 2)
    cv2.circle(multi_img, (200, 300), 50, (255, 255, 255), 2)

    circles = detector.detect_circles(multi_img, method='hough_improved', target_count=2)
    print(f"   限制检测2个圆，实际检测到 {len(circles)} 个")

    print("\n" + "=" * 60)
    print("[DONE] 圆检测测试完成")
    print("=" * 60)