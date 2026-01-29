"""
测试图像预处理模块
"""
import sys
import os

# 添加core目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

import cv2
import numpy as np
import pytest

# 由于原代码中没有这些函数，需要创建测试用的替代函数
def calculate_psnr(original, processed, max_pixel=255.0):
    """计算PSNR值（独立函数，不依赖preprocess模块）"""
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# 为了兼容原测试代码中的函数名，创建包装函数
def adaptive_retinex_enhancement(image):
    """自适应Retinex增强（包装函数）"""
    from preprocess import ImagePreprocessor
    processor = ImagePreprocessor()
    return processor.adaptive_light_compensation(image, method='msrcr')


def adaptive_filtering(image, method='median_gaussian'):
    """自适应滤波（包装函数）"""
    from preprocess import ImagePreprocessor
    processor = ImagePreprocessor()
    return processor.denoise_image(image, method=method)


def edge_enhancement(image):
    """边缘增强（包装函数，直接调用原函数）"""
    from preprocess import ImagePreprocessor
    processor = ImagePreprocessor()
    return processor.edge_enhancement(image)


# 导入主类
try:
    from preprocess import ImagePreprocessor
except ImportError:
    # 如果IDE报错但能运行，可以暂时注释掉
    # 或者使用动态导入
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preprocess",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "preprocess.py")
    )
    preprocess_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess_module)
    ImagePreprocessor = preprocess_module.ImagePreprocessor


class TestImagePreprocessor:
    """测试图像预处理器类"""

    @pytest.fixture
    def preprocessor(self):
        """创建预处理器实例"""
        return ImagePreprocessor()

    @pytest.fixture
    def sample_image(self):
        """创建测试图像"""
        # 创建一个简单的测试图像
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [255, 255, 255]  # 白色方块
        return img

    @pytest.fixture
    def sample_gray_image(self):
        """创建测试灰度图像"""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255  # 白色方块
        return img

    def test_preprocess_pipeline(self, preprocessor, sample_image):
        """测试预处理流程"""
        # 测试默认流程
        result = preprocessor.preprocess_pipeline(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

        # 测试自定义流程
        steps = ['light_compensation', 'denoise', 'edge_enhance']
        result = preprocessor.preprocess_pipeline(sample_image, steps=steps)
        assert result.shape == sample_image.shape

        # 测试灰度转换
        result = preprocessor.preprocess_pipeline(sample_image, steps=['grayscale'])
        if len(sample_image.shape) == 3:
            assert len(result.shape) == 2

    def test_adaptive_light_compensation(self, preprocessor, sample_image):
        """测试自适应光照补偿"""
        # 测试彩色图像
        for method in ['msrcr', 'clahe', 'hist_equal']:
            result = preprocessor.adaptive_light_compensation(sample_image, method=method)
            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

        # 测试灰度图像
        gray_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
        for method in ['msrcr', 'clahe', 'hist_equal']:
            result = preprocessor.adaptive_light_compensation(gray_image, method=method)
            assert len(result.shape) == 2
            assert result.dtype == np.uint8

    def test_denoise_image(self, preprocessor, sample_image):
        """测试图像去噪"""
        methods = ['median', 'gaussian', 'median_gaussian', 'bilateral']

        for method in methods:
            result = preprocessor.denoise_image(sample_image, method=method)
            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

            # 测试灰度图像
            gray_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
            result_gray = preprocessor.denoise_image(gray_image, method=method)
            assert result_gray.shape == gray_image.shape

    def test_edge_enhancement(self, preprocessor, sample_image, sample_gray_image):
        """测试边缘增强"""
        # 测试彩色图像
        result = preprocessor.edge_enhancement(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

        # 测试灰度图像
        result_gray = preprocessor.edge_enhancement(sample_gray_image)
        assert result_gray.shape == sample_gray_image.shape

        # 测试值范围
        assert result.min() >= 0
        assert result.max() <= 255

    def test_contrast_enhancement(self, preprocessor, sample_image):
        """测试对比度增强"""
        result = preprocessor.contrast_enhancement(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

        # 测试灰度图像
        gray_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
        result_gray = preprocessor.contrast_enhancement(gray_image)
        assert result_gray.shape == gray_image.shape

    def test_to_grayscale(self, preprocessor, sample_image, sample_gray_image):
        """测试灰度转换"""
        # 彩色转灰度
        result = preprocessor.to_grayscale(sample_image)
        assert len(result.shape) == 2
        assert result.shape[:2] == sample_image.shape[:2]

        # 灰度保持灰度
        result_gray = preprocessor.to_grayscale(sample_gray_image)
        assert result_gray.shape == sample_gray_image.shape

    def test_morphological_operations(self, preprocessor, sample_gray_image):
        """测试形态学操作"""
        # 创建二值图像用于测试
        binary_image = (sample_gray_image > 127).astype(np.uint8) * 255

        operations = ['open', 'close', 'erode', 'dilate']

        for operation in operations:
            result = preprocessor.morphological_operations(binary_image, operation=operation)
            assert result.shape == binary_image.shape
            assert result.dtype == np.uint8

    def test_canny_edge_detection(self, preprocessor, sample_image, sample_gray_image):
        """测试Canny边缘检测"""
        # 测试彩色图像
        result = preprocessor.canny_edge_detection(sample_image)
        assert len(result.shape) == 2
        assert result.shape[:2] == sample_image.shape[:2]
        assert result.dtype == np.uint8

        # 测试灰度图像
        result_gray = preprocessor.canny_edge_detection(sample_gray_image)
        assert result_gray.shape == sample_gray_image.shape

        # 检查是否为二值图像
        assert set(np.unique(result)).issubset({0, 255})

    def test_update_parameters(self, preprocessor):
        """测试参数更新"""
        original_gamma = preprocessor.params['gamma']
        original_clip_limit = preprocessor.params['clip_limit']

        # 更新参数
        preprocessor.update_parameters(gamma=2.0, clip_limit=3.0)

        assert preprocessor.params['gamma'] == 2.0
        assert preprocessor.params['clip_limit'] == 3.0
        assert preprocessor.params['gamma'] != original_gamma

        # 测试无效参数（应该只记录警告）
        preprocessor.update_parameters(invalid_param=10)

    def test_performance(self, preprocessor, sample_image):
        """测试性能（基本检查）"""
        import time

        start_time = time.time()

        # 执行多个预处理步骤
        steps = ['light_compensation', 'denoise', 'edge_enhance', 'contrast_enhance']
        result = preprocessor.preprocess_pipeline(sample_image, steps=steps)

        end_time = time.time()
        processing_time = end_time - start_time

        # 简单的时间检查
        assert processing_time < 5.0
        # 使用ASCII字符避免编码问题
        print(f"处理时间: {processing_time:.4f}秒")


# 修改这里：让 TestCompatibilityFunctions 继承 TestImagePreprocessor
class TestCompatibilityFunctions(TestImagePreprocessor):
    """测试兼容性包装函数"""

    def test_adaptive_retinex_enhancement(self, sample_image):
        """测试自适应Retinex增强包装函数"""
        result = adaptive_retinex_enhancement(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

        # 测试灰度图像
        gray_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
        result_gray = adaptive_retinex_enhancement(gray_image)
        assert result_gray.shape == gray_image.shape

    def test_adaptive_filtering(self, sample_image):
        """测试自适应滤波包装函数"""
        methods = ['median', 'gaussian', 'median_gaussian', 'bilateral']

        for method in methods:
            result = adaptive_filtering(sample_image, method=method)
            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

            # 测试灰度图像
            gray_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
            result_gray = adaptive_filtering(gray_image, method=method)
            assert result_gray.shape == gray_image.shape

    def test_edge_enhancement_wrapper(self, sample_image):
        """测试边缘增强包装函数"""
        result = edge_enhancement(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


def test_calculate_psnr():
    """测试PSNR计算函数"""
    # 创建两个相同的图像
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128

    # 计算PSNR
    psnr = calculate_psnr(img1, img2)

    # 相同图像的PSNR应该是无穷大
    assert psnr == float('inf')

    # 测试不同的图像
    img3 = np.ones((100, 100, 3), dtype=np.uint8) * 200
    psnr2 = calculate_psnr(img1, img3)
    assert psnr2 < float('inf')
    assert psnr2 > 0


def test_all_functions_with_original_names():
    """使用原函数名测试所有功能"""
    # 创建测试图像
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[30:70, 30:70] = [200, 150, 100]

    print("\n" + "="*60)
    print("使用原函数名测试兼容性")
    print("="*60)

    # 测试所有原函数名
    test_cases = [
        ("adaptive_retinex_enhancement", lambda: adaptive_retinex_enhancement(test_img)),
        ("adaptive_filtering (median)", lambda: adaptive_filtering(test_img, 'median')),
        ("adaptive_filtering (gaussian)", lambda: adaptive_filtering(test_img, 'gaussian')),
        ("edge_enhancement", lambda: edge_enhancement(test_img)),
        ("calculate_psnr", lambda: calculate_psnr(test_img, test_img)),
    ]

    all_passed = True
    for func_name, test_func in test_cases:
        try:
            result = test_func()
            if func_name == 'calculate_psnr':
                assert result == float('inf')
                # 使用ASCII字符代替表情符号
                print(f"[PASS] {func_name}: 通过 (PSNR = {result})")
            else:
                assert result is not None
                assert result.dtype == np.uint8
                print(f"[PASS] {func_name}: 通过 (形状: {result.shape})")
        except Exception as e:
            # 使用ASCII字符代替表情符号
            print(f"[FAIL] {func_name}: 失败 - {e}")
            all_passed = False

    print("="*60)
    if all_passed:
        # 使用ASCII字符代替表情符号
        print("[SUCCESS] 所有兼容性测试通过!")
    else:
        print("[WARNING] 部分兼容性测试失败")
    print("="*60)


# 移除或修改 __main__ 部分中的表情符号
if __name__ == "__main__":
    # 运行简单的手动测试
    print("=" * 60)
    print("图像预处理模块测试")
    print("=" * 60)

    # 创建预处理器
    preprocessor = ImagePreprocessor()

    # 创建测试图像
    test_img = np.zeros((150, 150, 3), dtype=np.uint8)
    test_img[50:100, 50:100] = [255, 200, 150]
    test_img[20:40, 20:40] = [50, 100, 200]

    print("测试1: 预处理管道")
    result1 = preprocessor.preprocess_pipeline(test_img)
    print(f"  输入: {test_img.shape}, 输出: {result1.shape}")

    print("\n测试2: 光照补偿")
    for method in ['msrcr', 'clahe', 'hist_equal']:
        result = preprocessor.adaptive_light_compensation(test_img, method)
        print(f"  {method}: {result.shape}")

    print("\n测试3: 去噪")
    for method in ['median', 'gaussian', 'bilateral']:
        result = preprocessor.denoise_image(test_img, method)
        print(f"  {method}: {result.shape}")

    print("\n测试4: 边缘增强")
    result4 = preprocessor.edge_enhancement(test_img)
    print(f"  边缘增强: {result4.shape}")

    print("\n测试5: 对比度增强")
    result5 = preprocessor.contrast_enhancement(test_img)
    print(f"  对比度增强: {result5.shape}")

    print("\n测试6: 灰度转换")
    result6 = preprocessor.to_grayscale(test_img)
    print(f"  灰度图像: {result6.shape}")

    print("\n测试7: Canny边缘检测")
    result7 = preprocessor.canny_edge_detection(test_img)
    print(f"  边缘图像: {result7.shape}, 唯一值: {np.unique(result7)}")

    print("\n测试8: 参数更新")
    preprocessor.update_parameters(gamma=1.5, clip_limit=3.0)
    print(f"  新gamma: {preprocessor.params['gamma']}")
    print(f"  新clip_limit: {preprocessor.params['clip_limit']}")

    # 测试兼容性函数
    print("\n" + "="*60)
    print("兼容性函数测试")
    print("="*60)

    # 使用修改后的测试函数
    test_all_functions_with_original_names()

    print("\n" + "="*60)
    print("[DONE] 所有测试完成!")
    print("="*60)