# test_fix.py - 放在 pythonProject2/ 目录下
import sys
import os

# 添加项目路径
sys.path.append('.')

print("测试中文路径修复...")
print("=" * 60)

# 测试1：检查文件是否存在
test_path = "data/images/屏幕截图 2026-01-15 202658.png"
print(f"测试文件: {test_path}")
print(f"文件是否存在: {os.path.exists(test_path)}")

if os.path.exists(test_path):
    # 测试2：测试新的加载函数
    try:
        from utils.image_loader import load_image_chinese

        img = load_image_chinese(test_path)
        if img is not None:
            print(f"✓ 图像加载成功！尺寸: {img.shape}")
        else:
            print("✗ 图像加载返回None")
    except ImportError as e:
        print(f"✗ 无法导入 image_loader: {e}")

    # 测试3：测试旧的cv2.imread（应该失败）
    print("\n对比测试：直接使用cv2.imread")
    import cv2

    img_old = cv2.imread(test_path)
    if img_old is not None:
        print(f"✗ cv2.imread 意外成功！这不应该发生")
    else:
        print(f"✓ cv2.imread 失败（符合预期）")
else:
    print("⚠ 测试文件不存在，请检查路径")

print("\n" + "=" * 60)
print("如果第一个测试成功，第二个测试失败，说明修复成功！")