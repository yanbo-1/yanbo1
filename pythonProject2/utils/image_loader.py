# pythonProject2/utils/image_loader.py
"""
图像加载工具模块，解决OpenCV中文路径问题
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def load_image_chinese(path: str, flags: int = cv2.IMREAD_COLOR):
    """
    支持中文路径的图像加载函数

    参数:
        path: 图像文件路径（绝对或相对路径）
        flags: OpenCV读取标志

    返回:
        成功: 返回图像数据 (numpy array)
        失败: 返回None
    """
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        return None

    try:
        # 核心解决方法：二进制读取 + cv2解码
        with open(path, 'rb') as f:
            img_bytes = np.frombuffer(f.read(), np.uint8)

        img = cv2.imdecode(img_bytes, flags)

        if img is None:
            logger.error(f"无法解码图像文件: {path}")
            return None

        logger.info(f"加载图像成功: {os.path.basename(path)} (尺寸: {img.shape})")
        return img

    except Exception as e:
        logger.error(f"读取图像失败 '{path}': {e}")
        return None


def save_image_chinese(path: str, img, params=None):
    """
    支持中文路径的图像保存函数
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 获取文件扩展名
        ext = os.path.splitext(path)[1]

        # 使用imencode编码
        success, encoded = cv2.imencode(ext, img, params)

        if success:
            with open(path, 'wb') as f:
                f.write(encoded.tobytes())
            return True
        return False
    except Exception as e:
        logger.error(f"保存图像失败: {e}")
        return False


# 测试函数
def test():
    """测试函数"""
    test_path = "data/images/屏幕截图 2026-01-15 202658.png"
    print(f"测试中文路径: {test_path}")

    img = load_image_chinese(test_path)
    if img is not None:
        print(f"✓ 测试成功！图像尺寸: {img.shape}")
        return True
    else:
        print("✗ 测试失败")
        return False


if __name__ == "__main__":
    test()