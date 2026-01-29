"""
图像采集模块 - 专门用于手机图像采集
"""
import cv2
import numpy as np
import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class PhoneImageLoader:
    """手机图像加载器类 - 优化版"""

    def __init__(self):
        self.current_image = None
        self.image_info = {}
        self.camera_capture = None
        self.is_streaming = False
        self.frame_buffer = deque(maxlen=3)  # 只保留最新的3帧
        self.latest_frame = None
        self.capture_thread = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_capture_time = 0
        self.fps = 10  # 目标帧率

    def connect_phone_camera(self, camera_url: str):
        """
        连接手机摄像头（通过IP摄像头应用）- 优化版本
        """
        try:
            # 尝试不同的后端
            backends = [
                cv2.CAP_FFMPEG,
                cv2.CAP_ANY
            ]

            for backend in backends:
                self.camera_capture = cv2.VideoCapture(camera_url, backend)
                if self.camera_capture.isOpened():
                    logger.info(f"使用后端 {backend} 连接成功: {camera_url}")
                    break

            if not self.camera_capture.isOpened():
                # 最后尝试默认方式
                self.camera_capture = cv2.VideoCapture(camera_url)

            if not self.camera_capture.isOpened():
                logger.error(f"无法连接手机摄像头: {camera_url}")
                return False

            # 设置摄像头参数以减少延迟
            self.camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
            self.camera_capture.set(cv2.CAP_PROP_FPS, 15)  # 设置帧率
            self.camera_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 使用MJPEG编码

            # 获取实际分辨率，如果太大则降低
            width = int(self.camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 如果分辨率过高，则降低
            if width > 1280 or height > 720:
                new_width = min(width, 1280)
                new_height = int(height * (new_width / width))
                self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
                self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
                logger.info(f"分辨率从 {width}x{height} 调整为 {new_width}x{new_height}")

            # 启动独立线程捕获帧
            self.is_streaming = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()

            logger.info(f"手机摄像头连接成功: {camera_url}, 分辨率: {width}x{height}")
            return True

        except Exception as e:
            logger.error(f"连接手机摄像头失败: {e}")
            return False

    def _capture_frames(self):
        """独立的帧捕获线程"""
        empty_frame_count = 0
        max_empty_frames = 10

        while self.is_streaming and self.camera_capture is not None:
            try:
                # 控制帧率
                current_time = time.time()
                if current_time - self.last_capture_time < (1.0 / self.fps):
                    time.sleep(0.01)  # 短暂休眠
                    continue

                # 清空缓冲区，获取最新帧
                self.camera_capture.grab()

                ret, frame = self.camera_capture.retrieve()
                if ret and frame is not None and frame.size > 0:
                    empty_frame_count = 0  # 重置空帧计数

                    # 转换为RGB格式
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 降低分辨率以提高性能（如果需要）
                    height, width = frame_rgb.shape[:2]
                    if height > 480:  # 降低到480p显示
                        scale = 480 / height
                        new_width = int(width * scale)
                        frame_rgb = cv2.resize(frame_rgb, (new_width, 480))

                    with self.lock:
                        self.latest_frame = frame_rgb
                        self.frame_count += 1

                    self.last_capture_time = current_time
                else:
                    empty_frame_count += 1
                    if empty_frame_count >= max_empty_frames:
                        logger.warning(f"连续收到 {empty_frame_count} 个空帧")
                        time.sleep(0.1)

            except Exception as e:
                logger.error(f"捕获帧时出错: {e}")
                time.sleep(0.1)

    def capture_from_phone(self):
        """
        从手机摄像头获取最新帧 - 优化版本
        """
        with self.lock:
            return self.latest_frame

    def get_frame_rate(self):
        """获取当前帧率"""
        if self.camera_capture is not None:
            fps = self.camera_capture.get(cv2.CAP_PROP_FPS)
            return fps
        return 0

    def disconnect_phone_camera(self):
        """断开手机摄像头连接"""
        self.is_streaming = False

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None

        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None

        with self.lock:
            self.latest_frame = None
            self.frame_count = 0

        logger.info("手机摄像头已断开连接")

    # ... 其他方法保持不变 ...

    def get_current_image(self):
        """获取当前图像"""
        return self.current_image

    def get_image_info(self):
        """获取当前图像信息"""
        return self.image_info

    def save_image(self, save_path: str):
        """
        保存当前图像

        Args:
            save_path: 保存路径

        Returns:
            bool: 保存是否成功
        """
        if self.current_image is None:
            logger.error("没有图像可保存")
            return False

        try:
            # 转换为BGR格式保存
            image_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            logger.info(f"图像保存成功: {save_path}")
            return True

        except Exception as e:
            logger.error(f"保存图像失败: {e}")
            return False


# camera.py 中的 ImageController 类修改

class ImageController:
    """图像控制器类 - 替代原来的CameraController"""

    def __init__(self):
        self.image_loader = PhoneImageLoader()
        self.is_connected = False  # 手机摄像头连接状态

    def is_opened(self):
        """检查是否已连接（兼容性方法）"""
        return self.is_connected

    def open(self, source_type="file", url=None):
        """
        打开图像源

        Args:
            source_type: 图像源类型，可以是 "file" 或 "camera"
            url: 手机摄像头URL（可选，仅在source_type="camera"时使用）

        Returns:
            bool: 打开是否成功
        """
        if source_type == "file":
            # 文件类型总是成功
            self.is_connected = True
            return True
        elif source_type == "camera":
            # 尝试连接手机摄像头
            # 如果提供了url，使用提供的url，否则使用默认值
            if url is None:
                url = "http://192.168.1.100:8080/video"
            success = self.image_loader.connect_phone_camera(url)
            self.is_connected = success
            return success
        return False

    def close(self):
        """关闭连接"""
        self.image_loader.disconnect_phone_camera()
        self.is_connected = False

    def get_frame(self):
        """获取帧（兼容性方法）"""
        return self.image_loader.get_current_image()

    def get_camera_info(self):
        """获取信息（兼容性方法）"""
        return self.image_loader.get_image_info()

    def load_image_file(self, image_path):
        """加载图像文件"""
        return self.image_loader.load_phone_image(image_path)

    def capture_frame(self):
        """从手机摄像头捕获帧"""
        return self.image_loader.capture_from_phone()