"""
日志记录工具模块
提供统一的日志记录功能
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from enum import Enum
import functools
import time


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, logger: logging.Logger = None):
        """
        初始化性能日志记录器

        Args:
            logger: 日志记录器，如果为None则创建新的
        """
        self.logger = logger or setup_logger("Performance")
        self.timers: Dict[str, float] = {}

    def start_timer(self, name: str):
        """
        开始计时

        Args:
            name: 计时器名称
        """
        self.timers[name] = time.time()
        self.logger.debug(f"开始计时: {name}")

    def stop_timer(self, name: str) -> float:
        """
        停止计时并返回耗时

        Args:
            name: 计时器名称

        Returns:
            耗时（秒）
        """
        if name not in self.timers:
            self.logger.warning(f"计时器不存在: {name}")
            return 0.0

        elapsed = time.time() - self.timers[name]
        self.logger.info(f"{name} 耗时: {elapsed:.3f}秒")

        del self.timers[name]
        return elapsed

    def log_memory_usage(self):
        """记录内存使用情况"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            self.logger.info(
                f"内存使用: RSS={memory_info.rss / 1024 / 1024:.1f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.1f}MB"
            )
        except ImportError:
            self.logger.warning("psutil未安装，无法记录内存使用")
        except Exception as e:
            self.logger.error(f"记录内存使用失败: {str(e)}")

    def log_system_info(self):
        """记录系统信息"""
        try:
            import platform

            self.logger.info(f"系统平台: {platform.platform()}")
            self.logger.info(f"Python版本: {platform.python_version()}")
            self.logger.info(f"处理器: {platform.processor()}")

        except Exception as e:
            self.logger.error(f"记录系统信息失败: {str(e)}")


def setup_logger(name: str,
                 level: LogLevel = LogLevel.INFO,
                 log_dir: str = "logs",
                 console_output: bool = True,
                 file_output: bool = True) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level.value)

    # 清除已有的处理器
    logger.handlers.clear()

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if file_output:
        try:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)

            # 创建日志文件名（按日期）
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(log_dir, f"{date_str}.log")

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level.value)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            print(f"创建文件日志处理器失败: {str(e)}")

    return logger


def setup_file_logger(name: str,
                      log_file: str,
                      level: LogLevel = LogLevel.INFO) -> logging.Logger:
    """
    设置文件日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.value)

    # 清除已有的处理器
    logger.handlers.clear()

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        # 确保目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    except Exception as e:
        print(f"创建文件日志处理器失败: {str(e)}")

    return logger


def setup_console_logger(name: str,
                         level: LogLevel = LogLevel.INFO) -> logging.Logger:
    """
    设置控制台日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.value)

    # 清除已有的处理器
    logger.handlers.clear()

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level.value)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器（如果不存在则创建）

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    else:
        return setup_logger(name)


def log_function_call(logger: logging.Logger = None):
    """
    函数调用日志装饰器

    Args:
        logger: 日志记录器，如果为None则使用函数名创建

    Returns:
        装饰器函数
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取日志记录器
            func_logger = logger or get_logger(func.__module__)

            # 记录函数调用开始
            func_logger.debug(f"调用函数: {func.__name__}")

            try:
                # 执行函数
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                # 记录函数调用成功
                func_logger.debug(
                    f"函数 {func.__name__} 执行成功，耗时: {elapsed_time:.3f}秒"
                )

                return result

            except Exception as e:
                # 记录函数调用失败
                func_logger.error(
                    f"函数 {func.__name__} 执行失败: {str(e)}",
                    exc_info=True
                )
                raise

        return wrapper

    return decorator


def log_exception(logger: logging.Logger,
                  exception: Exception,
                  context: str = "",
                  level: LogLevel = LogLevel.ERROR):
    """
    记录异常信息

    Args:
        logger: 日志记录器
        exception: 异常对象
        context: 上下文信息
        level: 日志级别
    """
    try:
        # 构建错误消息
        if context:
            message = f"{context}: {str(exception)}"
        else:
            message = str(exception)

        # 获取完整的异常跟踪信息
        exc_info = (type(exception), exception, exception.__traceback__)

        # 记录日志
        logger.log(level.value, message, exc_info=exc_info)

    except Exception as e:
        # 如果日志记录失败，至少打印到控制台
        print(f"日志记录失败: {str(e)}")
        print(f"原始异常: {str(exception)}")
        traceback.print_exc()


def log_performance(func: Callable):
    """
    性能日志装饰器

    Args:
        func: 被装饰的函数

    Returns:
        装饰后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("Performance")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            logger.info(
                f"函数 {func.__name__} 执行耗时: {elapsed_time:.3f}秒"
            )

            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"函数 {func.__name__} 执行失败，耗时: {elapsed_time:.3f}秒，错误: {str(e)}"
            )
            raise

    return wrapper


def create_rotating_file_handler(log_dir: str = "logs",
                                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                                 backup_count: int = 5) -> logging.FileHandler:
    """
    创建滚动文件处理器

    Args:
        log_dir: 日志目录
        max_bytes: 单个文件最大大小
        backup_count: 备份文件数量

    Returns:
        文件处理器
    """
    try:
        os.makedirs(log_dir, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        log_file = os.path.join(log_dir, "application.log")

        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handler.setFormatter(formatter)
        return handler

    except Exception as e:
        print(f"创建滚动文件处理器失败: {str(e)}")
        raise


def setup_detailed_logger(name: str,
                          log_dir: str = "logs",
                          level: LogLevel = LogLevel.DEBUG) -> logging.Logger:
    """
    设置详细日志记录器（包含调试信息）

    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.value)

    # 清除已有的处理器
    logger.handlers.clear()

    # 详细格式（包含文件名和行号）
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level.value)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # 文件处理器（滚动）
    try:
        file_handler = create_rotating_file_handler(log_dir)
        file_handler.setLevel(level.value)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"创建文件处理器失败: {str(e)}")

    return logger


# 创建默认的应用程序日志记录器
app_logger = setup_logger("ConcentricityDetection")

if __name__ == "__main__":
    # 测试代码
    logger = setup_logger("Test")

    logger.debug("调试信息")
    logger.info("一般信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.critical("严重错误")

    # 测试性能日志
    perf_logger = PerformanceLogger(logger)
    perf_logger.start_timer("测试任务")
    time.sleep(0.1)
    perf_logger.stop_timer("测试任务")


    # 测试函数调用日志
    @log_function_call(logger)
    def test_function(x, y):
        return x + y


    result = test_function(3, 4)
    print(f"测试函数结果: {result}")