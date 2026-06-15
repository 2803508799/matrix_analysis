# -*- coding: utf-8 -*-
"""
可复用的日志配置模块
- 路径只能设置一次，防止意外修改
- 提供全局 logger，支持直接导入使用
"""

import logging
import os
from datetime import datetime


class _logManager:
    """内部日志管理器（单例式配置）"""
    _initialized = False
    _log_dir = None
    _logger = None

    @classmethod
    def setup(cls, log_dir: str = "logs", log_level: int = logging.DEBUG):
        """
        初始化日志系统
        Args:
            log_dir: 日志文件存放目录
            log_level: 最低记录级别
        Returns:
            logging.Logger
        Raises:
            RuntimeError: 如果已经初始化过，路径不可更改
        """
        if cls._initialized:
            raise RuntimeError(
                f"日志路径已固定为 '{cls._log_dir}'，不允许再次设置"
            )
        cls._log_dir = log_dir
        cls._initialized = True

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)

        # 日志文件名按日期生成
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"app_{today}.log")

        logger = logging.getLogger("MyAppLogger")
        logger.setLevel(log_level)

        # 避免多次添加 Handler（重复调用 setup 前已拦截）
        if logger.hasHandlers():
            logger.handlers.clear()

        # --- 控制台 Handler（简洁信息，只展示 INFO 以上）---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # --- 文件 Handler（详细记录，包含文件名和行号）---
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        cls._logger = logger
        return logger

    @classmethod
    def get_logger(cls):
        """获取全局 logger，若未初始化则使用默认路径初始化"""
        if not cls._initialized:
            cls.setup()  # 使用默认路径 logs，同时锁定路径
        return cls._logger


def setup_log_path(log_dir: str = "logs", log_level: int = logging.DEBUG):
    """
    自定义日志路径（必须在任何模块使用 logger 之前调用）
    路径一旦设置，不可更改；若未调用本函数，首次使用时自动采用默认路径
    """
    return _logManager.setup(log_dir, log_level)


# -----------------------------------------------------------------
# 模块级「伪变量」: 其他模块直接 import logger 即可使用
# 通过 __getattr__ 实现惰性加载，确保只在第一次访问时初始化
# -----------------------------------------------------------------
def __getattr__(name):
    if name == "logger":
        return _logManager.get_logger()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")