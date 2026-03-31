import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


class IgnoreLogChangeDetectedFilter(logging.Filter):
    """自定义日志过滤器，用于忽略包含"Detected file change in"的日志记录。"""
    def filter(self, record: logging.LogRecord):
        # 过滤掉包含文件变更检测信息的日志，避免热重载时产生大量噪音日志
        return "Detected file change in" not in record.getMessage()


def setup_logging(format: str = None):
    """
    为应用程序配置日志，支持日志轮转。

    环境变量说明:
        LOG_LEVEL: 日志级别 (默认: INFO)
        LOG_FILE_PATH: 日志文件路径 (默认: logs/application.log)
        LOG_MAX_SIZE: 轮转前的最大文件大小，单位 MB (默认: 10MB)
        LOG_BACKUP_COUNT: 保留的备份文件数量 (默认: 5)

    确保日志目录存在，防止路径遍历攻击，并同时配置
    轮转文件处理器和控制台处理器。
    """
    # 确定日志目录和默认日志文件路径
    base_dir = Path(__file__).parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)  # 若目录不存在则创建
    default_log_file = log_dir / "application.log"

    # 从环境变量获取日志级别
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # 从环境变量获取日志文件路径
    log_file_path = Path(os.environ.get("LOG_FILE_PATH", str(default_log_file)))

    # 安全路径检查：日志文件必须位于 logs/ 目录内，防止路径遍历
    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(f"LOG_FILE_PATH '{log_file_path}' 位于受信任日志目录 '{log_dir_resolved}' 之外")

    # 确保父目录存在
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取最大日志文件大小（默认：10MB）
    try:
        max_mb = int(os.environ.get("LOG_MAX_SIZE", 10))  # 默认 10MB
        max_bytes = max_mb * 1024 * 1024
    except (TypeError, ValueError):
        max_bytes = 10 * 1024 * 1024  # 解析失败时回退到 10MB

    # 获取备份文件数量（默认：5）
    try:
        backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 5))
    except ValueError:
        backup_count = 5

    # 配置日志格式
    log_format = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"

    # 创建文件处理器（带轮转功能）和控制台处理器
    file_handler = RotatingFileHandler(resolved_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    console_handler = logging.StreamHandler()

    # 为两个处理器设置相同的日志格式
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加过滤器，抑制"Detected file change"等热重载噪音日志
    file_handler.addFilter(IgnoreLogChangeDetectedFilter())
    console_handler.addFilter(IgnoreLogChangeDetectedFilter())

    # 应用日志配置（force=True 确保覆盖已有配置）
    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler], force=True)

    # 输出日志配置信息（仅在 DEBUG 级别下显示）
    logger = logging.getLogger(__name__)
    logger.debug(
        f"日志已配置: level={log_level_str}, "
        f"file={resolved_path}, max_size={max_bytes} bytes, "
        f"backup_count={backup_count}"
    )
