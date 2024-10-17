import sys
from pathlib import Path


def get_logger(log_dir: Path, stdout: bool = True):
    from loguru import logger

    logger.remove()  # デフォルトの設定を削除
    custom_format = "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level} ] {message}</level>"
    logger.add(
        log_dir / "log_{time:YYYY-MM-DD-HH-mm-ss}.txt",
        level="INFO",
        colorize=False,
        format=custom_format,
    )
    if stdout:
        custom_format = "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level} ] {message}</level>"
        logger.add(sys.stdout, level="INFO", colorize=True, format=custom_format)
    return logger
