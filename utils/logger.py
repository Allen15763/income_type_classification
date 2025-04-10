import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name, log_level=logging.INFO, log_dir=None):
    """
    配置並返回日誌記錄器
    
    Parameters:
    -----------
    name : str
        日誌記錄器名稱
    log_level : int, optional
        日誌記錄級別
    log_dir : str, optional
        日誌文件目錄
        
    Returns:
    --------
    logging.Logger
        配置好的日誌記錄器
    """
    # 創建日誌記錄器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 如果已經有處理程序，則不再添加
    if logger.handlers:
        return logger
    
    # 創建格式化程序
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台處理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果提供了日誌目錄，添加文件處理程序
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger