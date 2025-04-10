import sys
import os
from pathlib import Path

# 確保可以導入模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import run_app
from utils.logger import get_logger
from utils.config import Config

# 初始化日誌記錄器
config = Config()
log_dir = config.get("app.log_dir")
logger = get_logger(__name__, log_dir=log_dir)


def main():
    """主程式入口"""
    logger.info("Starting Withholding Tax Classifier application")
    
    try:
        # 運行應用程式
        run_app()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Application closed")


if __name__ == "__main__":
    main()