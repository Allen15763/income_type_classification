import sys
import os

# 在任何其他導入前修補feature_engine的VERSION文件問題
def patch_feature_engine_version():
    """修補feature_engine的VERSION文件問題"""
    feature_engine_dir = None
    
    # 首先嘗試找到feature_engine目錄
    for path in sys.path:
        fe_path = os.path.join(path, 'feature_engine')
        if os.path.exists(fe_path) and os.path.isdir(fe_path):
            feature_engine_dir = fe_path
            break
    
    if feature_engine_dir:
        version_path = os.path.join(feature_engine_dir, 'VERSION')
        
        # 如果VERSION文件不存在，則創建一個
        if not os.path.exists(version_path):
            try:
                with open(version_path, 'w') as f:
                    f.write("1.8.3")  # 使用一個合理的版本號
                print(f"已創建feature_engine VERSION文件: {version_path}")
            except Exception as e:
                print(f"無法創建VERSION文件: {e}")
                # 如果無法創建文件，則使用monkey patch方式
                try:
                    import feature_engine
                    if not hasattr(feature_engine, '__version__'):
                        feature_engine.__version__ = "1.8.3"
                    print("已應用feature_engine版本補丁")
                except ImportError:
                    pass
# 應用修補
patch_feature_engine_version()

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