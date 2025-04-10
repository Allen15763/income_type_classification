import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class Config:
    """
    配置類別，負責管理應用配置
    實現了單例模式，確保整個應用只有一個配置實例
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 預設配置
        self.config = {
            "model": {
                "model_path": None,
                "encoder_path": None,
                "pipeline_path": None,
                "default_model_dir": str(Path.home() / "WithholdingTaxClassifier" / "models"),
            },
            "data": {
                "default_input_dir": str(Path.home() / "WithholdingTaxClassifier" / "input"),
                "default_output_dir": str(Path.home() / "WithholdingTaxClassifier" / "output"),
                "required_columns": [
                    "Payment Date", "Remit-to Supplier", "Supplier ID", 
                    "Line Desc", "Line Amount", "W/H Tax", "W/H Tax Name", 
                    "Income Type", "Income Type Name"
                ]
            },
            "ui": {
                "theme": "light",
                "language": "zh-TW",
                "window_size": [1000, 800]
            },
            "app": {
                "log_dir": str(Path.home() / "WithholdingTaxClassifier" / "logs"),
                "debug": False
            }
        }
        
        # 創建配置文件目錄
        self.config_dir = Path.home() / "WithholdingTaxClassifier" / "config"
        self.config_file = self.config_dir / "config.json"
        
        # 載入配置文件
        self.load_config()
        
        # 創建必要的目錄
        self._create_directories()
        
        self._initialized = True
    
    def _create_directories(self):
        """創建應用所需的目錄結構"""
        for dir_path in [
            self.config["model"]["default_model_dir"],
            self.config["data"]["default_input_dir"],
            self.config["data"]["default_output_dir"],
            self.config["app"]["log_dir"]
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """從配置文件載入配置"""
        # 創建配置目錄（如果不存在）
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果配置文件存在，則載入
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            # 如果不存在，則寫入預設配置
            self.save_config()
    
    def save_config(self):
        """保存配置到配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """從配置中獲取值"""
        # 支持用點號分隔的嵌套路徑，如 "model.model_path"
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """設置配置值"""
        # 支持用點號分隔的嵌套路徑，如 "model.model_path"
        keys = key.split('.')
        
        # 尋找最後一個鍵之前的字典
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 設置值
        config[keys[-1]] = value
        
        # 儲存更新後的配置
        self.save_config()
        logger.info(f"Configuration updated: {key} = {value}")