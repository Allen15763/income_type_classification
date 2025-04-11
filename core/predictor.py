import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, List, Optional

from core.data_processor import DataProcessor
from core.feature_engineering import FeatureEngineer
from core.model_manager import ModelManager
from utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    預測器類別，整合資料處理、特徵工程和模型預測功能
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 encoder_path: Optional[str] = None,
                 pipeline_path: Optional[str] = None):
        """
        初始化預測器
        
        Parameters:
        -----------
        model_path : str, optional
            模型檔案路徑
        encoder_path : str, optional
            標籤編碼器檔案路徑
        pipeline_path : str, optional
            特徵工程管道檔案路徑
        """
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager()
        
        # 如果提供了模型路徑，就載入模型
        if all([model_path, encoder_path]):
            self.load_model(model_path, encoder_path, pipeline_path)
    
    def load_model(self, model_path: str, encoder_path: str, pipeline_path: Optional[str] = None) -> None:
        """載入預訓練模型和組件"""
        logger.info("Loading model and components...")
        self.model_manager.load_model(model_path, encoder_path, pipeline_path)
        
        # 如果有特徵工程管道，也要設置
        if pipeline_path is not None and self.model_manager.feature_pipeline is not None:
            self.feature_engineer.feature_pipeline = self.model_manager.feature_pipeline
            logger.info("Feature pipeline loaded")
    
    def predict_from_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        從檔案讀取數據並進行預測
        
        Parameters:
        -----------
        file_path : str or Path
            輸入檔案路徑
            
        Returns:
        --------
        pd.DataFrame
            包含原始數據和預測結果的DataFrame
        """
        logger.info(f"Predicting from file: {file_path}")
        
        # 載入並處理數據
        raw_data = self.data_processor.load_data(file_path)
        cleaned_data = self.data_processor.clean_data()
        # features_data = self.data_processor.extract_features()
        
        # 預處理並特徵工程
        processed_data = self.feature_engineer.preprocess_data(cleaned_data)
        
        # 如果沒有特徵管道，創建一個
        if self.feature_engineer.feature_pipeline is None:
            logger.warning("Feature pipeline not found. Creating a new one.")
            self.feature_engineer.create_feature_pipeline()
            self.feature_engineer.fit_transform(processed_data)
            self.model_manager.set_feature_pipeline(self.feature_engineer.feature_pipeline)
        
        # 轉換特徵
        X = self.feature_engineer.transform(processed_data)
        
        # 預測
        predictions = self.model_manager.predict(X)
        
        # 將預測結果加入到原始數據
        result_df = processed_data.copy()
        result_df['predicted_income_type'] = predictions
        
        logger.info(f"Prediction completed. {len(predictions)} records processed.")
        return result_df
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        對DataFrame進行批量預測
        
        Parameters:
        -----------
        data : pd.DataFrame
            輸入DataFrame
            
        Returns:
        --------
        pd.DataFrame
            包含原始數據和預測結果的DataFrame
        """
        logger.info("Processing batch prediction...")
        
        # 預處理數據
        processed_data = self.feature_engineer.preprocess_data(data)
        
        # 轉換特徵
        X = self.feature_engineer.transform(processed_data)
        
        # 預測
        predictions = self.model_manager.predict(X)
        
        # 將預測結果加入到原始數據
        result_df = processed_data.copy()
        result_df['predicted_income_type'] = predictions
        
        logger.info(f"Batch prediction completed. {len(predictions)} records processed.")
        return result_df
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_path: Union[str, Path], 
                         format: str = 'excel') -> str:
        """
        保存預測結果到檔案
        
        Parameters:
        -----------
        predictions_df : pd.DataFrame
            包含預測結果的DataFrame
        output_path : str or Path
            輸出檔案路徑
        format : str, optional
            輸出檔案格式 ('excel', 'csv', 'parquet')
            
        Returns:
        --------
        str
            輸出檔案的絕對路徑
        """
        output_path = Path(output_path)
        logger.info(f"Saving predictions to {output_path}")
        
        # 創建包含目錄
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'excel':
            predictions_df.to_excel(output_path, index=False, engine='openpyxl')
        elif format.lower() == 'csv':
            predictions_df.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            predictions_df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'excel', 'csv', or 'parquet'.")
        
        logger.info(f"Predictions saved to {output_path}")
        return str(output_path.absolute())