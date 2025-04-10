import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Set, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    數據驗證類，用於檢查輸入數據的有效性
    """
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Tuple[bool, str]:
        """
        驗證文件路徑
        
        Parameters:
        -----------
        file_path : str or Path
            要驗證的文件路徑
        must_exist : bool
            如果為True，文件必須存在
            
        Returns:
        --------
        Tuple[bool, str]
            (是否有效, 錯誤訊息)
        """
        file_path = Path(file_path)
        
        # 檢查路徑是否為空
        if not file_path:
            return False, "File path is empty"
        
        # 檢查文件是否存在
        if must_exist and not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        # 檢查文件擴展名
        if file_path.suffix.lower() not in ['.xlsx', '.xls', '.csv', '.parquet']:
            return False, f"Unsupported file format: {file_path.suffix}"
        
        return True, ""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
        """
        驗證DataFrame是否包含所需的列
        
        Parameters:
        -----------
        df : pd.DataFrame
            要驗證的DataFrame
        required_columns : List[str]
            所需的列列表
            
        Returns:
        --------
        Tuple[bool, str]
            (是否有效, 錯誤訊息)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"
        
        # 檢查所需列是否缺失
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # 檢查是否有完全沒有數據的空列
        empty_columns = [col for col in required_columns if df[col].isna().all()]
        if empty_columns:
            logger.warning(f"Columns with all missing values: {', '.join(empty_columns)}")
        
        return True, ""
    
    @staticmethod
    def validate_model_files(model_path: Union[str, Path], encoder_path: Union[str, Path], 
                             pipeline_path: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
        """
        驗證模型文件
        
        Parameters:
        -----------
        model_path : str or Path
            模型文件路徑
        encoder_path : str or Path
            編碼器文件路徑
        pipeline_path : str or Path, optional
            特徵工程管道文件路徑
            
        Returns:
        --------
        Tuple[bool, str]
            (是否有效, 錯誤訊息)
        """
        model_path = Path(model_path)
        encoder_path = Path(encoder_path)
        
        # 檢查模型文件
        if not model_path.exists():
            return False, f"Model file does not exist: {model_path}"
        
        # 檢查編碼器文件
        if not encoder_path.exists():
            return False, f"Encoder file does not exist: {encoder_path}"
        
        # 檢查管道文件（如果提供）
        if pipeline_path is not None:
            pipeline_path = Path(pipeline_path)
            if not pipeline_path.exists():
                return False, f"Pipeline file does not exist: {pipeline_path}"
        
        return True, ""