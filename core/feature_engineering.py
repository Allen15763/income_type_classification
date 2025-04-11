# locate path for unit test
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.transformation import LogCpTransformer
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    負責特徵工程的類別，遵循單一職責原則
    主要功能：
    1. 特徵轉換和創建
    2. 特徵選擇
    3. 處理分類特徵
    """
    
    def __init__(self):
        self.feature_pipeline = None
        self.column_transformer = None
        self.preprocessed_data = None
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """資料預處理，進行基本轉換並返回處理後的DataFrame"""
        logger.info("Preprocessing data...")
        
        from core.data_processor import DataProcessor
        
        # 創建副本以避免修改原始數據
        processed_df = df.copy()
        
        # 處理供應商ID中的控制字符
        if 'Supplier ID' in processed_df.columns:
            processed_df['Supplier ID'] = DataProcessor.remove_control_chars_from_series(processed_df['Supplier ID'])
            
        # 分類供應商類型
        processed_df['supplier_type'] = processed_df.apply(DataProcessor.classify_supplier, axis=1)
        
        # 重命名列，與模型訓練時的格式一致
        column_mapping = {
            'Payment Date': 'payment_date',
            'Remit-to Supplier': 'supplier',
            'Supplier ID': 'supplier_id',
            'Line Amount': 'amount',
            'Income Type': 'income_type',
            'Line Desc': 'line_desc',
            'Income Type Name': 'income_type_name'
        }
        
        # 僅重命名存在的列
        existing_columns = {k: v for k, v in column_mapping.items() if k in processed_df.columns}
        processed_df = processed_df.rename(columns=existing_columns)
        
        # 確保所有必要的列都存在
        for new_col in column_mapping.values():
            if new_col not in processed_df.columns:
                processed_df[new_col] = np.nan
                
        # 轉換數據類型
        if 'payment_date' in processed_df.columns:
            processed_df['payment_date'] = pd.to_datetime(processed_df['payment_date'], errors='coerce')
            processed_df['year_month'] = processed_df['payment_date'].dt.to_period('M')
                
        if 'amount' in processed_df.columns:
            processed_df['amount'] = pd.to_numeric(processed_df['amount'], errors='coerce')
                
        # 清理描述欄位
        if 'line_desc' in processed_df.columns:
            processed_df['cleaned_line_desc'] = DataProcessor.clean_desc_optimized(
                processed_df['line_desc'].fillna('no description')
            )
            # 添加額外檢查，確保沒有空文本
            empty_mask = processed_df['cleaned_line_desc'].apply(
                lambda x: x is None or str(x).strip() == ''
            )
            processed_df.loc[empty_mask, 'cleaned_line_desc'] = 'unknown_item'
                
        # 填補缺失的income_type
        if 'income_type' in processed_df.columns:
            processed_df['income_type'] = processed_df['income_type'].fillna('免扣')
                
        if 'income_type_name' in processed_df.columns:
            processed_df['income_type_name'] = processed_df['income_type_name'].fillna('免扣')

        # 保留貸方項目 amount = constant + abs(amount)
        netting_df = DataProcessor.group_by_supplier(processed_df)
        netting_df = DataProcessor.create_mapping_key(netting_df)
        processed_df = DataProcessor.create_mapping_key(processed_df)
        processed_df = DataProcessor.copy_amount_for_log(processed_df)
        
        # 篩選出金額大於0的資料
        if 'amount' in processed_df.columns:
            processed_df = processed_df.loc[processed_df['amount'] > 0, :].reset_index(drop=True)
            
        logger.info(f"Data preprocessing complete. Shape: {processed_df.shape}")
        
        self.preprocessed_data = processed_df
        return processed_df
    
    def create_feature_pipeline(self) -> Pipeline:
        """創建特徵工程管道"""
        logger.info("Creating feature pipeline...")
        
        # 特徵分類
        categorical_low_card = ['supplier_type', 'income_type']
        categorical_high_card = ['year_month']
        numerical_features = ['amount']
        text_features = ['cleaned_line_desc']
        
        # 創建特徵處理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_low', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_low_card),
                
                ('cat_high', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=42000))
                ]), categorical_high_card),
                
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('log', LogCpTransformer()),
                ]), numerical_features),
                
                ('text', TfidfVectorizer(max_features=200), text_features[0])
            ],
            remainder='drop'
        )
        
        # 完整特徵工程管道
        feature_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False))
        ])
        
        self.feature_pipeline = feature_pipeline
        self.column_transformer = preprocessor
        
        logger.info("Feature pipeline created")
        return feature_pipeline
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """擬合並轉換數據"""
        if self.feature_pipeline is None:
            self.create_feature_pipeline()
        
        logger.info("Fitting and transforming data...")
        processed_df = df if self.preprocessed_data is None else self.preprocessed_data
        features = self.feature_pipeline.fit_transform(processed_df)
        logger.info(f"Data transformed. Shape: {features.shape}")
        
        return features
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """轉換數據"""
        if self.feature_pipeline is None:
            raise ValueError("Feature pipeline not initialized. Call create_feature_pipeline() first.")
        
        logger.info("Transforming data...")
        processed_df = df if self.preprocessed_data is None else self.preprocessed_data
        features = self.feature_pipeline.transform(processed_df)
        logger.info(f"Data transformed. Shape: {features.shape}")
        
        return features
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        """獲取預處理後的數據"""
        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data available. Call preprocess_data() first.")
        
        return self.preprocessed_data
    

if __name__ == "__main__":
    # 當此模組被直接執行時，運行自測程式碼
    import pandas as pd
    from pathlib import Path
    from core.data_processor import DataProcessor
    
    # 測試資料
    test_file = Path("C:/SEA/AP Tax Check/withholding_tax_classifier/tests/SEA_Open_Bill_Report_240325.xlsx")
    
    # 初始化資料處理器和特徵工程
    dp = DataProcessor()
    fe = FeatureEngineer()
    from core.model_manager import ModelManager
    mm = ModelManager()
    
    # 載入模型和標籤編碼器
    mm.load_model("./resources/models/xgb_income_type_classifier_20250408_174410.pkl", 
                  encoder_path="./resources/models/xgb_income_type_classifier_encoder_20250408_174410.pkl",
                  pipeline_path="./resources/models/xgb_income_type_classifier_preprocessing_20250408_174410.pkl")
    
    # 設置特徵工程管道
    fe.feature_pipeline = mm.feature_pipeline
    
    # 載入並處理資料
    raw_data = dp.load_data(test_file)
    cleaned_data = dp.clean_data()
    
    # 測試特徵工程
    print("測試特徵工程...")
    processed_data = fe.preprocess_data(cleaned_data)
    # fe.create_feature_pipeline()
    # features = fe.fit_transform(processed_data)
    features = fe.transform(processed_data)
    
    print(f"特徵矩陣形狀: {features.shape}")
    print("特徵工程測試完成!")