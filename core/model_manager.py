import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    負責模型管理的類別，遵循單一職責原則
    主要功能：
    1. 訓練模型
    2. 評估模型
    3. 保存和加載模型
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_pipeline = None
        self.trained = False
        
    def train(self, X: np.ndarray, y: pd.Series, params: Optional[Dict[str, Any]] = None) -> XGBClassifier:
        """訓練XGBoost分類器"""
        logger.info("Training model...")
        
        # 對標籤進行編碼
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 設置模型參數
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.01,
                'max_depth': 10,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 1.0,
                'objective': 'multi:softprob',
                'random_state': 42
            }
        
        # 創建模型
        self.model = XGBClassifier(
            **params,
            num_class=len(self.label_encoder.classes_)
        )
        
        # 訓練模型
        self.model.fit(X, y_encoded)
        self.trained = True
        
        logger.info("Model training complete")
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用模型進行預測"""
        if self.model is None or not self.trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        logger.info("Making predictions...")
        y_pred = self.model.predict(X)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        logger.info("Predictions complete")
        
        return y_pred_labels
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測類別概率"""
        if self.model is None or not self.trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        logger.info("Predicting class probabilities...")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """評估模型性能"""
        if self.model is None or not self.trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        logger.info("Evaluating model...")
        y_pred = self.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average='macro')
        weighted_f1 = f1_score(y, y_pred, average='weighted')
        
        report = classification_report(y, y_pred, output_dict=True)
        
        evaluation = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'classification_report': report
        }
        
        logger.info(f"Model evaluation: Accuracy = {accuracy:.4f}, Macro F1 = {macro_f1:.4f}")
        return evaluation
    
    def save_model(self, directory: Union[str, Path], 
                   filename_prefix: str = "withholding_tax_model") -> Tuple[str, str, str]:
        """保存模型及其組件"""
        if self.model is None or not self.trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = directory / f"{filename_prefix}_{timestamp}.pkl"
        encoder_path = directory / f"{filename_prefix}_encoder_{timestamp}.pkl"
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(self.model, model_path)
        
        logger.info(f"Saving label encoder to {encoder_path}")
        joblib.dump(self.label_encoder, encoder_path)
        
        # 如果有feature_pipeline，也要保存
        pipeline_path = None
        if self.feature_pipeline is not None:
            pipeline_path = directory / f"{filename_prefix}_pipeline_{timestamp}.pkl"
            logger.info(f"Saving feature pipeline to {pipeline_path}")
            joblib.dump(self.feature_pipeline, pipeline_path)
        
        return str(model_path), str(encoder_path), str(pipeline_path) if pipeline_path else None
    
    def load_model(self, model_path: Union[str, Path], encoder_path: Union[str, Path], 
                   pipeline_path: Optional[Union[str, Path]] = None) -> None:
        """載入模型及其組件"""
        logger.info(f"Loading model from {model_path}")
        
        model_path = Path(model_path)
        encoder_path = Path(encoder_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        if pipeline_path is not None:
            pipeline_path = Path(pipeline_path)
            if not pipeline_path.exists():
                logger.warning(f"Pipeline file not found: {pipeline_path}")
            else:
                logger.info(f"Loading feature pipeline from {pipeline_path}")
                self.feature_pipeline = joblib.load(pipeline_path)
        
        self.trained = True
        logger.info("Model loaded successfully")
    
    def set_feature_pipeline(self, pipeline):
        """設置特徵工程管道"""
        self.feature_pipeline = pipeline