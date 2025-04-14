import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class DataProcessor:
    mapping_columns = ['GL Date', 'payment_date', 'supplier', 'line_desc', 
                       'supplier_type', 'year_month', 'cleaned_line_desc', 'amount']
    """
    負責處理原始數據的類別，實現單一職責原則(SRP)
    主要處理：
    1. 讀取不同格式的文件
    2. 基本數據清理和格式轉換
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """載入資料檔案(Excel, CSV)"""
        file_path = Path(file_path)
        logger.info(f"Loading data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        if file_path.suffix.lower() == '.xlsx' or file_path.suffix.lower() == '.xls':
            try:
                # 處理額外的跳過列，依照原notebook邏輯處理
                self.raw_data = pd.read_excel(file_path, dtype=str, header=2)
                logger.info(f"Loaded Excel file with {self.raw_data.shape[0]} ",
                            f"rows and {self.raw_data.shape[1]} columns")
            except Exception as e:
                logger.error(f"Error loading Excel file: {e}")
                raise
        elif file_path.suffix.lower() == '.csv':
            try:
                self.raw_data = pd.read_csv(file_path, dtype=str)
                logger.info(f"Loaded CSV file with {self.raw_data.shape[0]} rows and {self.raw_data.shape[1]} columns")
            except Exception as e:
                logger.error(f"Error loading CSV file: {e}")
                raise
        elif file_path.suffix.lower() == '.parquet':
            try:
                self.raw_data = pd.read_parquet(file_path)
                logger.info(f"Loaded Parquet file with {self.raw_data.shape[0]} ",
                            f"rows and {self.raw_data.shape[1]} columns")
            except Exception as e:
                logger.error(f"Error loading Parquet file: {e}")
                raise
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self.raw_data

    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """進行基本數據清理"""
        if data is not None:
            self.raw_data = data
            
        if self.raw_data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        logger.info("Cleaning data...")
        df = self.raw_data.copy()
        
        # 處理日期欄位
        date_columns = ['GL Date', 'Payment Date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
        
        # 移除標題列（如果意外包含）
        if 'GL Date' in df.columns:
            df = df.loc[df['GL Date'] != 'GL Date', :]
        
        # 移除重複行
        df = df.drop_duplicates().reset_index(drop=True)
        
        # 移除空日期行
        if 'Payment Date' in df.columns:
            df = df.loc[~df['Payment Date'].isna()].reset_index(drop=True)
        
        self.processed_data = df
        logger.info(f"Data cleaned. Remaining rows: {df.shape[0]}")
        return self.processed_data
    
    def extract_features(self, required_columns: List[str] = None) -> pd.DataFrame:
        """提取所需列，準備進行特徵工程"""
        if self.processed_data is None:
            if self.raw_data is not None:
                self.clean_data()
            else:
                raise ValueError("No data loaded. Please load data first.")
                
        logger.info("Extracting features...")
        
        # 預設所需列
        if required_columns is None:
            required_columns = [
                'GL Date', 'AP Voucher No', 'Payment Date', 'Remit-to Supplier', 
                'Supplier ID', 'Supplier Name', 'Supplier Tax Number', 'LineAccount',
                'Line Desc', 'Line Amount', 'W/H Tax', 'W/H Tax Name', 
                'Income Type', 'Income Type Name'
            ]
        
        # 確認所需列是否存在
        missing_columns = [col for col in required_columns if col not in self.processed_data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # 對缺失列創建空值
            for col in missing_columns:
                self.processed_data[col] = np.nan
        
        # 提取所需列
        selected_data = self.processed_data[required_columns].copy()
        logger.info(f"Features extracted. Shape: {selected_data.shape}")
        
        return selected_data

    @staticmethod
    def remove_control_chars_from_series(series: pd.Series, 
                                         pattern: str = r'[\u202a-\u202e\u200e\u200f\ufeff\u200b-\u200d]') -> pd.Series:
        """移除Unicode控制字符"""
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a Pandas Series.")

        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern}. Error: {e}") from e

        def clean_value(value):
            if pd.isna(value):
                return value
            try:
                str_value = str(value)
                return compiled_pattern.sub('', str_value)
            except Exception as e:
                logger.warning(f"Warning: Could not process value '{value}'. Error: {e}")
                return value

        if series.dtype == 'string': 
            cleaned_series = series.str.replace(compiled_pattern, '', regex=True)
        else:
            cleaned_series = series.apply(clean_value)

        return cleaned_series.astype(series.dtype if series.dtype in ['string', 'object'] else object)

    @staticmethod
    def clean_desc_optimized(data_series: pd.Series) -> pd.Series:
        """最佳化描述欄位清理"""
        # 編譯一次正則表達式以提高效率
        pattern1 = re.compile(r'\D')
        pattern2 = re.compile(r'[^\/\.\、\(\)\&\[\]]')

        # 使用向量化字串操作
        data_series = data_series.str.replace('7-11', 'seven', regex=False)
        data_series = data_series.str.replace('711', 'seven', regex=False)
        data_series = data_series.str.replace('美聯社', '美廉社', regex=False)
        data_series = data_series.str.replace('(?i)Cross Border|Cross-Border', 'CB')
        data_series = data_series.str.replace('(?i)Digital Product', 'DP')

        def clean_string(x):
            if pd.isna(x):
                return x
            x = ''.join(pattern1.findall(x))
            x = ''.join(pattern2.findall(x))
            x = x.strip().lstrip('-_').rstrip('_')
            x = re.sub(r'_+|-+|,|:|【|】|"|\||#|\｜|\%|^', ' ', x)
            x = re.sub(r'\s{2,}', ' ', x).strip()
            x = re.sub('\u2765', '', x)
            x = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', x)
            return x

        new_series = data_series.apply(clean_string)

        # 移除其他項目描述
        new_series = new_series.str.extract(r'([^\[\]]+)', expand=False).fillna(new_series)

        return new_series

    @staticmethod
    def classify_supplier(row: pd.Series) -> str:
        """依供應商ID和名稱分類供應商類型"""
        supplier_id = str(row['Supplier ID']) if not pd.isna(row['Supplier ID']) else ""
        supplier_name = str(row['Remit-to Supplier']) if not pd.isna(row['Remit-to Supplier']) else ""
        
        hospital_keywords = ['醫療財團法人', '醫療社團法人', '市立', '縣立', '醫學大學附設']
        clinic_keywords = ['醫院', '診所']

        if len(supplier_id) == 10:
            return '個人'
        elif len(supplier_id) == 8 and '信託' in supplier_name:
            return '個人'
        elif len(supplier_id) == 8 and '事務所' in supplier_name:
            return '事務所'
        elif len(supplier_id) == 8 and any(keyword in supplier_name for keyword in hospital_keywords):
            return '醫院'
        elif len(supplier_id) == 8 and any(keyword in supplier_name for keyword in clinic_keywords):
            return '診所'
        elif len(supplier_id) == 8:
            return '企業'
        else:
            return '其他'
        
    @classmethod
    def group_by_supplier(self, df: pd.DataFrame) -> pd.DataFrame:
        """根據付款供應商名稱分組並計算金額總和
            - 計算在同一付款/GL日下的同一供應商的借貸淨額，大於零即為付款
                - 等於或小於零不視為付款，故不申報對方收入項目。
        """
        df_copy = df.copy()
        suppliers = df_copy['supplier'].unique()
        necessary_cols = self.mapping_columns.copy()
        grouped_cols = necessary_cols[:-1]  # 所有必要列，除了'amount'
        
        dfs = []
        for supplier in suppliers:
            df_supplier = df_copy.loc[df_copy['supplier'] == supplier, necessary_cols]
            df_grouped = df_supplier.groupby(grouped_cols)\
                .agg(amount=pd.NamedAgg(column="amount", aggfunc="sum")).reset_index()
            dfs.append(df_grouped)

        df_grouped = pd.concat(dfs, ignore_index=True)
        df_grouped = df_grouped.loc[df_grouped['amount'] > 0, :].reset_index(drop=True)
        return df_grouped
    
    @classmethod
    def create_mapping_key(self, df: pd.DataFrame) -> Dict[str, str]:
        df_copy = df.copy()
        columns_to_concat = [col for col in self.mapping_columns if col != 'amount']
        df_copy['mapping_key'] = df_copy.apply(lambda row: "".join(row[columns_to_concat].astype(str)), axis=1)
        return df_copy
    
    @classmethod
    def copy_amount_for_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """將金額複製到log欄位"""
        df_copy = df.copy()
        df_copy['origin_amount'] = df_copy['amount']
        df_copy['amount'] = df_copy['amount'].abs().add(1)
        return df_copy
    
    @classmethod
    def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除不必要的列"""
        # TODO
        df_copy = df.copy()
        unnecessary_cols = [col for col in df_copy.columns if col not in self.mapping_columns]
        df_copy.drop(columns=unnecessary_cols, inplace=True, errors='ignore')
        return df_copy
    
    def reformat_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化結果, columns refer to self.extract_features"""
        df_copy = df.copy()
        df_copy = df_copy.assign(
            amount=df_copy['origin_amount']
        ).drop(columns=['origin_amount', 'mapping_key'])
        return df_copy