import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QTabWidget, QTableView, QHeaderView, QComboBox, QLineEdit,
    QStatusBar, QSpinBox, QToolBar, QDialog, QDialogButtonBox,
    QFormLayout, QSplitter, QStackedWidget, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QIcon, QFont, QPixmap

import pandas as pd
import numpy as np

from core.data_processor import DataProcessor
from core.feature_engineering import FeatureEngineer
from core.model_manager import ModelManager
from core.predictor import Predictor
from utils.config import Config
from utils.logger import get_logger
from utils.validators import DataValidator
from ui.widgets import PandasTableModel, ResultsTableView

logger = get_logger(__name__)
config = Config()


class PredictionWorker(QThread):
    """
    背景工作處理預測任務的執行緒
    """
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, predictor: Predictor, file_path: str):
        super().__init__()
        self.predictor = predictor
        self.file_path = file_path
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # 預測
            predictions_df = self.predictor.predict_from_file(self.file_path)
            
            self.progress.emit(90)
            
            # 發出結果
            self.result.emit(predictions_df)
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            self.error.emit(str(e))
            self.finished.emit()


class SaveResultsWorker(QThread):
    """
    背景工作處理儲存預測結果的執行緒
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, predictor: Predictor, df: pd.DataFrame, output_path: str, format: str):
        super().__init__()
        self.predictor = predictor
        self.df = df
        self.output_path = output_path
        self.format = format
    
    def run(self):
        try:
            self.progress.emit(50)
            
            # 保存結果
            saved_path = self.predictor.save_predictions(
                self.df, self.output_path, self.format
            )
            
            self.progress.emit(100)
            self.finished.emit(saved_path)
        except Exception as e:
            logger.error(f"Save results error: {e}", exc_info=True)
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """主視窗類別"""
    
    def __init__(self):
        super().__init__()
        
        self.predictor = None
        self.results_df = None
        
        self.init_ui()
        self.load_config()
        self.setup_predictor()
    
    def init_ui(self):
        """初始化使用者介面"""
        self.setWindowTitle("稅務扣繳類型自動分類系統")
        self.setMinimumSize(1000, 700)
        
        # 創建中央視窗
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QVBoxLayout(central_widget)
        
        # 頂部配置區域
        settings_group = QGroupBox("配置")
        settings_layout = QHBoxLayout(settings_group)
        
        # 模型配置
        model_group = QGroupBox("模型")
        model_layout = QFormLayout(model_group)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_browse_btn = QPushButton("瀏覽...")
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.model_path_browse_btn)
        model_layout.addRow("模型路徑:", model_path_layout)
        
        self.encoder_path_edit = QLineEdit()
        self.encoder_path_edit.setReadOnly(True)
        self.encoder_path_browse_btn = QPushButton("瀏覽...")
        encoder_path_layout = QHBoxLayout()
        encoder_path_layout.addWidget(self.encoder_path_edit)
        encoder_path_layout.addWidget(self.encoder_path_browse_btn)
        model_layout.addRow("編碼器路徑:", encoder_path_layout)
        
        self.pipeline_path_edit = QLineEdit()
        self.pipeline_path_edit.setReadOnly(True)
        self.pipeline_path_browse_btn = QPushButton("瀏覽...")
        pipeline_path_layout = QHBoxLayout()
        pipeline_path_layout.addWidget(self.pipeline_path_edit)
        pipeline_path_layout.addWidget(self.pipeline_path_browse_btn)
        model_layout.addRow("特徵管道路徑:", pipeline_path_layout)
        
        self.reload_model_btn = QPushButton("重新載入模型")
        model_layout.addRow("", self.reload_model_btn)
        
        # 輸入/輸出配置
        io_group = QGroupBox("輸入/輸出")
        io_layout = QFormLayout(io_group)
        
        self.input_dir_edit = QLineEdit()
        self.input_dir_browse_btn = QPushButton("瀏覽...")
        input_dir_layout = QHBoxLayout()
        input_dir_layout.addWidget(self.input_dir_edit)
        input_dir_layout.addWidget(self.input_dir_browse_btn)
        io_layout.addRow("輸入目錄:", input_dir_layout)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_browse_btn = QPushButton("瀏覽...")
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_browse_btn)
        io_layout.addRow("輸出目錄:", output_dir_layout)
        
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["Excel", "CSV", "Parquet"])
        io_layout.addRow("輸出格式:", self.output_format_combo)
        
        # 添加到設置佈局
        settings_layout.addWidget(model_group, 1)
        settings_layout.addWidget(io_group, 1)
        
        # 操作區域
        operation_layout = QHBoxLayout()
        
        # 檔案選擇區域
        file_group = QGroupBox("檔案選擇")
        file_layout = QVBoxLayout(file_group)
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_browse_btn = QPushButton("瀏覽檔案")
        self.file_path_browse_btn.setMinimumHeight(40)
        
        file_layout.addWidget(QLabel("選擇要處理的檔案:"))
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.file_path_browse_btn)
        file_layout.addStretch()
        
        # 執行區域
        execute_group = QGroupBox("執行")
        execute_layout = QVBoxLayout(execute_group)
        
        self.predict_btn = QPushButton("預測")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.setEnabled(False)
        
        self.save_btn = QPushButton("儲存結果")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        execute_layout.addWidget(self.predict_btn)
        execute_layout.addWidget(self.save_btn)
        execute_layout.addWidget(QLabel("處理進度:"))
        execute_layout.addWidget(self.progress_bar)
        
        # 添加到操作佈局
        operation_layout.addWidget(file_group, 1)
        operation_layout.addWidget(execute_group, 1)
        
        # 結果顯示區域
        results_group = QGroupBox("預測結果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = ResultsTableView()
        
        results_layout.addWidget(self.results_table)
        
        # 添加所有佈局到主佈局
        main_layout.addWidget(settings_group)
        main_layout.addLayout(operation_layout)
        main_layout.addWidget(results_group)
        
        # 狀態列
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 設置初始狀態
        self.update_status("就緒")
        
        # 連接按鈕事件
        self.connect_events()
    
    def connect_events(self):
        """連接事件處理函數"""
        # 設置相關
        self.model_path_browse_btn.clicked.connect(self.browse_model_path)
        self.encoder_path_browse_btn.clicked.connect(self.browse_encoder_path)
        self.pipeline_path_browse_btn.clicked.connect(self.browse_pipeline_path)
        self.reload_model_btn.clicked.connect(self.reload_model)
        
        self.input_dir_browse_btn.clicked.connect(self.browse_input_dir)
        self.output_dir_browse_btn.clicked.connect(self.browse_output_dir)
        
        # 檔案相關
        self.file_path_browse_btn.clicked.connect(self.browse_file)
        
        # 執行相關
        self.predict_btn.clicked.connect(self.start_prediction)
        self.save_btn.clicked.connect(self.save_results)
    
    def load_config(self):
        """載入配置設定"""
        # 模型路徑
        model_path = config.get("model.model_path")
        if model_path:
            self.model_path_edit.setText(model_path)
        
        encoder_path = config.get("model.encoder_path")
        if encoder_path:
            self.encoder_path_edit.setText(encoder_path)
        
        pipeline_path = config.get("model.pipeline_path")
        if pipeline_path:
            self.pipeline_path_edit.setText(pipeline_path)
        
        # 輸入/輸出目錄
        input_dir = config.get("data.default_input_dir")
        if input_dir:
            self.input_dir_edit.setText(input_dir)
        
        output_dir = config.get("data.default_output_dir")
        if output_dir:
            self.output_dir_edit.setText(output_dir)
    
    def setup_predictor(self):
        """設置預測器"""
        model_path = self.model_path_edit.text()
        encoder_path = self.encoder_path_edit.text()
        pipeline_path = self.pipeline_path_edit.text()
        
        # 驗證模型文件
        if model_path and encoder_path:
            valid, error_msg = DataValidator.validate_model_files(
                model_path, encoder_path, pipeline_path if pipeline_path else None
            )
            
            if valid:
                try:
                    self.predictor = Predictor(model_path, encoder_path, pipeline_path)
                    self.update_status("模型載入成功")
                except Exception as e:
                    logger.error(f"Error loading model: {e}", exc_info=True)
                    self.show_error("模型載入錯誤", f"載入模型時發生錯誤: {e}")
                    self.update_status("模型載入失敗")
            else:
                logger.warning(f"Invalid model files: {error_msg}")
                self.update_status("模型檔案無效")
        else:
            self.predictor = Predictor()  # 初始化無模型的預測器
            self.update_status("未載入模型")
    
    def browse_model_path(self):
        """瀏覽選擇模型檔案"""
        model_dir = Path(config.get("model.default_model_dir"))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇模型檔案", str(model_dir), "模型檔案 (*.pkl)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
            config.set("model.model_path", file_path)
    
    def browse_encoder_path(self):
        """瀏覽選擇編碼器檔案"""
        model_dir = Path(config.get("model.default_model_dir"))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇編碼器檔案", str(model_dir), "編碼器檔案 (*.pkl)"
        )
        
        if file_path:
            self.encoder_path_edit.setText(file_path)
            config.set("model.encoder_path", file_path)
    
    def browse_pipeline_path(self):
        """瀏覽選擇特徵管道檔案"""
        model_dir = Path(config.get("model.default_model_dir"))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇特徵管道檔案", str(model_dir), "特徵管道檔案 (*.pkl)"
        )
        
        if file_path:
            self.pipeline_path_edit.setText(file_path)
            config.set("model.pipeline_path", file_path)
    
    def reload_model(self):
        """重新載入模型"""
        self.setup_predictor()
    
    def browse_input_dir(self):
        """瀏覽選擇輸入目錄"""
        input_dir = QFileDialog.getExistingDirectory(
            self, "選擇輸入目錄", self.input_dir_edit.text()
        )
        
        if input_dir:
            self.input_dir_edit.setText(input_dir)
            config.set("data.default_input_dir", input_dir)
    
    def browse_output_dir(self):
        """瀏覽選擇輸出目錄"""
        output_dir = QFileDialog.getExistingDirectory(
            self, "選擇輸出目錄", self.output_dir_edit.text()
        )
        
        if output_dir:
            self.output_dir_edit.setText(output_dir)
            config.set("data.default_output_dir", output_dir)
    
    def browse_file(self):
        """瀏覽選擇檔案"""
        input_dir = self.input_dir_edit.text()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇檔案", input_dir, 
            "支援的檔案 (*.xlsx *.xls *.csv *.parquet);;Excel檔案 (*.xlsx *.xls);;CSV檔案 (*.csv);;Parquet檔案 (*.parquet)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            self.predict_btn.setEnabled(True)
            self.update_status("已選擇檔案，可以開始預測")
    
    def start_prediction(self):
        """開始預測"""
        if not self.predictor:
            self.show_error("預測錯誤", "預測器未初始化，請先載入模型")
            return
        
        file_path = self.file_path_edit.text()
        if not file_path:
            self.show_error("預測錯誤", "請先選擇檔案")
            return
        
        # 驗證檔案路徑
        valid, error_msg = DataValidator.validate_file_path(file_path)
        if not valid:
            self.show_error("檔案錯誤", error_msg)
            return
        
        # 禁用按鈕
        self.predict_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 創建並運行背景工作
        self.prediction_worker = PredictionWorker(self.predictor, file_path)
        self.prediction_worker.progress.connect(self.update_progress)
        self.prediction_worker.result.connect(self.handle_prediction_result)
        self.prediction_worker.error.connect(self.handle_prediction_error)
        self.prediction_worker.finished.connect(self.prediction_finished)
        
        self.update_status("正在進行預測...")
        self.prediction_worker.start()
    
    def save_results(self):
        """儲存預測結果"""
        if self.results_df is None:
            self.show_error("儲存錯誤", "沒有可以儲存的結果")
            return
        
        # 準備輸出文件名
        output_dir = self.output_dir_edit.text()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        format_idx = self.output_format_combo.currentIndex()
        format_map = {0: 'excel', 1: 'csv', 2: 'parquet'}
        output_format = format_map.get(format_idx, 'excel')
        
        file_extensions = {'excel': '.xlsx', 'csv': '.csv', 'parquet': '.parquet'}
        extension = file_extensions.get(output_format, '.xlsx')
        
        # 建議的檔名
        input_file = Path(self.file_path_edit.text()).name
        input_stem = Path(input_file).stem
        default_name = f"{input_stem}_predicted_{timestamp}{extension}"
        
        # 讓使用者選擇儲存路徑
        file_filters = {
            'excel': "Excel 檔案 (*.xlsx)",
            'csv': "CSV 檔案 (*.csv)",
            'parquet': "Parquet 檔案 (*.parquet)"
        }
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "儲存預測結果", str(Path(output_dir) / default_name), 
            file_filters.get(output_format, "Excel 檔案 (*.xlsx)")
        )
        
        if not output_path:
            return
        
        # 禁用按鈕
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 創建並運行背景工作
        self.save_worker = SaveResultsWorker(self.predictor, self.results_df, output_path, output_format)
        self.save_worker.progress.connect(self.update_progress)
        self.save_worker.finished.connect(self.handle_save_finished)
        self.save_worker.error.connect(self.handle_save_error)
        
        self.update_status("正在儲存結果...")
        self.save_worker.start()
    
    def update_progress(self, value):
        """更新進度條"""
        self.progress_bar.setValue(value)
    
    def handle_prediction_result(self, df):
        """處理預測結果"""
        self.results_df = df
        
        # 顯示結果表格
        model = PandasTableModel(df)
        self.results_table.setModel(model)
        self.results_table.resizeColumnsToContents()
        
        # 啟用儲存按鈕
        self.save_btn.setEnabled(True)
        
        # 更新狀態
        self.update_status(f"預測完成，共 {len(df)} 筆資料")
    
    def handle_prediction_error(self, error_msg):
        """處理預測錯誤"""
        self.show_error("預測錯誤", f"預測過程中發生錯誤: {error_msg}")
        self.update_status("預測失敗")
        self.predict_btn.setEnabled(True)
    
    def prediction_finished(self):
        """預測完成處理"""
        self.predict_btn.setEnabled(True)
    
    def handle_save_finished(self, saved_path):
        """儲存完成處理"""
        self.save_btn.setEnabled(True)
        self.update_status(f"結果已儲存到 {saved_path}")
        
        # 詢問是否打開檔案
        reply = QMessageBox.question(
            self, "儲存完成", f"結果已成功儲存到 {saved_path}\n\n要打開檔案位置嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 打開檔案所在的目錄
            os.startfile(str(Path(saved_path).parent))
    
    def handle_save_error(self, error_msg):
        """處理儲存錯誤"""
        self.show_error("儲存錯誤", f"儲存過程中發生錯誤: {error_msg}")
        self.update_status("儲存失敗")
        self.save_btn.setEnabled(True)
    
    def update_status(self, message):
        """更新狀態欄訊息"""
        self.status_bar.showMessage(message)
    
    def show_error(self, title, message):
        """顯示錯誤訊息"""
        QMessageBox.critical(self, title, message)


def run_app():
    """運行應用程式"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()