from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant
from PyQt6.QtWidgets import QTableView, QAbstractItemView, QHeaderView
import pandas as pd
import numpy as np


class PandasTableModel(QAbstractTableModel):
    """
    用於在QTableView中顯示pandas DataFrame的模型
    """
    
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self._data.columns)
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
        
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            value = self._data.iloc[index.row(), index.column()]
            
            # 格式化日期時間
            if pd.isna(value):
                return ""
            if isinstance(value, pd.Timestamp):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(value, pd.Period):
                return str(value)
            
            # 格式化數值
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    return ""
                if isinstance(value, int):
                    return str(value)
                if value == int(value):
                    return str(int(value))
                return f"{value:.2f}"
            
            return str(value)
        
        # 突出顯示預測列
        if role == Qt.ItemDataRole.BackgroundRole:
            col_name = self._data.columns[index.column()]
            if col_name == 'predicted_income_type':
                return Qt.GlobalColor.lightGray
        
        return QVariant()
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        
        return QVariant()
    
    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class ResultsTableView(QTableView):
    """
    用於顯示預測結果的表格視圖
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(True)
    
    def resizeEvent(self, event):
        """調整列寬以適應表格大小"""
        super().resizeEvent(event)
        if self.model() is not None:
            width = self.viewport().width()
            col_count = self.model().columnCount()
            if col_count > 0:
                for col in range(col_count):
                    self.setColumnWidth(int(col), int(width / col_count))