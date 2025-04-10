import os
import sys
import shutil
from pathlib import Path
import PyInstaller.__main__

"""
打包應用程式為可執行檔
使用方法: python build.py
or
pyinstaller main.py --name income_type_classification -F -w --add-data "resources;resources" --icon "resources/icon.ico"
"""

def build_app():
    """打包應用程式"""
    # 清理舊的構建文件
    dist_dir = Path("dist")
    build_dir = Path("build")
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    # 設置PyInstaller參數
    pyinstaller_args = [
        "main.py",                         # 主入口點
        "--name=稅務扣繳類型自動分類系統",    # 輸出的執行檔名稱
        "--windowed",                      # 不顯示控制台窗口
        "--onedir",                        # 打包為一個目錄
        "--clean",                         # 清理臨時文件
        "--add-data=resources;resources",  # 添加資源文件 linux use ":" windows use ";"
        "--icon=resources/icon.ico",       # 應用程式圖標
    ]
    
    # 添加隱含導入
    hidden_imports = [
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=xgboost",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.preprocessing",
        "--hidden-import=sklearn.feature_extraction.text",
        "--hidden-import=sklearn.compose",
        "--hidden-import=sklearn.pipeline",
        "--hidden-import=sklearn.impute",
        "--hidden-import=feature_engine.transformation",
        "--hidden-import=joblib",
    ]
    
    pyinstaller_args.extend(hidden_imports)
    
    # 運行PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)
    
    # 複製預訓練模型到輸出目錄
    models_dir = Path("resources/models")
    if models_dir.exists():
        dist_models_dir = dist_dir / "稅務扣繳類型自動分類系統" / "resources" / "models"
        dist_models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_file in models_dir.glob("*.pkl"):
            shutil.copy(model_file, dist_models_dir)
    
    print(f"應用程式已打包到 {dist_dir / '稅務扣繳類型自動分類系統'}")


if __name__ == "__main__":
    # 创建resources目录和图标
    resources_dir = Path("resources")
    resources_dir.mkdir(exist_ok=True)
    
    # 检查图标是否存在，如果不存在则创建一个空图标
    icon_path = resources_dir / "icon.ico"
    if not icon_path.exists():
        print("警告: 找不到圖標文件 resources/icon.ico，請確保在打包前提供圖標")
        
    build_app()