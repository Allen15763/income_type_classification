import os
import sys
import shutil
from pathlib import Path
import PyInstaller.__main__

"""
打包應用程式為可執行檔
使用方法: python build.py
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
    
    # 根據作業系統選擇正確的分隔符 linus和macOS使用':'，Windows使用';'
    separator = ';' if sys.platform.startswith('win') else ':'
    
    # 設置PyInstaller參數
    pyinstaller_args = [
        "main.py",                         # 主入口點
        "--name=income_type_classification",    # 輸出的執行檔名稱
        "--windowed",                      # 不顯示控制台窗口
        "--onedir",                        # 打包為一個目錄
        "--clean",                         # 清理臨時文件
        f"--add-data=resources{separator}resources",  # 添加資源文件
    ]
    
    # 添加隱含導入
    hidden_imports = [
        # 基本模組
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=xgboost",
        
        # Scikit-learn 模組
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.preprocessing",
        "--hidden-import=sklearn.feature_extraction.text",
        "--hidden-import=sklearn.compose",
        "--hidden-import=sklearn.pipeline",
        "--hidden-import=sklearn.impute",
        "--hidden-import=sklearn.base",
        "--hidden-import=sklearn.utils",
        
        # Feature-engine 模組
        "--hidden-import=feature_engine.transformation",
        
        # Joblib 模組
        "--hidden-import=joblib",
        
        # SciPy 模組和子模組
        "--hidden-import=scipy",
        "--hidden-import=scipy.sparse",
        "--hidden-import=scipy._lib",
        "--hidden-import=scipy._lib.array_api_compat",
        "--hidden-import=scipy._lib.array_api_compat.numpy",
        "--hidden-import=scipy._lib.array_api_compat.numpy.fft",
        "--hidden-import=scipy.fft",
        "--hidden-import=scipy.sparse._sputils",
        "--hidden-import=scipy.sparse._base",
        "--hidden-import=scipy._lib._util",
        "--hidden-import=scipy._lib.array_api",
        
        # PyQt 相關模組
        "--hidden-import=PyQt6",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=PyQt6.QtGui",
        
        # 其他可能的依賴
        "--hidden-import=numbers",
        "--hidden-import=packaging.version",
        "--hidden-import=packaging.specifiers",
        "--hidden-import=packaging.requirements"
    ]
    
    pyinstaller_args.extend(hidden_imports)
    
    # 添加其他有用的選項
    pyinstaller_args.extend([
        "--log-level=DEBUG",    # 更詳細的日志輸出
        "--noconfirm",          # 不詢問是否覆蓋
    ])
    
    # 如果圖標存在，添加圖標參數
    icon_path = Path("resources/icon.ico")
    if icon_path.exists():
        pyinstaller_args.append(f"--icon={icon_path}")
    
    print("Running PyInstaller with arguments:")
    print(" ".join(pyinstaller_args))
    
    # 運行PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)
    
    # 複製預訓練模型到輸出目錄
    models_dir = Path("resources/models")
    if models_dir.exists():
        dist_models_dir = dist_dir / "income_type_classification" / "resources" / "models"
        dist_models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_file in models_dir.glob("*.pkl"):
            shutil.copy(model_file, dist_models_dir)
    
    print(f"應用程式已打包到 {dist_dir / 'income_type_classification'}")


if __name__ == "__main__":
    # 創建resources目錄
    resources_dir = Path("resources")
    resources_dir.mkdir(exist_ok=True)
    
    # 創建models目錄
    models_dir = Path("resources/models")
    models_dir.mkdir(exist_ok=True)
    
    # 檢查圖標是否存在
    icon_path = resources_dir / "icon.ico"
    if not icon_path.exists():
        print("警告: 找不到圖標文件 resources/icon.ico，應用程式將使用默認圖標")
    
    try:
        build_app()
        print("打包完成!")
    except Exception as e:
        print(f"打包過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()