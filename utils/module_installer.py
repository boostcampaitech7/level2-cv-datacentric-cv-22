import subprocess
import sys

def install_sklearn():
    try:
        from sklearn.model_selection import KFold
    except ImportError:
        print("scikit-learn 라이브러리가 설치되어 있지 않습니다. 설치를 진행합니다.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        from sklearn.model_selection import KFold


def install_pandas():
    try:
        import pandas as pd
    except ImportError:
        print("pandas 라이브러리가 설치되어 있지 않습니다. 설치를 진행합니다.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd


def install_tabulate():
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulate 라이브러리가 설치되어 있지 않습니다. 설치를 진행합니다.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        from tabulate import tabulate