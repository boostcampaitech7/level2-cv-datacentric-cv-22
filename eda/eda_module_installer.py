import subprocess
import sys

def install_pandas():
    try:
        import pandas as pd
    except ImportError:
        print("pandas 라이브러리가 설치되어 있지 않습니다. 설치를 진행합니다.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd


def install_seaborn():
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn 라이브러리가 설치되어 있지 않습니다. 설치를 진행합니다.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns