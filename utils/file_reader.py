import os
import json

from module_installer import install_pandas
install_pandas()
import pandas as pd

import cv2
from PIL import Image
import numpy as np

def read_file(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.txt']:
        # 텍스트 파일 읽기
        with open(file_path, 'r') as f:
            return f.read()
    elif ext in ['.csv']:
        # CSV 파일 읽기
        return pd.read_csv(file_path)
    elif ext in ['.json']:
        # JSON 파일 읽기
        with open(file_path, 'r') as f:
            return json.load(f)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # 이미지 파일 읽기 (Pillow 사용)
        return Image.open(file_path)
    elif ext in ['.npy']:
        return np.load(file_path)
    elif ext in ['.bin']:
        with open(file_path, 'rb') as f:
            return f.read()    
    else:
        raise ValueError(f"Unsupported file type: {ext}")