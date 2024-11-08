import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict

from data_loader import load_data

def visualize_and_save(data_dict: Dict[str, Dict[str, List[Dict]]], output_dir: str):
    """
        이미지를 저장하는 함수
    """
    for lang, splits in data_dict.items():
        for split, images in splits.items():
            for image_data_list in images:
                
                if not image_data_list:
                    continue
                    
                image_path = image_data_list[0]['image_path']
                image_name = image_data_list[0]['image_name']

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Image {image_name} not found in {image_path}.")
                    continue

                # OpenCV의 BGR 이미지를 RGB로 변환하여 시각화 준비
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                for word_data in image_data_list:
                    points = word_data['points']
                    word_id = word_data['word_id']

                    # 바운딩 박스 그리기
                    if points:
                        pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        image_rgb = cv2.polylines(image_rgb, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

                        x, y = int(points[0][0]), int(points[0][1])
                        image_rgb = cv2.putText(
                            image_rgb, str(word_id), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA 
                        )

                
                plt.figure(figsize=(10, 10))
                plt.imshow(image_rgb)
                plt.title(f"{lang} - {split} - {image_name}")
                plt.axis('off')

                # 저장 경로 설정
                lang_output_dir = os.path.join(output_dir, lang, split)
                os.makedirs(lang_output_dir, exist_ok=True)
                output_path = os.path.join(lang_output_dir, f"visualized_{image_name}")
                plt.savefig(output_path)
                plt.close()


root = '/data/ephemeral/home/yeyechu_baby'

# ───────────────────────────────── 설정 ─────────────────────────────────

input_dir = root + '/repo/code/data' 
languages = ['chinese', 'japanese', 'thai', 'vietnamese']

split = 'train'  # 'train', 'validation' 에서 선택
output_dir = root + f'/image_from_{split}_rerelabling'

word = ''

# ───────────────────────────────────────────────────────────────────────

data_dict = load_data(input_dir, languages, split, word)
visualize_and_save(data_dict, output_dir)