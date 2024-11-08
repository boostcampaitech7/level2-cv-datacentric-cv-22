import json
import cv2
import os
import numpy as np
from tqdm import tqdm

def draw_boxes_on_image(json_path, img_dir, output_dir):
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 이미지에 대해 처리
    for img_name, img_data in tqdm(data['images'].items()):
        # 이미지 파일 경로
        img_path = os.path.join(img_dir, img_name)
        
        # 이미지가 존재하는지 확인
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # words에 있는 모든 bbox 그리기
        for word_id, word_data in img_data['words'].items():
            points = word_data.get('points', [])
            if points:
                # points를 numpy array로 변환
                points = np.array(points, dtype=np.int32)
                
                # bbox 그리기
                cv2.polylines(img, [points], True, (128, 0, 0), 2)
        
        # 결과 이미지 저장
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    # 처리할 국가 리스트
    countries = ['chinese', 'japanese', 'thai', 'vietnamese']

    # 각 국가별 데이터 처리
    for country in countries:
        json_path = f'code/data/{country}_receipt/ufo/train.json'
        img_dir = f'code/data/{country}_receipt/img/train'
        output_dir = f'new_data/{country}_receipt/train'
        
        print(f"\nProcessing {country} receipts...")
        draw_boxes_on_image(json_path, img_dir, output_dir)