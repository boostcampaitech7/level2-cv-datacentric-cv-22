import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def merge_split_images(input_dir, output_dir):
    """16등분된 SR 이미지들을 원래 형태로 합치기"""
    
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 모든 이미지 파일 가져오기
    image_files = [f for f in os.listdir(input_dir) if f.endswith('_x2_SR.png')]
    
    # 기본 이미지 이름 추출 (숫자와 SR 부분 제외)
    base_names = set()
    for f in image_files:
        # _x2_SR.png 제거
        name_without_suffix = f.replace('_x2_SR.png', '')
        # 마지막 숫자 제거
        base_name = '_'.join(name_without_suffix.split('_')[:-1])
        base_names.add(base_name)
    
    print(f"Found {len(base_names)} images to merge")
    
    for base_name in tqdm(base_names, desc="Merging images"):
        split_images = []
        
        # 16개의 분할 이미지 로드
        for i in range(1, 17):
            split_name = f"{base_name}_{i}_x2_SR.png"
            split_path = os.path.join(input_dir, split_name)
            
            if not os.path.exists(split_path):
                print(f"Warning: Missing split image {split_name}")
                continue
                
            img = cv2.imread(split_path)
            split_images.append(img)
        
        # 모든 분할 이미지가 있는지 확인
        if len(split_images) != 16:
            print(f"Error: Could not find all 16 splits for {base_name}")
            continue
        
        # 이미지 크기 계산
        h, w = split_images[0].shape[:2]
        merged_h = h * 4
        merged_w = w * 4
        
        # 빈 이미지 생성
        merged = np.zeros((merged_h, merged_w, 3), dtype=np.uint8)
        
        # 4x4 그리드로 이미지 합치기
        for idx in range(16):
            i = idx // 4  # 세로 인덱스
            j = idx % 4   # 가로 인덱스
            
            y_start = i * h
            y_end = (i + 1) * h
            x_start = j * w
            x_end = (j + 1) * w
            
            merged[y_start:y_end, x_start:x_end] = split_images[idx]
        
        # 합친 이미지 저장
        output_name = f"{base_name}.png"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, merged)
        print(f"Saved merged image: {output_name} ({merged_h}x{merged_w})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge split SR images back into original form')
    parser.add_argument('--input_dir', type=str, 
                        default="/EDSR-PyTorch/experiment/test/results-Demo",
                        help='Input directory containing split SR images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for merged images')
    
    args = parser.parse_args()
    
    print("Starting image merging process...")
    merge_split_images(args.input_dir, args.output_dir)
    print("Merging completed!")