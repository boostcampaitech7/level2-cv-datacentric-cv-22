import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def split_and_save_images(input_dir, output_dir):
    """이미지들을 8등분하여 PNG로 저장"""
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 입력 디렉토리의 모든 이미지 파일 찾기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    # 전체 이미지 진행상황을 보여주는 tqdm
    for image_file in tqdm(image_files, desc="Processing images"):
        # 이미지 읽기
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error reading image: {image_file}")
            continue
            
        h, w = img.shape[:2]
        half_h = h // 2
        quarter_w = w // 4
        
        # 파일 이름에서 확장자 제거
        file_name = Path(image_file).stem
        
        # 8등분하여 PNG로 저장
        idx = 1
        for i in range(2):  # 세로 2등분
            for j in range(4):  # 가로 4등분
                y_start = i * half_h
                y_end = (i + 1) * half_h
                x_start = j * quarter_w
                x_end = (j + 1) * quarter_w
                
                split = img[y_start:y_end, x_start:x_end]
                
                # 새 파일 이름 생성 (예: image_1.png)
                new_file_name = f"{file_name}_{idx}.png"
                output_path = os.path.join(output_dir, new_file_name)
                
                # 분할된 이미지를 PNG로 저장
                cv2.imwrite(output_path, split)
                idx += 1

def split_16_and_save_images(input_dir, output_dir):
    """이미지들을 16등분하여 PNG로 저장"""
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 입력 디렉토리의 모든 이미지 파일 찾기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    # 전체 이미지 진행상황을 보여주는 tqdm
    for image_file in tqdm(image_files, desc="Processing images"):
        # 이미지 읽기
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error reading image: {image_file}")
            continue
            
        h, w = img.shape[:2]
        quarter_h = h // 4  # 세로를 4등분
        quarter_w = w // 4  # 가로를 4등분
        
        # 파일 이름에서 확장자 제거
        file_name = Path(image_file).stem
        
        # 16등분하여 PNG로 저장
        idx = 1
        for i in range(4):  # 세로 4등분
            for j in range(4):  # 가로 4등분
                y_start = i * quarter_h
                y_end = (i + 1) * quarter_h
                x_start = j * quarter_w
                x_end = (j + 1) * quarter_w
                
                split = img[y_start:y_end, x_start:x_end]
                
                # 새 파일 이름 생성 (예: image_1.png)
                new_file_name = f"{file_name}_{idx}.png"
                output_path = os.path.join(output_dir, new_file_name)
                
                # 분할된 이미지를 PNG로 저장
                cv2.imwrite(output_path, split)
                idx += 1

def check_and_split_images(input_dir, output_dir, split_type=16):
    """이미지 분할 전 체크 및 분할 수행"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    issues_found = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error reading image: {image_file}")
            continue
            
        h, w = img.shape[:2]
        
        # 문제 체크
        issues = []
        
        # 1. 이미지 크기가 분할하기에 적절한지 확인
        if split_type == 16:
            if h % 4 != 0:
                issues.append(f"Height ({h}) is not divisible by 4")
            if w % 4 != 0:
                issues.append(f"Width ({w}) is not divisible by 4")
        else:  # 8등분
            if h % 2 != 0:
                issues.append(f"Height ({h}) is not divisible by 2")
            if w % 4 != 0:
                issues.append(f"Width ({w}) is not divisible by 4")
        
        # 2. 최소 크기 확인
        min_size = 32  # 예시 최소 크기
        if split_type == 16:
            split_h, split_w = h // 4, w // 4
        else:
            split_h, split_w = h // 2, w // 4
            
        if split_h < min_size:
            issues.append(f"Split height ({split_h}) is too small")
        if split_w < min_size:
            issues.append(f"Split width ({split_w}) is too small")
        
        # 3. 이미지 비율 확인
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2:
            issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        if issues:
            issues_found.append({
                'file': image_file,
                'size': f"{w}x{h}",
                'split_size': f"{split_w}x{split_h}",
                'issues': issues
            })
            continue
        
        # 문제가 없는 경우에만 분할 수행
        file_name = Path(image_file).stem
        idx = 1
        
        if split_type == 16:
            quarter_h, quarter_w = h // 4, w // 4
            splits_h, splits_w = 4, 4
        else:
            quarter_h, quarter_w = h // 2, w // 4
            splits_h, splits_w = 2, 4
            
        for i in range(splits_h):
            for j in range(splits_w):
                y_start = i * quarter_h
                y_end = (i + 1) * quarter_h
                x_start = j * quarter_w
                x_end = (j + 1) * quarter_w
                
                split = img[y_start:y_end, x_start:x_end]
                new_file_name = f"{file_name}_{idx}.png"
                output_path = os.path.join(output_dir, new_file_name)
                cv2.imwrite(output_path, split)
                idx += 1
    
    # 문제 보고서 출력
    if issues_found:
        print("\nIssues found in images:")
        for item in issues_found:
            print(f"\nFile: {item['file']}")
            print(f"Size: {item['size']}")
            print(f"Split size would be: {item['split_size']}")
            print("Issues:")
            for issue in item['issues']:
                print(f"  - {issue}")
        print(f"\nTotal images with issues: {len(issues_found)}")
    else:
        print("\nNo issues found. All images were processed successfully.")

def split_and_save_images_with_padding(input_dir, output_dir, split_type=16):
    """패딩을 적용하여 이미지 분할"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # 이미지 읽기
        img = cv2.imread(os.path.join(input_dir, image_file))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 패딩 계산 수정
        if split_type == 16:
            # 16등분의 경우 가로/세로 모두 4의 배수여야 함
            pad_h = (4 - (h % 4)) if (h % 4) != 0 else 0
            pad_w = (4 - (w % 4)) if (w % 4) != 0 else 0
        else:
            # 8등분의 경우 세로는 2의 배수, 가로는 4의 배수여야 함
            pad_h = (2 - (h % 2)) if (h % 2) != 0 else 0
            pad_w = (4 - (w % 4)) if (w % 4) != 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            # 반사 패딩 적용
            img = cv2.copyMakeBorder(img, 
                                   0, pad_h,  # 위, 아래
                                   0, pad_w,  # 왼쪽, 오른쪽
                                   cv2.BORDER_REFLECT)
            
        new_h, new_w = img.shape[:2]
        
        # 패딩 후 크기 확인
        if split_type == 16:
            assert new_h % 4 == 0 and new_w % 4 == 0, f"Padded size ({new_w}x{new_h}) is not divisible by 4"
        else:
            assert new_h % 2 == 0 and new_w % 4 == 0, f"Padded size not correct for 8-split"
        
        file_name = Path(image_file).stem
        
        # 분할 크기 계산
        if split_type == 16:
            h_split, w_split = new_h // 4, new_w // 4
            splits_h, splits_w = 4, 4
        else:
            h_split, w_split = new_h // 2, new_w // 4
            splits_h, splits_w = 2, 4
        
        # 이미지 분할 및 저장
        idx = 1
        for i in range(splits_h):
            for j in range(splits_w):
                y_start = i * h_split
                y_end = (i + 1) * h_split
                x_start = j * w_split
                x_end = (j + 1) * w_split
                
                split = img[y_start:y_end, x_start:x_end]
                new_file_name = f"{file_name}_{idx}.png"
                cv2.imwrite(os.path.join(output_dir, new_file_name), split)
                idx += 1
        
        # 패딩 정보 저장 (나중에 복원할 때 사용)
        if pad_h > 0 or pad_w > 0:
            padding_info = {
                'original_size': [h, w],
                'padding': [pad_h, pad_w]
            }
            with open(os.path.join(output_dir, f"{file_name}_padding.txt"), 'w') as f:
                f.write(str(padding_info))

def verify_split_images(input_dir, output_dir, split_type=16):
    """패딩 적용된 분할 이미지 검증"""
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
    
    verification_results = []
    
    for image_file in tqdm(image_files, desc="Verifying splits"):
        # 원본 이미지 읽기
        original_img = cv2.imread(os.path.join(input_dir, image_file))
        if original_img is None:
            continue
            
        file_name = Path(image_file).stem
        expected_splits = 16 if split_type == 16 else 8
        
        # 분할된 이미지들 확인
        split_files = [f for f in os.listdir(output_dir) 
                      if f.startswith(file_name + "_") and f.endswith(".png")]
        
        issues = []
        
        # 1. 분할 파일 개수 확인
        if len(split_files) != expected_splits:
            issues.append(f"Expected {expected_splits} splits, found {len(split_files)}")
        
        # 2. 각 분할 이미지 크기 확인
        split_sizes = []
        for split_file in split_files:
            split_img = cv2.imread(os.path.join(output_dir, split_file))
            if split_img is not None:
                split_sizes.append(split_img.shape[:2])
        
        # 모든 분할 이미지의 크기가 동일한지 확인
        if len(set(split_sizes)) > 1:
            issues.append(f"Inconsistent split sizes: {set(split_sizes)}")
        
        # 3. 패딩 정보 확인
        padding_file = os.path.join(output_dir, f"{file_name}_padding.txt")
        if os.path.exists(padding_file):
            with open(padding_file, 'r') as f:
                padding_info = eval(f.read())
            original_size = padding_info['original_size']
            padding = padding_info['padding']
            
            # 패딩된 크기 계산
            padded_h = original_size[0] + padding[0]
            padded_w = original_size[1] + padding[1]
            
            # 패딩된 크기가 4의 배수인지 확인
            if padded_h % 4 != 0 or padded_w % 4 != 0:
                issues.append(f"Padded size ({padded_w}x{padded_h}) is not divisible by 4")
            
            # 분할 크기가 예상과 일치하는지 확인
            expected_h = padded_h // (4 if split_type == 16 else 2)
            expected_w = padded_w // 4
            
            if split_sizes and split_sizes[0] != (expected_h, expected_w):
                issues.append(f"Split size mismatch: expected {expected_h}x{expected_w}, got {split_sizes[0]}")
        
        if issues:
            verification_results.append({
                'file': image_file,
                'issues': issues
            })
    
    # 검증 결과 출력
    if verification_results:
        print("\nVerification issues found:")
        for result in verification_results:
            print(f"\nFile: {result['file']}")
            print("Issues:")
            for issue in result['issues']:
                print(f"  - {issue}")
        print(f"\nTotal files with issues: {len(verification_results)}")
    else:
        print("\nAll splits verified successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split images into smaller parts with padding')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for split images')
    parser.add_argument('--split_type', type=int, default=16, choices=[8, 16], 
                        help='Number of splits (8 or 16)')
    
    args = parser.parse_args()
    
    print("Starting image processing with checks...")
    split_and_save_images_with_padding(args.input_dir, args.output_dir, split_type=args.split_type)
    print("\nVerifying splits...")
    verify_split_images(args.input_dir, args.output_dir, split_type=args.split_type)
    print("Processing completed!")