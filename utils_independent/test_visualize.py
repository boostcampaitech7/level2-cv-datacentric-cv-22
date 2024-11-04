import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path

def parse_args():
    """Command line arguments를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Visualize test results with bounding boxes')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Name of the CSV file (without .csv extension)')
    return parser.parse_args()

def load_ufo_format(file_path):
    """UFO 포맷의 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_language_from_filename(filename):
    """파일명에서 언어 코드를 추출합니다."""
    try:
        # extractor.zh.in_house... 형식에서 언어 코드 추출
        parts = filename.split('.')
        lang_code = parts[1]  # 두 번째 부분이 언어 코드
        
        # 언어 코드를 전체 언어명으로 매핑
        language_map = {
            'zh': 'chinese',
            'ja': 'japanese',
            'th': 'thai',
            'vi': 'vietnamese'
        }
        return language_map.get(lang_code, 'unknown')
    except:
        return 'unknown'

def organize_by_language(annotations):
    """어노테이션을 언어별로 분류합니다."""
    language_files = {
        'chinese': [],
        'japanese': [],
        'thai': [],
        'vietnamese': [],
        'unknown': []
    }
    
    for image_name in annotations['images'].keys():
        lang = get_language_from_filename(image_name)
        language_files[lang].append(image_name)
    
    return language_files

def draw_boxes(image_path, annotations, output_path):
    """이미지에 바운딩 박스를 그립니다."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    for word_info in annotations['words'].values():
        points = np.array(word_info['points'], dtype=np.int32)
        
        # 바운딩 박스 그리기
        cv2.polylines(img, [points], True, (0, 255, 0), 2)
    
    # 결과 이미지 저장
    cv2.imwrite(output_path, img)

def main():
    # Parse command line arguments
    args = parse_args()
    # UFO 포맷 파일 경로
    ufo_path = f'../code/predictions/{args.csv_name}.csv'
    
    # 테스트 이미지 디렉토리들
    test_dirs = {
        'chinese': '../code/data/chinese_receipt/img/test',
        'japanese': '../code/data/japanese_receipt/img/test',
        'thai': '../code/data/thai_receipt/img/test',
        'vietnamese': '../code/data/vietnamese_receipt/img/test'
    }
    
    # 결과 저장 디렉토리 (CSV 파일 이름으로 생성)
    output_base_dir = os.path.join('../visualized_result', args.csv_name)
    
    print(f'\nStarting visualization for {args.csv_name}')
    print(f'Loading annotations from: {ufo_path}')
    print(f'Results will be saved to: {output_base_dir}')
    
    # UFO 포맷 데이터 로드
    annotations = load_ufo_format(ufo_path)
    
    # 언어별로 파일 분류
    language_files = organize_by_language(annotations)
    
    # 언어별 통계 출력
    print("\nFiles count by language:")
    for lang, files in language_files.items():
        print(f"{lang}: {len(files)} files")
    
    # 각 언어별로 처리
    for lang, files in language_files.items():
        if lang == 'unknown':
            print("\nUnknown language files:")
            for f in files:
                print(f"  - {f}")
            continue
            
        print(f"\nProcessing {lang} files...")
        
        # 해당 언어의 출력 디렉토리 생성
        output_dir = os.path.join(output_base_dir, lang)
        os.makedirs(output_dir, exist_ok=True)
        
        # 각 이미지 처리
        for image_name in files:
            # 원본 이미지 찾기
            image_found = False
            for test_dir in [test_dirs[lang]]:  # 해당 언어의 테스트 디렉토리에서만 검색
                image_path = os.path.join(test_dir, image_name)
                if os.path.exists(image_path):
                    image_found = True
                    break
            
            if not image_found:
                print(f"Image not found: {image_name}")
                continue
            
            # 결과 이미지 경로
            output_path = os.path.join(output_dir, f'visualized_{image_name}')
            
            # 바운딩 박스 그리기
            draw_boxes(image_path, annotations['images'][image_name], output_path)
            print(f'Processed: {image_name}')

if __name__ == '__main__':
    main()