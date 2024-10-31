import os
import json
import shutil
import random


def stratified_split_bbox_count(data, ratio=0.1):
    """ bbox 개수에 따라 데이터를 구간별로 나누고 일정 비율로 샘플링 """
    low, medium, high = [], [], []
    
    for image_id, image_info in data.items():
        bbox_count = len(image_info['words'])
        if bbox_count < 10:
            low.append((image_id, image_info))
        elif bbox_count < 30:
            medium.append((image_id, image_info))
        else:
            high.append((image_id, image_info))

    val_data = random.sample(low, int(ratio * len(low))) + \
               random.sample(medium, int(ratio * len(medium))) + \
               random.sample(high, int(ratio * len(high)))

    # 나머지는 train으로 분류
    val_ids = set([img_id for img_id, _ in val_data])
    train_data = [(img_id, info) for img_id, info in data.items() if img_id not in val_ids]
    
    return train_data, val_data


# 경로 설정
# ────────────────────────────────────────────────────────────────────────────────────────────────



def save_stratified_split(data_dir, output_dir, languages, validation_ratio=0.1):
    """
    Bounding box 개수를 기준으로 stratified split을 수행하여 데이터를 분할하고 저장

    Args:
        data_dir (str): 원본 데이터 디렉토리 경로
        output_dir (str): 저장될 데이터 디렉토리 경로
        languages (list): 분할할 언어 리스트
        validation_ratio (float): validation 데이터의 비율

    """

    os.makedirs(output_dir, exist_ok=True)
    
    train_data = {}
    val_data = {}

    for lang in languages:

        # 초기화
        json_path = os.path.join(data_dir, lang, 'ufo', 'train.json')
        img_dir = os.path.join(data_dir, lang, 'img', 'train')

        train_img_dir = os.path.join(output_dir, lang, 'img', 'train')
        val_img_dir = os.path.join(output_dir, lang, 'img', 'validation')
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)

        train_data[lang] = {"images": {}}
        val_data[lang] = {"images": {}}


        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Bounding box count를 기반으로 stratified split
        train_split, val_split = stratified_split_bbox_count(data['images'], ratio=validation_ratio)

        # Train set 데이터 복사 및 JSON 생성
        for image_id, image_info in train_split:
            train_data[lang]['images'][image_id] = image_info

            src_image_path = os.path.join(img_dir, image_id)
            dst_image_path = os.path.join(train_img_dir, image_id)

            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)

        # Validation set 데이터 복사 및 JSON 생성
        for image_id, image_info in val_split:
            val_data[lang]['images'][image_id] = image_info

            src_image_path = os.path.join(img_dir, image_id)
            dst_image_path = os.path.join(val_img_dir, image_id)

            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)

        
        # 데이터 저장
        val_json_path = os.path.join(output_dir, lang, 'ufo', 'validation.json')
        train_json_path = os.path.join(output_dir, lang, 'ufo', 'train.json')

        os.makedirs(os.path.dirname(val_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(train_json_path), exist_ok=True)

        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_data[lang], f, indent=4, ensure_ascii=False)
        
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_data[lang], f, indent=4, ensure_ascii=False)

    print("데이터셋 분할 완료.")


data_dir = '/data/ephemeral/home/code/data_original'
output_dir = './code/data_stratified'
languages = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt']
validation_ratio = 0.1

save_stratified_split(data_dir, output_dir, languages, validation_ratio)
