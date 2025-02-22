import os
import json
from glob import glob

def labelme_to_ufo(labelme_dir, output_path):
    ufo_data = {"images": {}}

    # LabelMe JSON 파일 읽기
    for labelme_json_path in glob(os.path.join(labelme_dir, '*.json')):
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        
        # 이미지 파일 이름
        image_filename = labelme_data["imagePath"]
        
        # UFO 포맷 이미지 데이터 초기화
        ufo_data["images"][image_filename] = {
            "paragraphs": {},  # 빈 paragraphs 필드 추가
            "words": {},
            "chars": {},  # 빈 chars 필드 추가
            "img_w": labelme_data.get("imageWidth", 0),
            "img_h": labelme_data.get("imageHeight", 0),
            "num_patches": None,
            "tags": [],
            "relations": {},
            "annotation_log": {
                "worker": "worker",  # 필요에 따라 수정 가능
                "timestamp": "",     # 필요에 따라 수정 가능
                "tool_version": "",
                "source": None
            },
            "license_tag": {
                "usability": True,
                "public": False,
                "commercial": True,
                "type": None,
                "holder": "Upstage"  # 필요에 따라 수정 가능
            }
        }

        # 바운딩 박스 정보 변환
        for idx, shape in enumerate(labelme_data["shapes"]):
            transcription = shape.get("label", "unknown")  # transcription이 없으면 "unknown"
            points = shape.get("points", [])
            
            # 좌표의 정밀도 유지
            points = [[round(p[0], 6), round(p[1], 6)] for p in points]

            # UFO 포맷 단어 데이터 추가 - ID를 "0001", "0002" 형식으로 설정
            word_id = str(idx + 1).zfill(4)
            ufo_data["images"][image_filename]["words"][word_id] = {
                "transcription": transcription,
                "points": points
            }

    # UFO 포맷 JSON으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_data, f, ensure_ascii=False, indent=4)

    print("Conversion completed. UFO JSON file saved at:", output_path)

# LabelMe JSON 폴더 경로와 UFO 형식 결과 저장 경로 설정
languages = ['chinese', 'japanese', 'thai', 'vietnamese']

for language in languages:
    labelme_dir = f'labelme_jsons/{language}_receipt'
    output_path = f'../code/data/{language}_receipt/ufo/train_new.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labelme_to_ufo(labelme_dir, output_path)
