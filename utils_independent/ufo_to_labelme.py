import json
import os
import shutil


def ufo_to_labelme(ufo_json_path, image_dir, output_dir):
    with open(ufo_json_path, 'r', encoding='utf-8') as f:
        ufo_data = json.load(f)

    # UFO 포맷을 LabelMe 포맷으로 전환
    for image_filename, image_data in ufo_data['images'].items():
        labelme_data = {
            "version": "5.0.3",
            "flags": {},
            "shapes": [],
            "imagePath": image_filename,
            "imageData": None,
            "imageHeight": image_data["img_h"],
            "imageWidth": image_data["img_w"]
        }

        for word_id, word_data in image_data["words"].items():
            transcription = word_data["transcription"]
            points = word_data["points"]

            # transcription이 null인 경우 처리
            if transcription is None:
                transcription = "unknown"

            # LabelMe 형식으로 shape 생성
            shape = {
                "label": transcription,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_data["shapes"].append(shape)
        
        labelme_json_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.json')
        with open(labelme_json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=4)
        
        source_image_path = os.path.join(image_dir, image_filename)
        target_image_path = os.path.join(output_dir, image_filename)
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, target_image_path)

    print("Conversion completed. LabelMe JSON files and images saved in:", output_dir)

    
def main():
    languages = ['chinese', 'japanese', 'thai', 'vietnamese']

    for language in languages:
        ufo_file_path = f'code/data/{language}_receipt/ufo/train.json'
        image_dir = f'code/data/{language}_receipt/img/train' 
        output_dir = f'labelme_jsons/{language}_receipt'
        os.makedirs(output_dir, exist_ok=True)
        ufo_to_labelme(ufo_file_path, image_dir, output_dir)

if __name__ == '__main__':
    main()