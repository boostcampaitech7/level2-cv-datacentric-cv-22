import os
import json

def ufo_to_labelme(ufo_json_path, output_dir):
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
            
            # transcription = null인 경우 처리
            if transcription == None:
                transcription = "unknown"

            # LabelMe의 shape
            shape = {
                "label": transcription,
                "points": points,
                "group_id": None,
                "shape_type": "polygon", # rectangular가 아닌 polygon으로 설정
                "flags": {}
            }
            labelme_data["shapes"].append(shape)

        # LabelMe JSON으로 저장
        labelme_json_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.json')
        with open(labelme_json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=4)

    print("Conversion completed. LabelMe JSON files saved in:", output_dir)


def main():
    languages = ['chinese', 'japanese', 'thai', 'vietnamese']

    for language in languages:
        ufo_file_path = f'code/data/{language}_receipt/ufo/train.json'
        output_dir = f'labelme_jsons/{language}_receipt'
        os.makedirs(output_dir, exist_ok=True)
    ufo_to_labelme(ufo_file_path, output_dir)

if __name__ == '__main__':
    main()