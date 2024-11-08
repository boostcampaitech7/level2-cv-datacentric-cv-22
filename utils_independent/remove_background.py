import os
import cv2
import json
import numpy as np


def apply_background_mask(image_path, points, output_path):
    image = cv2.imread(image_path)

    # 영수증 영역 구하기
    min_x = int(min([point[0] for word in points for point in word]))
    max_x = int(max([point[0] for word in points for point in word]))
    min_y = int(min([point[1] for word in points for point in word]))
    max_y = int(max([point[1] for word in points for point in word]))

    # 영수증 영역 제외 하얀색으로 masking
    mask = np.ones_like(image) * 255
    mask[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]

    cv2.imwrite(output_path, mask)

def process_recipt_images(input_folder, json_path, output_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for image_name, details in data['images'].items():
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)
        
        points = [word_data['points'] for word_data in details['words'].values()]
        
        if points:
            apply_background_mask(image_path, points, output_path)


def main():
    root = 'code/data/'
    languages = ['chinese', 'japanese', 'thai', 'vietnamese']

    for language in languages:
        input_folder = f'{root}{language}_receipt/img/train'
        json_path = f'{root}{language}_receipt/ufo/train.json'
        output_folder = f'remove_background_data/{language}_reciept/img/train'
        os.makedirs(output_folder, exist_ok=True)
        process_recipt_images(input_folder, json_path, output_folder)
    print("Background removal using bounding boxes completed.")

if __name__ == '__main__':
    main()