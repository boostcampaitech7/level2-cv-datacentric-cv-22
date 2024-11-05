import cv2
import json
import numpy as np
import os


def remove_background_and_binarize(image_path, points, output_path):
    # 이미지 grayscale로 변환
    image = cv2.imread(image_path)

    # 영수증 영역 구하기
    for word in points:
        for point in word:
            if point[0] < 0 or point[1] < 0:
                print(image_path, point)
    min_x = int(min([point[0] for word in points for point in word]))
    max_x = int(max([point[0] for word in points for point in word]))
    min_y = int(min([point[1] for word in points for point in word]))
    max_y = int(max([point[1] for word in points for point in word]))

    # 영수증 영역 제외 하얀색으로 masking
    mask = np.ones_like(image) * 255
    mask[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]

    # Save the processed image
    cv2.imwrite(output_path, mask)

def preprocess(input_folder, json_path, output_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for image_name, details in data['images'].items():
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)
        
        points = [word_data['points'] for word_data in details['words'].values()]
        
        if points:
            remove_background_and_binarize(image_path, points, output_path)

def main():
    languages = {'chinese' : 'zh',
                'japanese' : 'ja',
                'thai' : 'th',
                'vietnamese' : 'vi'}
    root = './code/data_original/'
    
    for key, value in languages.items():
        input_folder = root + key + '_receipt/img/train'
        json_path = root + key + '_receipt/ufo/' + 'train copy.json'
        output_folder = 'remove_background_original/' + key + '_reciept/img/train'
        os.makedirs(output_folder, exist_ok=True)
        preprocess(input_folder, json_path, output_folder)
    print("Background removal using bounding boxes completed.")

main()