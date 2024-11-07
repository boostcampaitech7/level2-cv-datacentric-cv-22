import os
from PIL import Image


def convert_rgb(input_folder, output_folder):
    image_list = os.listdir(input_folder)

    for image in image_list:
        image_path = os.path.join(input_folder, image)
        output_path = os.path.join(output_folder, image)
        img = Image.open(image_path)

        # EXIF가 존재하는지 확인
        # Orientation에 따라 rotation 적용
        exif = img._getexif()
        if exif:
            orientation = exif.get(274)  # EXIF Orientation tag
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

        rgb_img = img.convert('RGB')

        rgb_img.save(output_path)

def main():
    root = 'code/data/'
    languages = ['chinese', 'japanese', 'thai', 'vietnamese']

    for language in languages:
        input_folder = f'{root}{language}_receipt/img/train'
        output_folder = f'rgb_data/{language}_receipt/img/train'
        os.makedirs(output_folder, exist_ok=True)
        convert_rgb(input_folder, output_folder)
    print("Convert image to RGB mode completed.")

main()