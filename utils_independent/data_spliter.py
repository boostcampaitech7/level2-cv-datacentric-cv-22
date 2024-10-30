import os
import json
import shutil

# 입력된 validation용 이미지 파일명 리스트
# ────────────────────────────────────────────────────────────────────────────────────────────────
validation_filenames = set([
    "extractor.zh.in_house.appen_000088_page0001.jpg",
    "extractor.zh.in_house.appen_000118_page0001.jpg",
    "extractor.zh.in_house.appen_000124_page0001.jpg",
    "extractor.zh.in_house.appen_000224_page0001.jpg",
    "extractor.zh.in_house.appen_000314_page0001.jpg",
    "extractor.zh.in_house.appen_000326_page0001.jpg",
    "extractor.zh.in_house.appen_000426_page0001.jpg",
    "extractor.zh.in_house.appen_000825_page0001.jpg",
    "extractor.zh.in_house.appen_000858_page0001.jpg",
    "extractor.zh.in_house.appen2_001094_page0001.jpg",
    
    "extractor.ja.in_house.appen_000029_page0001.jpg",
    "extractor.ja.in_house.appen_000167_page0001.jpg",
    "extractor.ja.in_house.appen_000196_page0001.jpg",
    "extractor.ja.in_house.appen_000200_page0001.jpg",
    "extractor.ja.in_house.appen_000294_page0001.jpg",
    "extractor.ja.in_house.appen_000677_page0001.jpg",
    "extractor.ja.in_house.appen_000812_page0001.jpg",
    "extractor.ja.in_house.appen_000922_page0001.jpg",
    "extractor.ja.in_house.appen_000989_page0001.jpg",
    "extractor.ja.in_house.appen_001017_page0001.jpg",

    "extractor.th.in_house.appen_000056_page0001.jpg",
    "extractor.th.in_house.appen_000090_page0001.jpg",
    "extractor.th.in_house.appen_000102_page0001.jpg",
    "extractor.th.in_house.appen_000110_page0001.jpg",
    "extractor.th.in_house.appen_000540_page0001.jpg",
    "extractor.th.in_house.appen_000607_page0001.jpg",
    "extractor.th.in_house.appen_000630_page0001.jpg",
    "extractor.th.in_house.appen_000693_page0001.jpg",
    "extractor.th.in_house.appen_000742_page0001.jpg",
    "extractor.th.in_house.appen_001016_page0001.jpg",

    "extractor.vi.in_house.appen_000144_page0001.jpg",
    "extractor.vi.in_house.appen_000172_page0001.jpg",
    "extractor.vi.in_house.appen_000297_page0001.jpg",
    "extractor.vi.in_house.appen_000323_page0001.jpg",
    "extractor.vi.in_house.appen_000326_page0001.jpg",
    "extractor.vi.in_house.appen_000357_page0001.jpg",
    "extractor.vi.in_house.appen_000373_page0001.jpg",
    "extractor.vi.in_house.appen_000597_page0001.jpg",
    "extractor.vi.in_house.appen_000614_page0001.jpg",
    "extractor.vi.in_house.appen_000737_page0001.jpg",
    
    "extractor.zh.in_house.appen_000030_page0001.jpg",
    "extractor.zh.in_house.appen_000090_page0001.jpg",
    "extractor.zh.in_house.appen_000185_page0001.jpg",
    "extractor.zh.in_house.appen_000266_page0001.jpg",
    "extractor.zh.in_house.appen_000684_page0001.jpg",
    "extractor.zh.in_house.appen_000705_page0001.jpg",
    "extractor.zh.in_house.appen2_001082_page0001.jpg",
    
    "extractor.ja.in_house.appen_000084_page0001.jpg",
    "extractor.ja.in_house.appen_000242_page0001.jpg",
    "extractor.ja.in_house.appen_000618_page0001.jpg",
    "extractor.ja.in_house.appen_000664_page0001.jpg",
    "extractor.ja.in_house.appen_000817_page0001.jpg",
    "extractor.ja.in_house.appen_000835_page0001.jpg",
    "extractor.ja.in_house.appen_000837_page0001.jpg",
    
    "extractor.th.in_house.appen_000064_page0001.jpg",
    "extractor.th.in_house.appen_000377_page0001.jpg",
    "extractor.th.in_house.appen_000491_page0001.jpg",
    "extractor.th.in_house.appen_000490_page0001.jpg",
    "extractor.th.in_house.appen_000522_page0001.jpg",
    "extractor.th.in_house.appen_000804_page0001.jpg",
    "extractor.th.in_house.appen_001003_page0001.jpg",
    
    "extractor.vi.in_house.appen_000043_page0001.jpg",
    "extractor.vi.in_house.appen_000478_page0001.jpg",
    "extractor.vi.in_house.appen_000769_page0001.jpg",
    "extractor.vi.in_house.appen_000780_page0001.jpg",
    "extractor.vi.in_house.appen_000960_page0001.jpg",
    "extractor.vi.in_house.appen_001047_page0001.jpg",
    "extractor.vi.in_house.appen_001114_page0001.jpg",
    
    "extractor.zh.in_house.appen_000330_page0001.jpg",
    "extractor.zh.in_house.appen_000333_page0001.jpg",
    "extractor.zh.in_house.appen_000991_page0001.jpg",
    
    "extractor.ja.in_house.appen_000316_page0001.jpg",
    "extractor.ja.in_house.appen_000484_page0001.jpg",
    "extractor.ja.in_house.appen_000760_page0001.jpg",
    
    "extractor.th.in_house.appen_000395_page0001.jpg",
    "extractor.th.in_house.appen_000424_page0001.jpg",
    "extractor.th.in_house.appen_000095_page0001.jpg",
    
    "extractor.vi.in_house.appen_000061_page0001.jpg",
    "extractor.vi.in_house.appen_000478_page0001.jpg",
    "extractor.vi.in_house.appen_000658_page0001.jpg"
])



# 데이터셋 경로 설정 및 결과 디렉토리 생성
# ────────────────────────────────────────────────────────────────────────────────────────────────
root = '/data/ephemeral/home'
data_dir = root + '/code/data_original'

output_dir = root + '/code/data'
os.makedirs(output_dir, exist_ok=True)

languages = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt']

# ────────────────────────────────────────────────────────────────────────────────────────────────

train_data = {}
val_data = {}

for lang in languages:
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


    for image_id, image_info in data['images'].items():
        image_filename = image_id
        src_image_path = os.path.join(img_dir, image_filename)
        
        if image_filename in validation_filenames:
            # Validation 데이터로 분류
            print(f"Validation으로 분류: {image_filename}")
            val_data[lang]['images'][image_filename] = image_info
            dst_image_path = os.path.join(val_img_dir, image_filename)
        else:
            # Train 데이터로 분류
            train_data[lang]['images'][image_filename] = image_info
            dst_image_path = os.path.join(train_img_dir, image_filename)

        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)


for lang in languages:
    val_json_path = os.path.join(output_dir, lang, 'ufo', 'validation.json')
    train_json_path = os.path.join(output_dir, lang, 'ufo', 'train.json')

    os.makedirs(os.path.dirname(val_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_json_path), exist_ok=True)

    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data[lang], f, indent=4, ensure_ascii=False)
    
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_data[lang], f, indent=4, ensure_ascii=False)


print("데이터셋 분할 완료.")
