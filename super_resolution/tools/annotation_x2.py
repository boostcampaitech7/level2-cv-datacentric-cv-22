import json
import sys

def double_coordinates(json_data):
    # images 딕셔너리의 각 이미지에 대해
    for image_data in json_data["images"].values():
        # words 딕셔너리의 각 단어에 대해
        for word in image_data["words"].values():
            # points 배열의 각 좌표에 대해
            for i in range(len(word["points"])):
                # x, y 좌표 각각 2배
                word["points"][i] = [
                    word["points"][i][0] * 2,
                    word["points"][i][1] * 2
                ]
        
        # img_w와 img_h도 2배로 변경
        image_data["img_w"] *= 2
        image_data["img_h"] *= 2

def process_file(input_path, output_path):
    try:
        # JSON 파일 읽기
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 좌표 2배로 변경
        double_coordinates(data)
        
        # 결과를 새 JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully processed {input_path} -> {output_path}")
        return True
    
    except Exception as e:
        print(f"Error processing file: {str(e)}", file=sys.stderr)
        return False

def main():
    # 파일 경로 직접 지정
    input_file = "/data/ephemeral/home/repo/code/data/chinese_receipt/ufo/train.json"
    output_file = "/data/ephemeral/home/repo/code/data/chinese_receipt/ufo/train_doubled.json"
    
    success = process_file(input_file, output_file)
    if success:
        print("Coordinate doubling completed successfully")
    else:
        print("Failed to process the file")

if __name__ == "__main__":
    main()