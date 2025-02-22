import os
import json
from typing import List, Dict

def load_data(root_dir: str, languages: List[str], split: str, word: str = "") -> Dict[str, Dict[str, List[Dict]]]:
    """
        특정 transcriptions에 대한 데이터만 불러오는 함수
    """
    
    data_dict = {}

    for lang in languages:
        json_path = os.path.join(root_dir, f"{lang}_receipt/ufo/{split}.json")
        images_dir = os.path.join(root_dir, f"{lang}_receipt/img/{split}")
        
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 각 이미지에서 입력한 transcription을 가진 단어 좌표 정보 수집
        for image_name, image_data in data['images'].items():
            words = image_data.get('words', {})
            empty_transcriptions = []

            for word_id, word_data in words.items():
                
                if word in word_data['transcription']:  # 포함하는
                #if word_data['transcription'] == word:   # 일치하는
                    points = word_data['points']
                    
                    
                    empty_transcriptions.append({
                        'word_id': word_id,
                        'points': points,
                        'image_path': os.path.join(images_dir, image_name),
                        'lang': lang,
                        'split': split,
                        'image_name': image_name
                    })

            # 해당하는 transcription 항목이 있을 경우 data_dict에 저장
            if empty_transcriptions:
                if lang not in data_dict:
                    data_dict[lang] = {}
                if split not in data_dict[lang]:
                    data_dict[lang][split] = []
                data_dict[lang][split].append(empty_transcriptions)

    return data_dict