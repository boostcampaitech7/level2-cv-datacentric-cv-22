import json

def filter_and_count_bboxes(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    removed_count = 0
    for image in data['images'].values():
        filtered_words = {}
        for word_id, word in image['words'].items():
            if 'points' in word:
                points = word['points']
                width = points[1][0] - points[0][0]
                if width > 4:
                    filtered_words[word_id] = word
                else:
                    removed_count += 1
            else:
                filtered_words[word_id] = word
        image['words'] = filtered_words
    
    new_file_path = file_path.replace('train.json', 'train_remove_4.json')
    with open(new_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return removed_count

# List of file paths to process
file_paths = [
    '/data/ephemeral/home/repo/code/data/vietnamese_receipt/ufo/train.json',
    '/data/ephemeral/home/repo/code/data/thai_receipt/ufo/train.json',
    '/data/ephemeral/home/repo/code/data/japanese_receipt/ufo/train.json',
    '/data/ephemeral/home/repo/code/data/chinese_receipt/ufo/train.json'
]

total_removed = 0
for file_path in file_paths:
    total_removed += filter_and_count_bboxes(file_path)

print(f"Total number of bboxes removed: {total_removed}")