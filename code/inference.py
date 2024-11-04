import os
import os.path as osp
import json
import subprocess
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect

CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data_fixed_bbox'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    # 체크포인트 폴더명과 에포크 번호를 입력받는 인자 추가
    parser.add_argument('--checkpoint_folder', type=str, default='trained_models',
                       help='Name of the checkpoint folder (e.g., 20240321_1430)')
    parser.add_argument('--epoch_num', type=int, default=85,
                       help='Epoch number of the checkpoint to load')
    parser.add_argument('--output_name', type=str, default='output',
                       help='Name of the experiment for the output file')
    
    # visualization 옵션 추가
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize detection results')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


# use_val : val 사용 여부에 따라 체크포인트 로드 방식 결정
def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test', use_val=False, model_dir='trained_models', ouput_dir='predictions'):
    
    if use_val:
        # val 사용 했을 경우
        checkpoint = torch.load(ckpt_fpath, map_location="cuda")
        model_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model.state_dict()}
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # val 사용 안 한 경우
        model.load_state_dict(torch.load(ckpt_fpath, map_location='cuda'), strict=False)

    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):

    model = EAST(pretrained=False).to(args.device)
    ckpt_fpath = osp.join(args.checkpoint_folder, f'epoch_{args.epoch_num}.pth')

    if not osp.exists(ckpt_fpath):
        raise FileNotFoundError(f'Checkpoint not found at: {ckpt_fpath}')
    
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    # output 파일 이름 수정
    output_fname = f'{args.output_name}.csv'
    
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)

    # Visualization 실행
    if args.visualize:
        print('\nStarting visualization...')
        
        visualize_script = '../utils_independent/test_visualize.py'
        subprocess.run(['python', visualize_script, '--csv_name', args.output_name])


if __name__ == '__main__':
    args = parse_args()
    main(args)