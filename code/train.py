import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--checkpoint_path', type=str, default=None, help="학습 재개 시 체크포인트 파일 경로 지정을 위한 인자")
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, checkpoint_path=None):
    
    # 데이터 초기화 ──────────────────────────────────────────────────────────────────────────────
    
    train_dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size,)
    train_dataset = EASTDataset(train_dataset)

    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = SceneTextDataset(data_dir, split='validation', image_size=image_size, crop_size=input_size,)
    val_dataset = EASTDataset(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    # 체크 포인트 ────────────────────────────────────────────────────────────────────────────────

    start_epoch = 0
    
    if checkpoint_path and osp.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']

    model.train()

    for epoch in range(start_epoch, max_epoch):
        # 학습 ───────────────────────────────────────────────────────────────────────────────────
        epoch_loss, epoch_start = 0, time.time()

        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                img, gt_score_map, gt_geo_map, roi_mask = (
                    img.to(device), gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device)
                )
                
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        
        # 검증 ───────────────────────────────────────────────────────────────────────────────────

        val_loss = 0

        model.eval()
        with torch.no_grad():
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                img, gt_score_map, gt_geo_map, roi_mask = (
                    img.to(device), gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device)
                )

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print('Epoch {} Validation Loss: {:.4f}'.format(epoch + 1, avg_val_loss))

        model.train()

        
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
            }, ckpt_fpath)
            
            print(f'Model checkpoint saved at {ckpt_fpath}')



def main(args):
    # ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ 체크포인트 쓸 때 지정해야 할 것 ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
    args.checkpoint_path = '/data/ephemeral/home/repo/code/trained_models/epoch_35.pth'

    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)



# 원래 train 코드
# def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
#                 learning_rate, max_epoch, save_interval):
#     dataset = SceneTextDataset(
#         data_dir,
#         split='train',
#         image_size=image_size,
#         crop_size=input_size,
#     )
#     dataset = EASTDataset(dataset)
#     num_batches = math.ceil(len(dataset) / batch_size)
#     train_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers
#     )

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = EAST()
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

#     model.train()
#     for epoch in range(max_epoch):
#         epoch_loss, epoch_start = 0, time.time()
#         with tqdm(total=num_batches) as pbar:
#             for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
#                 pbar.set_description('[Epoch {}]'.format(epoch + 1))

#                 loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 loss_val = loss.item()
#                 epoch_loss += loss_val

#                 pbar.update(1)
#                 val_dict = {
#                     'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
#                     'IoU loss': extra_info['iou_loss']
#                 }
#                 pbar.set_postfix(val_dict)

#         scheduler.step()

#         print('Mean loss: {:.4f} | Elapsed time: {}'.format(
#             epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

#         if (epoch + 1) % save_interval == 0:
#             if not osp.exists(model_dir):
#                 os.makedirs(model_dir)

#             ckpt_fpath = osp.join(model_dir, 'latest.pth')
#             torch.save(model.state_dict(), ckpt_fpath)
