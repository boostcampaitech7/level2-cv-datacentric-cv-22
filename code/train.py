import os
import os.path as osp
import time
import math
import wandb
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
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    
    wandb.init(project="Data-Centric", entity='jhs7027-naver', name='hyungjoon',config={
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "image_size": image_size,
        "input_size": input_size,
        "num_workers": num_workers,
    })

    # 함수 시작 부분에 현재 시간으로 폴더명 생성
    current_time = time.strftime("%y%m%d_%H%M")
    save_dir = osp.join(model_dir, current_time)
    
    train_dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    train_dataset = EASTDataset(train_dataset)
    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataset = SceneTextDataset(
        data_dir,
        split='validation',
        image_size=image_size,
        crop_size=input_size,
    )
    val_dataset = EASTDataset(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    milestones = [max_epoch // 2]  
    gamma = 0.1
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    wandb.log({
        "optimizer": optimizer.__class__.__name__,  
        "initial_learning_rate": learning_rate,
        "scheduler": scheduler.__class__.__name__,  
        "milestones": milestones,  
        "gamma": gamma  
    })

    model.train()
    for epoch in range(max_epoch):
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

                wandb.log(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        wandb.log({"mean_train_loss": epoch_loss / num_batches})
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                img, gt_score_map, gt_geo_map, roi_mask = (
                    img.to(device), gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device)
                )
                
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                val_loss += loss.item()

                v_val_dict = {
                            'v_Cls loss': extra_info['cls_loss'], 'v_Angle loss': extra_info['angle_loss'],
                            'v_IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(v_val_dict)
                pbar.update(1)  

                wandb.log(v_val_dict)

        avg_val_loss = val_loss / len(val_loader)
        print('Mean Validation Loss: {:.4f} | Elapsed time: {}'.format(
            avg_val_loss, timedelta(seconds=time.time() - epoch_start)))

        wandb.log({"mean_validation_loss": avg_val_loss})

        model.train()

        
        # 모델 저장 부분만 수정
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(save_dir):
                os.makedirs(save_dir)

            ckpt_fpath = osp.join(save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print(f'Model checkpoint saved at {ckpt_fpath}')


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


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)