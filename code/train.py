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

# 추가한 라이브러리
import wandb
import random
import numpy as np
import random
from detect import get_bboxes
from validate_bbox import ensure_bbox_format, extract_true_bboxes
from deteval import calc_deteval_metrics

def set_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data')) # 전체 학습 데이터 경로 지정
    parser.add_argument('--data_val_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_VAL', 'data_val')) # 검증 데이터 경로 지정
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--checkpoint_path', type=str, default=None, help="학습 재개 시 체크포인트 파일 경로 지정을 위한 인자")
    parser.add_argument('--validate', type=bool, default=False, help="Validation 실행 여부") # True/False 사용하여 검증 실행 여부 결정
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args 


def do_training(data_dir, data_val_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, checkpoint_path=None, validate=False):

    current_time = time.strftime('%Y%m%d_%H%M')
    save_dir = osp.join(model_dir, current_time)
    
    set_seed()

    # wandb 초기화 ─────────────────────────────────────────────────────────────────────────────
    wandb.init(project="Data-Centric", entity='jhs7027-naver', group = 'hyungjoon', name='hyungjoon',config={
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "image_size": image_size,
        "input_size": input_size,
        "num_workers": num_workers,
    })
    
    # 데이터 초기화 ──────────────────────────────────────────────────────────────────────────────
    
    train_dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size,)
    train_dataset = EASTDataset(train_dataset)

    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if validate:
        val_dataset = SceneTextDataset(data_val_dir, split='validation', image_size=image_size, crop_size=input_size,)
        val_dataset = EASTDataset(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)

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
                    't_Cls loss': extra_info['cls_loss'], 't_Angle loss': extra_info['angle_loss'],
                    't_IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                
                wandb.log(val_dict) 

        wandb.log({"epoch": epoch + 1})  


        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        wandb.log({"mean_train_loss": epoch_loss / num_batches})
        
        # 검증 ───────────────────────────────────────────────────────────────────────────────────
        if validate and (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad(), tqdm(total=len(val_loader), desc=f"[Epoch {epoch + 1}] Validation", position=0) as val_pbar:
                val_loss = 0
                pred_bboxes_dict, gt_bboxes_dict = {}, {}

                for batch_idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(val_loader):
                    img, gt_score_map, gt_geo_map, roi_mask = (
                        img.to(device), gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device)
                    )
                    
                    pred_score_map, pred_geo_map = model(img)

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    val_loss += loss.item()
                    
                    v_val_dict = {
                                'v_Cls loss': extra_info['cls_loss'], 'v_Angle loss': extra_info['angle_loss'],
                                'v_IoU loss': extra_info['iou_loss']
                    }
                    
                    wandb.log(v_val_dict)

                    for i in range(img.size(0)):
                        score = pred_score_map[i].cpu().numpy()
                        geo = pred_geo_map[i].cpu().numpy()

                        gt_bboxes = extract_true_bboxes(gt_score_map[i].cpu().numpy(), gt_geo_map[i].cpu().numpy())
                        if gt_bboxes is not None and gt_bboxes.size > 0:
                            gt_bboxes = ensure_bbox_format(gt_bboxes)

                        pred_bboxes = get_bboxes(score, geo)
                        if pred_bboxes is not None and pred_bboxes.size > 0:
                            pred_bboxes = ensure_bbox_format(pred_bboxes)

                        img_id = f"batch_{batch_idx}_img_{i}"

                        if pred_bboxes is not None and pred_bboxes.size > 0:
                            pred_bboxes_dict[img_id] = pred_bboxes

                        if gt_bboxes is not None and gt_bboxes.size > 0:
                            gt_bboxes_dict[img_id] = gt_bboxes

                        if pred_bboxes is None: 
                            continue

                        val_pbar.update(1)
                
                # pred_bboxes가 None인지 확인
                if pred_bboxes is not None:
                    print("\npred_bboxes shape:", pred_bboxes.shape)
                else:
                    print("\npred_bboxes is None")
                
                if gt_bboxes is not None:
                    print("gt_bboxes shape:", gt_bboxes.shape)
                else:
                    print("gt_bboxes is None")
                    

                metrics_result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict,)
                avg_f1_score = metrics_result['total']['hmean']
                avg_precision = metrics_result['total']['precision']
                avg_recall = metrics_result['total']['recall']

            
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1_score:.4f}')
            
            wandb.log({"val_loss": avg_val_loss, "precision": avg_precision, "recall": avg_recall, "f1_score": avg_f1_score})
            
        model.train()

        
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(save_dir):
                os.makedirs(save_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
          
            if validate:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_val_loss
                }, ckpt_fpath)
            else:   # validate가 False일 경우 model_state_dict만 저장
                torch.save(model.state_dict(), ckpt_fpath)
                
            print(f'Model checkpoint saved at {ckpt_fpath}')

def main(args):
    # ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ 체크포인트 쓸 때 지정해야 할 것 ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
    #args.checkpoint_path = '/data/ephemeral/home/repo/code/trained_models/epoch_40.pth'

    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)