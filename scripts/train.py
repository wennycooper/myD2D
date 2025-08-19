#!/usr/bin/env python3
"""
Training Script for Hybrid Anomaly Detection
結合D2D與AnomalyCLIP learnable prompts的訓練腳本
"""

import ipdb
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# 添加專案路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.hybrid_model import HybridAnomalyDetector
from losses.hybrid_loss import CrossAttentionLoss
from config.base_config import HybridConfig, get_default_config, get_debug_config
from data.dataset import Dataset
from utils.transforms import get_transform

# Fallback logger since we don't have the original D2D logger
try:
    from logger import get_logger
except ImportError:
    print("Warning: D2D logger not available, using fallback implementation")
    
    # Fallback logger
    import logging
    def get_logger(save_path):
        os.makedirs(save_path, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(save_path, 'train.log')),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    # Fallback transform
    def get_transform(args):
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((args.data.resolution, args.data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.data.normalize_mean, std=args.data.normalize_std)
        ])
        target_transform = transforms.Compose([
            transforms.Resize((args.data.resolution, args.data.resolution)),
            transforms.ToTensor()
        ])
        return preprocess, target_transform


def setup_seed(seed: int):
    """設定隨機種子確保可重現性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HybridTrainer:
    """混合異常檢測訓練器"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # 設定隨機種子
        setup_seed(config.system.seed)
        
        # 初始化logger
        self.logger = get_logger(config.system.save_path)
        self.logger.info("Initializing Hybrid Anomaly Detection Trainer...")
        
        # 初始化模型
        self.model = self._setup_model()
        
        # 初始化loss function
        self.loss_fn = CrossAttentionLoss(config.training.loss_weights)
        
        # 初始化optimizer
        self.optimizer = self._setup_optimizer()
        
        # 初始化學習率調度器
        self.scheduler = self._setup_scheduler()
        
        # 訓練狀態
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 保存配置
        config.save(os.path.join(config.system.save_path, 'config.json'))
        self.logger.info(f"Configuration saved to {config.system.save_path}/config.json")
        
    def _setup_model(self) -> HybridAnomalyDetector:
        """初始化模型"""
        model_config = {
            'pretrained_model_name_or_path': self.config.model.pretrained_model_name_or_path,
            'n_ctx': self.config.model.n_ctx,
            'depth': self.config.model.depth,
            't_n_ctx': self.config.model.t_n_ctx,
            'use_fp16': self.config.model.use_fp16
        }
        
        model = HybridAnomalyDetector(model_config).to(self.device)
        
        # 載入checkpoint (如果提供)
        if self.config.system.checkpoint_path:
            checkpoint = model.load_checkpoint(self.config.system.checkpoint_path)
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('loss', float('inf'))
            
        # 打印模型資訊
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
        
        self.logger.info(f"Model initialized:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
        
        return model
    
    def _setup_optimizer(self):
        """初始化優化器"""
        if self.config.training.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.get_trainable_parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.get_trainable_parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
            
        self.logger.info(f"Optimizer: {self.config.training.optimizer}, LR: {self.config.training.learning_rate}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """初始化學習率調度器"""
        if not self.config.training.use_scheduler:
            return None
            
        if self.config.training.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.epochs // 3,
                gamma=0.5
            )
        else:
            scheduler = None
            
        if scheduler:
            self.logger.info(f"Scheduler: {self.config.training.scheduler_type}")
            
        return scheduler
    
    def _setup_dataloader(self) -> DataLoader:
        """設定資料載入器"""
        try:
            # 使用原專案的transform和dataset
            preprocess, target_transform = get_transform(self.config)
            
            train_dataset = Dataset(
                root=self.config.data.data_path,
                transform=preprocess,
                target_transform=target_transform,
                dataset_name=self.config.data.dataset_name
            )
            
        except Exception as e:
            self.logger.warning(f"Could not load original dataset: {e}")
            self.logger.warning("Using dummy dataset for testing...")
            
            # Fallback: 創建dummy dataset for testing
            class DummyDataset:
                def __init__(self, size=100):
                    self.size = size
                    
                def __len__(self):
                    return self.size
                    
                def __getitem__(self, idx):
                    return {
                        'img': torch.randn(3, 512, 512),
                        'img_mask': torch.randint(0, 2, (512, 512)).float(),
                        'anomaly': torch.randint(0, 2, (1,)).float(),
                        'cls_name': ['dummy'],
                        'img_path': [f'dummy_{idx}.jpg']
                    }
            
            train_dataset = DummyDataset()
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.logger.info(f"Training dataset: {len(train_dataset)} samples")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        
        return train_dataloader
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """訓練一個epoch"""
        self.model.train()
        self.model.prompt_learner.train()  # 確保prompt learner在訓練模式
        
        total_loss = 0
        loss_components = {}
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.training.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 準備資料
                images = batch['img'].to(self.device)
                masks = batch['img_mask'].to(self.device)
                labels = batch['anomaly'].to(self.device).squeeze().long()
                
                # 調整mask維度到64x64 (attention resolution)
                masks_64 = F.interpolate(
                    masks, 
                    size=(64, 64), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
                
                # 前向傳播
                predictions = self.model(images, mode='train')
                
                # 準備targets
                targets = {
                    'masks': masks_64,
                    'labels': labels
                }
                
                # 計算loss
                loss, loss_dict = self.loss_fn(predictions, targets)
                
                # 反向傳播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪 (可選)
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 累積loss
                total_loss += loss.item()
                
                # 累積loss components
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        if key not in loss_components:
                            loss_components[key] = 0
                        loss_components[key] += value.item()
                
                # 更新進度條
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # 定期打印詳細loss
                if batch_idx % (len(dataloader) // 4) == 0:
                    self.logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # 計算平均loss
        avg_loss = total_loss / len(dataloader)
        
        # 計算平均loss components
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        # 記錄詳細loss breakdown
        self.logger.info(f"Epoch {epoch} completed:")
        self.logger.info(f"  Average Loss: {avg_loss:.4f}")
        for key, value in avg_components.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return avg_loss
    
    def train(self):
        """完整訓練流程"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.training.epochs} epochs")
        
        # 設定資料載入器
        train_dataloader = self._setup_dataloader()
        
        # 訓練循環
        for epoch in range(self.current_epoch + 1, self.config.training.epochs + 1):
            # 訓練一個epoch
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            # 學習率調度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 保存checkpoint
            if epoch % self.config.training.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.config.system.save_path, 
                    f'checkpoint_epoch_{epoch}.pth'
                )
                self.model.save_checkpoint(checkpoint_path, epoch, avg_loss)
                
                # 如果是最佳模型，額外保存為best.pth
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    best_path = os.path.join(self.config.system.save_path, 'best.pth')
                    self.model.save_checkpoint(best_path, epoch, avg_loss)
                    self.logger.info(f"New best model saved: {avg_loss:.4f}")
            
            # 記錄訓練進度
            if epoch % self.config.training.print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        
        # 保存最終模型
        final_path = os.path.join(self.config.system.save_path, 'final.pth')
        self.model.save_checkpoint(final_path, self.config.training.epochs, avg_loss)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Anomaly Detection Training")
    
    # 基本參數
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--data_path", type=str, default="./data/mvtec", help="Dataset path")
    parser.add_argument("--save_path", type=str, default="./results", help="Save path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    
    # 訓練參數
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # 模型參數
    parser.add_argument("--n_ctx", type=int, default=12, help="Number of learnable tokens")
    parser.add_argument("--depth", type=int, default=9, help="Compound prompts depth")
    
    # 系統參數
    parser.add_argument("--seed", type=int, default=111, help="Random seed")
    parser.add_argument("--debug", action='store_true', help="Use debug config")
    
    args = parser.parse_args()
    
    # 載入或創建配置
    if args.config:
        config = HybridConfig.load(args.config)
    elif args.debug:
        config = get_debug_config()
    else:
        config = get_default_config()
    
    # 用命令行參數覆蓋配置
    if args.data_path:
        config.data.data_path = args.data_path
    if args.save_path:
        config.system.save_path = args.save_path
    if args.checkpoint:
        config.system.checkpoint_path = args.checkpoint
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.n_ctx:
        config.model.n_ctx = args.n_ctx
    if args.depth:
        config.model.depth = args.depth
    if args.seed:
        config.system.seed = args.seed
    
    # 創建保存目錄
    os.makedirs(config.system.save_path, exist_ok=True)
    
    # 打印配置
    print("=== Training Configuration ===")
    print(f"Data path: {config.data.data_path}")
    print(f"Save path: {config.system.save_path}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Device: {config.system.device}")
    print("=" * 30)
    
    try:
        # 初始化訓練器
        trainer = HybridTrainer(config)
        
        # 開始訓練
        trainer.train()
        
        print("✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
