"""
Hybrid Loss Functions for Cross-Attention Level Prompt Optimization
基於cross-attention的混合損失函數，用於優化learnable prompts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
import sys
import os

# 使用本地utilities - 修復相對import問題
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.loss_functions import FocalLoss, BinaryDiceLoss


class CrossAttentionLoss(nn.Module):
    """
    Cross-Attention Level的Loss Function
    專門設計用於在attention maps上優化learnable prompts
    """
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        
        # 預設loss權重
        default_weights = {
            'focal': 1.0,
            'dice': 1.0, 
            'image': 1.0,
            'sharpness': 0.5,
            'coverage': 0.3,
            'consistency': 0.2
        }
        
        self.weights = loss_weights if loss_weights is not None else default_weights
        
        # 基礎loss組件 (來自AnomalyCLIP)
        self.focal_loss = FocalLoss()
        self.dice_loss = BinaryDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 新增的cross-attention specific losses
        self.attention_sharpness_loss = AttentionSharpnessLoss()
        self.attention_coverage_loss = AttentionCoverageLoss()
        self.attention_consistency_loss = AttentionConsistencyLoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        計算完整的混合損失
        
        Args:
            predictions: 模型預測結果
                - normal_attention: (batch, H, W)
                - abnormal_attention: (batch, H, W)
                - prompt_embeddings: dict with normal/abnormal embeddings
            targets: 真實標籤
                - masks: (batch, H, W) - pixel-level ground truth
                - labels: (batch,) - image-level ground truth
                
        Returns:
            total_loss: 總損失
            loss_dict: 各項損失的詳細數值
        """
        
        normal_attention = predictions['normal_attention']
        abnormal_attention = predictions['abnormal_attention']
        gt_masks = targets['masks']
        gt_labels = targets['labels']
        
        total_loss = 0
        loss_dict = {}
        
        # 1. 基礎pixel-level losses (來自AnomalyCLIP)
        pixel_loss, pixel_dict = self._compute_pixel_losses(
            normal_attention, abnormal_attention, gt_masks
        )
        total_loss += pixel_loss
        loss_dict.update(pixel_dict)
        
        # 2. Image-level loss
        image_loss, image_dict = self._compute_image_level_loss(
            normal_attention, abnormal_attention, gt_labels
        )
        total_loss += image_loss
        loss_dict.update(image_dict)
        
        # 3. Cross-attention specific losses
        attention_loss, attention_dict = self._compute_attention_losses(
            normal_attention, abnormal_attention, gt_masks
        )
        total_loss += attention_loss
        loss_dict.update(attention_dict)
        
        # 4. Prompt consistency loss (可選)
        if 'prompt_embeddings' in predictions:
            consistency_loss = self._compute_consistency_loss(predictions['prompt_embeddings'])
            total_loss += self.weights['consistency'] * consistency_loss
            loss_dict['consistency_loss'] = consistency_loss
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def _compute_pixel_losses(self, normal_attention: torch.Tensor, 
                            abnormal_attention: torch.Tensor,
                            gt_masks: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """計算pixel-level losses (直接應用AnomalyCLIP的loss函數)"""
        
        # 組合attention maps成similarity-like format
        attention_stack = torch.stack([normal_attention, abnormal_attention], dim=1)  # (batch, 2, H, W)
        
        # 調整gt_masks維度到與attention maps相同
        if gt_masks.shape[-2:] != normal_attention.shape[-2:]:
            # 確保gt_masks是4D: (batch, 1, H, W)
            if len(gt_masks.shape) == 3:
                gt_masks_4d = gt_masks.unsqueeze(1).float()
            elif len(gt_masks.shape) == 2:
                gt_masks_4d = gt_masks.unsqueeze(0).unsqueeze(1).float()
            else:
                gt_masks_4d = gt_masks.float()
            
            gt_masks_4d = F.interpolate(
                gt_masks_4d, 
                size=normal_attention.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            gt_masks = gt_masks_4d.squeeze(1)
        
        # FocalLoss on combined attention maps
        focal_loss = self.focal_loss(attention_stack, gt_masks)
        
        # DiceLoss on individual channels
        dice_abnormal = self.dice_loss(abnormal_attention, gt_masks)
        dice_normal = self.dice_loss(normal_attention, 1 - gt_masks)
        
        # 加權組合
        pixel_loss = (
            self.weights['focal'] * focal_loss + 
            self.weights['dice'] * (dice_abnormal + dice_normal)
        )
        
        loss_dict = {
            'focal_loss': focal_loss,
            'dice_abnormal': dice_abnormal,
            'dice_normal': dice_normal,
            'pixel_loss': pixel_loss
        }
        
        return pixel_loss, loss_dict
    
    def _compute_image_level_loss(self, normal_attention: torch.Tensor,
                                abnormal_attention: torch.Tensor,
                                gt_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """計算image-level classification loss"""
        
        # 基於attention statistics計算image-level logits
        image_logits = self._compute_image_level_logits(normal_attention, abnormal_attention)
        
        # Cross-entropy loss
        image_loss = self.ce_loss(image_logits, gt_labels.long())
        
        # 加權
        weighted_image_loss = self.weights['image'] * image_loss
        
        loss_dict = {
            'image_ce_loss': image_loss,
            'weighted_image_loss': weighted_image_loss
        }
        
        return weighted_image_loss, loss_dict
    
    def _compute_image_level_logits(self, normal_attention: torch.Tensor,
                                  abnormal_attention: torch.Tensor) -> torch.Tensor:
        """基於attention maps計算image-level logits"""
        
        batch_size = normal_attention.shape[0]
        logits = []
        
        for i in range(batch_size):
            # 簡單統計方法
            normal_score = torch.mean(normal_attention[i])
            abnormal_score = torch.mean(abnormal_attention[i])
            
            # 也可以使用更sophisticated的方法，如top-k statistics
            # k = int(0.04 * abnormal_attention[i].numel())
            # abnormal_top_k = torch.topk(abnormal_attention[i].flatten(), k).values.mean()
            # abnormal_median = torch.median(abnormal_attention[i])
            # abnormal_score = 1 - (abnormal_median / (abnormal_top_k + 1e-6))
            
            image_logit = torch.stack([normal_score, abnormal_score])
            logits.append(image_logit)
        
        return torch.stack(logits)  # (batch, 2)
    
    def _compute_attention_losses(self, normal_attention: torch.Tensor,
                                abnormal_attention: torch.Tensor,
                                gt_masks: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """計算cross-attention specific losses"""
        
        # Sharpness loss: 鼓勵attention在defect區域更focused
        sharpness_loss = self.attention_sharpness_loss(abnormal_attention, gt_masks)
        
        # Coverage loss: 確保attention覆蓋所有defect areas
        coverage_loss = self.attention_coverage_loss(abnormal_attention, gt_masks)
        
        # Consistency loss: normal和abnormal attention的對比
        consistency_loss = self.attention_consistency_loss(normal_attention, abnormal_attention, gt_masks)
        
        # 加權組合
        total_attention_loss = (
            self.weights['sharpness'] * sharpness_loss +
            self.weights['coverage'] * coverage_loss +
            0.1 * consistency_loss  # 較小權重的consistency
        )
        
        loss_dict = {
            'sharpness_loss': sharpness_loss,
            'coverage_loss': coverage_loss,
            'attention_consistency_loss': consistency_loss,
            'total_attention_loss': total_attention_loss
        }
        
        return total_attention_loss, loss_dict
    
    def _compute_consistency_loss(self, prompt_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """計算prompt embeddings的consistency loss"""
        
        normal_embeds = prompt_embeddings['normal']    # (1, 77, 768)
        abnormal_embeds = prompt_embeddings['abnormal'] # (1, 77, 768)
        
        # 確保normal和abnormal embeddings有適當的distance
        # 計算cosine similarity
        normal_flat = normal_embeds.view(-1)
        abnormal_flat = abnormal_embeds.view(-1)
        
        cosine_sim = F.cosine_similarity(normal_flat.unsqueeze(0), abnormal_flat.unsqueeze(0))
        
        # 我們希望similarity不要太高（鼓勵diversity），但也不要太低（保持meaningful）
        target_similarity = 0.3  # 目標相似度
        consistency_loss = F.mse_loss(cosine_sim, torch.tensor(target_similarity, device=cosine_sim.device))
        
        return consistency_loss


class AttentionSharpnessLoss(nn.Module):
    """鼓勵attention在defect區域更sharp/focused"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, attention_maps: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_maps: (batch, H, W)
            gt_masks: (batch, H, W)
        """
        
        # 調整mask維度
        if gt_masks.shape[-2:] != attention_maps.shape[-2:]:
            # 確保gt_masks是4D: (batch, 1, H, W)
            if len(gt_masks.shape) == 3:
                gt_masks_4d = gt_masks.unsqueeze(1).float()
            elif len(gt_masks.shape) == 2:
                gt_masks_4d = gt_masks.unsqueeze(0).unsqueeze(1).float()
            else:
                gt_masks_4d = gt_masks.float()
                
            gt_masks_4d = F.interpolate(
                gt_masks_4d,
                size=attention_maps.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            gt_masks = gt_masks_4d.squeeze(1)
        
        # 只在有defect的區域計算sharpness
        defect_areas = gt_masks > 0.5
        
        total_loss = 0
        valid_samples = 0
        
        for i in range(attention_maps.shape[0]):
            if defect_areas[i].sum() > 0:
                # 在defect區域內的attention values
                defect_attention = attention_maps[i][defect_areas[i]]
                
                # 計算variance作為sharpness measure (越大越sharp)
                attention_var = torch.var(defect_attention)
                
                # 我們要maximize variance，所以minimize negative variance
                total_loss += -attention_var
                valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=attention_maps.device)


class AttentionCoverageLoss(nn.Module):
    """確保attention覆蓋所有defect areas"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, attention_maps: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        計算IoU-like loss來確保coverage
        """
        
        # 調整mask維度
        if gt_masks.shape[-2:] != attention_maps.shape[-2:]:
            # 確保gt_masks是4D: (batch, 1, H, W)
            if len(gt_masks.shape) == 3:
                gt_masks_4d = gt_masks.unsqueeze(1).float()
            elif len(gt_masks.shape) == 2:
                gt_masks_4d = gt_masks.unsqueeze(0).unsqueeze(1).float()
            else:
                gt_masks_4d = gt_masks.float()
                
            gt_masks_4d = F.interpolate(
                gt_masks_4d,
                size=attention_maps.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            gt_masks = gt_masks_4d.squeeze(1)
        
        # 將attention map二值化
        attention_threshold = attention_maps.view(attention_maps.shape[0], -1).mean(dim=1, keepdim=True).unsqueeze(-1)
        attention_binary = (attention_maps > attention_threshold).float()
        
        # 計算IoU
        intersection = (attention_binary * gt_masks).sum(dim=(-2, -1))
        union = attention_binary.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection
        
        iou = intersection / (union + 1e-6)
        coverage_loss = 1 - iou.mean()  # Minimize (1 - IoU)
        
        return coverage_loss


class AttentionConsistencyLoss(nn.Module):
    """確保normal和abnormal attention的合理對比"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, normal_attention: torch.Tensor, 
                abnormal_attention: torch.Tensor,
                gt_masks: torch.Tensor) -> torch.Tensor:
        """
        確保在defect區域abnormal > normal，在normal區域normal > abnormal
        """
        
        # 調整mask維度
        if gt_masks.shape[-2:] != normal_attention.shape[-2:]:
            # 確保gt_masks是4D: (batch, 1, H, W)
            if len(gt_masks.shape) == 3:
                gt_masks_4d = gt_masks.unsqueeze(1).float()
            elif len(gt_masks.shape) == 2:
                gt_masks_4d = gt_masks.unsqueeze(0).unsqueeze(1).float()
            else:
                gt_masks_4d = gt_masks.float()
                
            gt_masks_4d = F.interpolate(
                gt_masks_4d,
                size=normal_attention.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            gt_masks = gt_masks_4d.squeeze(1)
        
        # 在defect區域，abnormal attention應該更高
        defect_areas = gt_masks > 0.5
        defect_consistency = torch.zeros(1, device=normal_attention.device)
        
        if defect_areas.sum() > 0:
            normal_in_defect = normal_attention[defect_areas]
            abnormal_in_defect = abnormal_attention[defect_areas]
            
            # 鼓勵abnormal > normal in defect areas
            defect_consistency = F.relu(normal_in_defect - abnormal_in_defect + 0.1).mean()
        
        # 在normal區域，normal attention應該更高
        normal_areas = gt_masks <= 0.5
        normal_consistency = torch.zeros(1, device=normal_attention.device)
        
        if normal_areas.sum() > 0:
            normal_in_normal = normal_attention[normal_areas]
            abnormal_in_normal = abnormal_attention[normal_areas]
            
            # 鼓勵normal > abnormal in normal areas  
            normal_consistency = F.relu(abnormal_in_normal - normal_in_normal + 0.1).mean()
        
        return defect_consistency + normal_consistency


if __name__ == "__main__":
    # 測試loss functions
    print("Testing CrossAttentionLoss...")
    
    # 創建dummy data
    batch_size = 2
    h, w = 64, 64
    
    predictions = {
        'normal_attention': torch.randn(batch_size, h, w),
        'abnormal_attention': torch.randn(batch_size, h, w),
        'prompt_embeddings': {
            'normal': torch.randn(1, 77, 768),
            'abnormal': torch.randn(1, 77, 768)
        }
    }
    
    targets = {
        'masks': torch.randint(0, 2, (batch_size, h, w)).float(),
        'labels': torch.randint(0, 2, (batch_size,))
    }
    
    # 測試loss
    loss_fn = CrossAttentionLoss()
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss breakdown:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    print("✓ CrossAttentionLoss test passed!")