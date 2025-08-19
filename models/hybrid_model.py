"""
Hybrid Anomaly Detector
結合D2D diffusion pipeline與AnomalyCLIP learnable prompts的主模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import sys
import os

# 使用本地pipeline - 修復相對import問題
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.diffusion import EasonADPipeline
from models.prompt_learner import MinimalPromptLearner


class HybridAnomalyDetector(nn.Module):
    """
    混合異常檢測模型
    結合D2D的cross-attention機制與AnomalyCLIP的learnable prompts
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 載入並設定D2D pipeline
        self.pipeline = self._setup_diffusion_pipeline()
        
        # 2. 初始化prompt learner
        self.prompt_learner = self._setup_prompt_learner()
        
        # 3. 設定attention extraction參數
        self.target_resolution = 64  # D2D使用64x64的attention resolution
        self.normal_token_pos = self._find_token_position("object")
        self.abnormal_token_pos = self._find_token_position("damaged")
        
        # 4. 凍結D2D pipeline的所有參數
        self._freeze_diffusion_parameters()
        
    def _setup_diffusion_pipeline(self) -> EasonADPipeline:
        """設定D2D diffusion pipeline"""
        pipeline = EasonADPipeline.from_pretrained(
            self.config.get('pretrained_model_name_or_path', 'stabilityai/stable-diffusion-2-1-base'),
            torch_dtype=torch.float16 if self.config.get('use_fp16', True) else torch.float32,
        ).to(self.device)
        
        # 禁用xformers (for reproducibility)
        try:
            pipeline.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Warning: disable_xformers_memory_efficient_attention not available")
            
        return pipeline
    
    def _setup_prompt_learner(self) -> MinimalPromptLearner:
        """設定prompt learner"""
        prompt_config = {
            "Prompt_length": self.config.get('n_ctx', 12),
            "learnabel_text_embedding_depth": self.config.get('depth', 9),
            "learnabel_text_embedding_length": self.config.get('t_n_ctx', 4)
        }
        
        prompt_learner = MinimalPromptLearner(
            self.pipeline.text_encoder,
            prompt_config,
            tokenizer=self.pipeline.tokenizer
        ).to(self.device)
        
        return prompt_learner
    
    def _find_token_position(self, word: str) -> int:
        """
        找到特定word在tokenized sequence中的位置
        這是一個簡化版本，實際使用中可能需要更sophisticated的token matching
        """
        # 對於我們的設計，我們知道learnable tokens在前面，class tokens在後面
        # 這裡返回一個預估的位置，實際訓練中可能需要調整
        if word == "object":
            return 13  # [CLS] + 12 learnable tokens + 1 = 13
        elif word == "damaged":  
            return 13  # 同樣位置，因為我們主要關注semantic concept
        else:
            return 13  # default position
    
    def _freeze_diffusion_parameters(self):
        """凍結D2D pipeline的所有參數，只訓練prompt learner"""
        # Freeze all pipeline components
        total_frozen = 0
        
        for component_name in ['vae', 'text_encoder', 'unet', 'scheduler']:
            component = getattr(self.pipeline, component_name, None)
            if component is not None and hasattr(component, 'parameters'):
                for param in component.parameters():
                    param.requires_grad = False
                    total_frozen += param.numel()
                    
        print(f"Frozen {total_frozen} diffusion parameters")
        print(f"Trainable prompt parameters: {sum(p.numel() for p in self.prompt_learner.parameters())}")
    
    def forward(self, images: torch.Tensor, mode: str = 'train') -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            images: Input images (batch_size, 3, H, W)
            mode: 'train' or 'eval'
            
        Returns:
            Dictionary containing attention maps and other outputs
        """
        
        if mode == 'train':
            return self._training_forward(images)
        else:
            return self._inference_forward(images)
    
    def _training_forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """訓練時的前向傳播"""
        
        # 1. 生成learnable prompt embeddings
        normal_embeds, abnormal_embeds, compound_prompts = self.prompt_learner()
        
        # 2. 提取normal attention maps
        normal_attention = self._extract_attention_maps(
            images, normal_embeds, self.normal_token_pos
        )
        
        # 3. 重置attention controller並提取abnormal attention maps
        self._reset_attention_controller()
        abnormal_attention = self._extract_attention_maps(
            images, abnormal_embeds, self.abnormal_token_pos
        )
        
        return {
            'normal_attention': normal_attention,      # (batch, 64, 64)
            'abnormal_attention': abnormal_attention,  # (batch, 64, 64)
            'prompt_embeddings': {
                'normal': normal_embeds,
                'abnormal': abnormal_embeds
            },
            'compound_prompts': compound_prompts
        }
    
    def _inference_forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理時的前向傳播，與訓練相同但包含額外的便利輸出"""
        
        with torch.no_grad():
            results = self._training_forward(images)
            
            # 添加便於使用的輸出
            normal_attention = results['normal_attention']
            abnormal_attention = results['abnormal_attention']
            
            # 計算image-level預測分數
            image_scores = self._compute_image_level_scores(normal_attention, abnormal_attention)
            
            # 組合成similarity-like maps for compatibility
            similarity_maps = torch.stack([normal_attention, abnormal_attention], dim=1)  # (batch, 2, 64, 64)
            
            results.update({
                'image_scores': image_scores,      # (batch,)
                'similarity_maps': similarity_maps, # (batch, 2, 64, 64)
                'anomaly_maps': abnormal_attention  # (batch, 64, 64) - main output
            })
            
        return results
    
    def _extract_attention_maps(self, images: torch.Tensor, prompt_embeds: torch.Tensor, 
                              target_token_pos: int) -> torch.Tensor:
        """
        使用給定的prompt embeddings提取cross-attention maps
        
        Args:
            images: Input images (batch, 3, H, W)
            prompt_embeds: Prompt embeddings (1, 77, 768)
            target_token_pos: Target token position to extract attention for
            
        Returns:
            Attention maps (batch, 64, 64)
        """
        
        # 設定attention controller
        self.pipeline.controller = self.pipeline.create_controller()
        self.pipeline.register_attention_control(self.pipeline.controller)
        
        # 轉換圖像為latents
        with torch.no_grad():
            latents = self.pipeline.image2latent(images)
            
            # 擴展prompt_embeds到batch size
            batch_size = images.shape[0]
            if prompt_embeds.shape[0] == 1 and batch_size > 1:
                prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            
            # 運行UNet forward pass來觸發attention capture
            timestep = torch.tensor([50], device=self.device)  # 固定timestep
            noise_pred = self.pipeline.unet(
                latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]
        
        # 從attention controller提取maps
        attention_maps = self._aggregate_attention_maps(target_token_pos, batch_size)
        
        return attention_maps
    
    def _aggregate_attention_maps(self, target_token_pos: int, batch_size: int) -> torch.Tensor:
        """
        聚合不同layer和位置的attention maps
        
        Args:
            target_token_pos: Target token position
            batch_size: Batch size
            
        Returns:
            Aggregated attention maps (batch_size, 64, 64)
        """
        
        attention_store = self.pipeline.controller.get_average_attention()
        target_attentions = []
        
        # 遍歷不同的attention locations
        for location in ["up", "down", "mid"]:
            location_key = f"{location}_cross"
            if location_key in attention_store:
                for attn in attention_store[location_key]:
                    # 只處理64x64解析度的attention
                    if attn.shape[1] == self.target_resolution ** 2:
                        # attn shape: (batch*heads, spatial_tokens, text_tokens)
                        # 提取target token的attention
                        token_attn = attn[:, :, target_token_pos]  # (batch*heads, spatial_tokens)
                        
                        # 平均across heads
                        heads_per_batch = token_attn.shape[0] // batch_size
                        token_attn = token_attn.view(batch_size, heads_per_batch, -1).mean(dim=1)  # (batch, spatial_tokens)
                        
                        # Reshape到spatial dimensions
                        token_attn = token_attn.view(batch_size, self.target_resolution, self.target_resolution)
                        
                        target_attentions.append(token_attn)
        
        # 聚合所有attention maps
        if target_attentions:
            # 平均所有layers的attention
            aggregated = torch.stack(target_attentions, dim=0).mean(dim=0)  # (batch, 64, 64)
        else:
            # Fallback: 返回zeros
            aggregated = torch.zeros(batch_size, self.target_resolution, self.target_resolution, 
                                   device=self.device)
            print("Warning: No attention maps found, returning zeros")
        
        return aggregated
    
    def _reset_attention_controller(self):
        """重置attention controller以進行新的attention capture"""
        self.pipeline.controller = self.pipeline.create_controller()
        self.pipeline.register_attention_control(self.pipeline.controller)
    
    def _compute_image_level_scores(self, normal_attention: torch.Tensor, 
                                   abnormal_attention: torch.Tensor) -> torch.Tensor:
        """
        計算image-level異常分數
        
        Args:
            normal_attention: (batch, 64, 64)
            abnormal_attention: (batch, 64, 64)
            
        Returns:
            Image-level scores (batch,)
        """
        
        # 方法1: 簡單的mean ratio
        normal_scores = torch.mean(normal_attention.view(normal_attention.shape[0], -1), dim=1)
        abnormal_scores = torch.mean(abnormal_attention.view(abnormal_attention.shape[0], -1), dim=1)
        
        # 計算異常分數 (0-1, 1表示更可能是異常)
        total_scores = normal_scores + abnormal_scores + 1e-6
        anomaly_scores = abnormal_scores / total_scores
        
        return anomaly_scores
    
    def get_trainable_parameters(self):
        """獲取可訓練的參數"""
        return self.prompt_learner.parameters()
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'prompt_learner_state_dict': self.prompt_learner.state_dict(),
            'loss': loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """載入checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.prompt_learner.load_state_dict(checkpoint['prompt_learner_state_dict'])
        print(f"Checkpoint loaded from {path}, epoch: {checkpoint.get('epoch', 'unknown')}")
        return checkpoint


if __name__ == "__main__":
    # 測試HybridAnomalyDetector
    print("Testing HybridAnomalyDetector...")
    
    config = {
        'pretrained_model_name_or_path': 'stabilityai/stable-diffusion-2-1-base',
        'n_ctx': 12,
        'depth': 9,
        't_n_ctx': 4,
        'use_fp16': False  # 測試時使用fp32
    }
    
    try:
        model = HybridAnomalyDetector(config)
        
        # 測試forward pass
        dummy_images = torch.randn(2, 3, 512, 512)  # batch_size=2
        
        print("Testing training forward...")
        results = model(dummy_images, mode='train')
        
        print(f"Normal attention shape: {results['normal_attention'].shape}")
        print(f"Abnormal attention shape: {results['abnormal_attention'].shape}")
        
        print("Testing inference forward...")
        results = model(dummy_images, mode='eval')
        
        print(f"Image scores shape: {results['image_scores'].shape}")
        print(f"Similarity maps shape: {results['similarity_maps'].shape}")
        print(f"Anomaly maps shape: {results['anomaly_maps'].shape}")
        
        print("✓ HybridAnomalyDetector test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: This test requires access to D2D pipeline and may fail without proper setup")