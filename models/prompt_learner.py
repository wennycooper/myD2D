"""
Minimal Prompt Learner for Hybrid Anomaly Detection
基於AnomalyCLIP的最小化prompt learner實現
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from copy import deepcopy


class MinimalPromptLearner(nn.Module):
    """
    從AnomalyCLIP提取的最小prompt learner
    專門用於生成learnable normal/abnormal prompt embeddings
    """
    
    def __init__(self, text_encoder, design_details: dict, tokenizer=None):
        super().__init__()
        
        # 基本設定
        self.n_ctx = design_details.get("Prompt_length", 12)  # learnable tokens數量
        self.text_encoder_n_ctx = design_details.get("learnabel_text_embedding_length", 4)
        self.compound_prompts_depth = design_details.get("learnabel_text_embedding_depth", 9)
        
        # 獲取text encoder的hidden size
        if hasattr(text_encoder, 'config'):
            ctx_dim = text_encoder.config.hidden_size  # 768 for SD text encoder
        else:
            ctx_dim = 768  # fallback
            
        self.ctx_dim = ctx_dim
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        # Learnable context vectors for normal and abnormal
        self.ctx_normal = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        self.ctx_abnormal = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        
        # Deep compound prompts (來自AnomalyCLIP的advanced feature)
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        
        # 初始化parameters
        self._initialize_parameters()
        
        # 設定固定的template embeddings
        self._setup_templates()
        
    def _initialize_parameters(self):
        """初始化所有learnable parameters"""
        nn.init.normal_(self.ctx_normal, std=0.02)
        nn.init.normal_(self.ctx_abnormal, std=0.02)
        
        for prompt in self.compound_prompts_text:
            nn.init.normal_(prompt, std=0.02)
            
    def _setup_templates(self):
        """設定normal/abnormal模板的固定部分"""
        # 模板設計: [CLS] + [learnable_tokens] + [fixed_class_tokens] + [PAD]
        normal_template = "object."  # 正常物體模板
        abnormal_template = "damaged object."  # 異常物體模板
        
        # Tokenize templates
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for template setup. Please pass tokenizer to MinimalPromptLearner.")
        tokenizer = self.tokenizer
        
        normal_tokens = tokenizer(
            normal_template, 
            padding="max_length", 
            max_length=77,  # SD text encoder max length
            return_tensors="pt"
        )
        
        abnormal_tokens = tokenizer(
            abnormal_template,
            padding="max_length", 
            max_length=77, 
            return_tensors="pt"
        )
        
        # 獲取token embeddings
        with torch.no_grad():
            # 確保token tensor在正確的設備上
            device = next(self.text_encoder.parameters()).device
            normal_ids = normal_tokens.input_ids.to(device)
            abnormal_ids = abnormal_tokens.input_ids.to(device)
            
            if hasattr(self.text_encoder, 'text_model'):
                # For CLIP-like text encoders
                normal_emb = self.text_encoder.text_model.embeddings.token_embedding(normal_ids)
                abnormal_emb = self.text_encoder.text_model.embeddings.token_embedding(abnormal_ids)
            else:
                # Fallback for other text encoders
                normal_emb = self.text_encoder.get_input_embeddings()(normal_ids)
                abnormal_emb = self.text_encoder.get_input_embeddings()(abnormal_ids)
        
        # 分離template的不同部分
        # Template structure: [CLS] [learnable_ctx] [class_tokens] [PAD]...
        
        # [CLS] token
        self.register_buffer("normal_prefix", normal_emb[:, :1, :])      # (1, 1, 768)
        self.register_buffer("abnormal_prefix", abnormal_emb[:, :1, :])  # (1, 1, 768)
        
        # 固定的class tokens + [PAD] tokens  
        # 為learnable tokens預留位置: 1 (CLS) + n_ctx (learnable) + remaining (fixed)
        suffix_start_idx = 1  # 跳過CLS，因為我們會手動插入learnable tokens
        self.register_buffer("normal_suffix", normal_emb[:, suffix_start_idx:, :])    # (1, 76, 768)
        self.register_buffer("abnormal_suffix", abnormal_emb[:, suffix_start_idx:, :]) # (1, 76, 768)
        
        # 同時保存tokenized prompts for reference
        self.register_buffer("normal_tokens", normal_tokens.input_ids)
        self.register_buffer("abnormal_tokens", abnormal_tokens.input_ids)
        
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, List[nn.Parameter]]:
        """
        生成normal和abnormal prompt embeddings
        
        Returns:
            normal_prompt: (1, 77, 768) - Normal prompt embeddings
            abnormal_prompt: (1, 77, 768) - Abnormal prompt embeddings  
            compound_prompts: List of compound prompt parameters
        """
        
        # 組裝normal prompt
        # Structure: [CLS] + [learnable_ctx] + [object.] + [PAD]...
        normal_prompt = self._assemble_prompt(
            self.normal_prefix,     # [CLS]
            self.ctx_normal,        # learnable context  
            self.normal_suffix      # [object.] + [PAD]...
        )
        
        # 組裝abnormal prompt  
        # Structure: [CLS] + [learnable_ctx] + [damaged object.] + [PAD]...
        abnormal_prompt = self._assemble_prompt(
            self.abnormal_prefix,   # [CLS]
            self.ctx_abnormal,      # learnable context
            self.abnormal_suffix    # [damaged object.] + [PAD]...
        )
        
        return normal_prompt, abnormal_prompt, self.compound_prompts_text
    
    def _assemble_prompt(self, prefix: torch.Tensor, ctx: torch.Tensor, suffix: torch.Tensor) -> torch.Tensor:
        """
        組裝完整的prompt embedding
        
        Args:
            prefix: [CLS] token embedding (1, 1, 768)
            ctx: Learnable context embeddings (1, n_ctx, 768)  
            suffix: Fixed suffix embeddings (1, remaining_len, 768)
            
        Returns:
            Complete prompt embedding (1, 77, 768)
        """
        
        # 計算suffix需要截斷的長度，確保總長度為77
        total_prefix_ctx_len = prefix.shape[1] + ctx.shape[1]  # 1 + n_ctx
        remaining_len = 77 - total_prefix_ctx_len
        
        # 截斷suffix到合適長度
        suffix_truncated = suffix[:, :remaining_len, :]
        
        # 組裝: [prefix] + [ctx] + [suffix_truncated]
        prompt = torch.cat([
            prefix,            # (1, 1, 768)
            ctx,               # (1, n_ctx, 768)  
            suffix_truncated   # (1, remaining_len, 768)
        ], dim=1)  # (1, 77, 768)
        
        return prompt
    
    def get_learnable_parameters(self) -> List[nn.Parameter]:
        """獲取所有可學習的參數，用於optimizer設定"""
        params = [self.ctx_normal, self.ctx_abnormal]
        params.extend(self.compound_prompts_text)
        return params
    
    def get_prompt_info(self) -> dict:
        """獲取prompt配置信息，用於debug和logging"""
        return {
            'n_ctx': self.n_ctx,
            'ctx_dim': self.ctx_dim,
            'compound_depth': self.compound_prompts_depth,
            'text_encoder_n_ctx': self.text_encoder_n_ctx,
            'total_learnable_params': sum(p.numel() for p in self.get_learnable_parameters())
        }


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """創建模組的多個副本 (從AnomalyCLIP移植)"""
    return nn.ModuleList([deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    # 簡單測試
    print("Testing MinimalPromptLearner...")
    
    # Mock text encoder for testing
    class MockTextEncoder:
        def __init__(self):
            self.config = type('Config', (), {'hidden_size': 768})()
            
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                # 返回mock tokenization結果
                return type('TokenizerOutput', (), {
                    'input_ids': torch.randint(0, 1000, (1, 77))
                })()
                
        def __init__(self):
            self.config = type('Config', (), {'hidden_size': 768})()
            self.tokenizer = self.MockTokenizer()
            
        def get_input_embeddings(self):
            return nn.Embedding(1000, 768)
    
    # 測試
    mock_encoder = MockTextEncoder()
    design_details = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": 9,
        "learnabel_text_embedding_length": 4
    }
    
    prompt_learner = MinimalPromptLearner(mock_encoder, design_details)
    
    # Forward pass
    normal_prompt, abnormal_prompt, compound_prompts = prompt_learner()
    
    print(f"Normal prompt shape: {normal_prompt.shape}")
    print(f"Abnormal prompt shape: {abnormal_prompt.shape}")
    print(f"Number of compound prompts: {len(compound_prompts)}")
    print(f"Prompt info: {prompt_learner.get_prompt_info()}")
    
    print("✓ MinimalPromptLearner test passed!")