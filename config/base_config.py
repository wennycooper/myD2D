"""
Base Configuration for Hybrid Anomaly Detection
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ModelConfig:
    """模型相關配置"""
    
    # D2D Pipeline設定
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
    use_fp16: bool = True
    
    # Prompt Learner設定
    n_ctx: int = 12  # learnable tokens數量
    depth: int = 9   # compound prompts深度
    t_n_ctx: int = 4 # text encoder context長度
    
    # Attention extraction設定
    target_resolution: int = 64
    normal_token_pos: int = 13
    abnormal_token_pos: int = 13


@dataclass 
class TrainingConfig:
    """訓練相關配置"""
    
    # 基本訓練參數
    epochs: int = 15
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss權重
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'focal': 1.0,
        'dice': 1.0, 
        'image': 1.0,
        'sharpness': 0.5,
        'coverage': 0.3
    })
    
    # Optimizer設定
    optimizer: str = "adam"
    betas: tuple = (0.5, 0.999)
    
    # 學習率調度
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    
    # Checkpoint設定
    save_freq: int = 5
    print_freq: int = 1
    eval_freq: int = 5


@dataclass
class DataConfig:
    """資料相關配置"""
    
    # 資料集路徑
    data_path: str = "./data/mvtec"
    dataset_name: str = "mvtec"
    
    # 圖像預處理
    resolution: int = 512
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # 資料增強
    use_augmentation: bool = True
    rotation_degrees: int = 10
    brightness: float = 0.1
    contrast: float = 0.1
    
    # 資料載入
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """評估相關配置"""
    
    # 評估指標
    metrics: str = "image-pixel-level"  # 'image-level', 'pixel-level', 'image-pixel-level'
    
    # 後處理
    sigma: int = 4  # 高斯濾波sigma
    
    # 視覺化
    save_visualizations: bool = False
    save_detailed_results: bool = True
    
    # 測試設定
    test_batch_size: int = 1


@dataclass
class SystemConfig:
    """系統相關配置"""
    
    # 路徑設定
    save_path: str = "./results"
    checkpoint_path: Optional[str] = None
    
    # 系統設定
    seed: int = 111
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # 記錄設定
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None


@dataclass
class HybridConfig:
    """完整的混合模型配置"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """後處理配置"""
        # 確保save_path存在
        os.makedirs(self.system.save_path, exist_ok=True)
        
        # 自動設定device
        if self.system.device == "auto":
            import torch
            self.system.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__
        }
    
    def save(self, path: str):
        """保存配置到文件"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """從文件載入配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        return config


def get_default_config() -> HybridConfig:
    """獲取預設配置"""
    return HybridConfig()


def get_mvtec_config() -> HybridConfig:
    """獲取MVTec資料集的配置"""
    config = HybridConfig()
    
    # 調整為MVTec設定
    config.data.data_path = "./data/mvtec"
    config.data.dataset_name = "mvtec"
    config.system.save_path = "./results/mvtec"
    
    return config


def get_visa_config() -> HybridConfig:
    """獲取VisA資料集的配置"""
    config = HybridConfig()
    
    # 調整為VisA設定
    config.data.data_path = "./data/visa"
    config.data.dataset_name = "visa"
    config.system.save_path = "./results/visa"
    
    return config


def get_debug_config() -> HybridConfig:
    """獲取debug用的配置 (快速測試)"""
    config = HybridConfig()
    
    # 減少訓練時間的設定
    config.training.epochs = 2
    config.training.batch_size = 2
    config.training.save_freq = 1
    config.training.print_freq = 1
    
    # 關閉FP16避免potential issues
    config.model.use_fp16 = False
    
    # 簡化評估
    config.evaluation.metrics = "image-level"
    
    return config


if __name__ == "__main__":
    # 測試配置
    print("Testing configuration...")
    
    # 測試預設配置
    config = get_default_config()
    print("Default config created successfully")
    print(f"Model config: {config.model}")
    print(f"Training config: {config.training}")
    
    # 測試保存和載入
    config.save("test_config.json")
    loaded_config = HybridConfig.load("test_config.json")
    print("Config save/load test passed")
    
    # 清理測試檔案
    os.remove("test_config.json")
    
    print("✓ Configuration test passed!")