#!/usr/bin/env python3
"""
Basic Test Script for Hybrid Anomaly Detection
測試基本功能的腳本，不依賴完整的D2D和AnomalyCLIP環境
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# 添加專案路徑
sys.path.append('/home/kkuei/my-proj')

def test_config():
    """測試配置模組"""
    print("Testing configuration module...")
    
    try:
        from config.base_config import get_default_config, get_debug_config
        
        # 測試預設配置
        config = get_default_config()
        print(f"✓ Default config loaded")
        print(f"  Model n_ctx: {config.model.n_ctx}")
        print(f"  Training epochs: {config.training.epochs}")
        print(f"  Data path: {config.data.data_path}")
        
        # 測試debug配置
        debug_config = get_debug_config()
        print(f"✓ Debug config loaded")
        print(f"  Debug epochs: {debug_config.training.epochs}")
        
        # 測試保存和載入
        test_path = "/tmp/test_config.json"
        config.save(test_path)
        loaded_config = config.load(test_path)
        print("✓ Config save/load test passed")
        
        # 清理
        os.remove(test_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_prompt_learner():
    """測試prompt learner模組"""
    print("\nTesting prompt learner module...")
    
    try:
        from models.prompt_learner import MinimalPromptLearner
        
        # 創建mock text encoder
        class MockTextEncoder:
            def __init__(self):
                self.config = type('Config', (), {'hidden_size': 768})()
                self.tokenizer = self.MockTokenizer()
                
            class MockTokenizer:
                def __call__(self, text, **kwargs):
                    return type('TokenizerOutput', (), {
                        'input_ids': torch.randint(0, 1000, (1, 77))
                    })()
                    
            def get_input_embeddings(self):
                return nn.Embedding(1000, 768)
        
        # 測試prompt learner
        mock_encoder = MockTextEncoder()
        design_details = {
            "Prompt_length": 12,
            "learnabel_text_embedding_depth": 9,
            "learnabel_text_embedding_length": 4
        }
        
        prompt_learner = MinimalPromptLearner(mock_encoder, design_details)
        
        # 測試forward pass
        normal_prompt, abnormal_prompt, compound_prompts = prompt_learner()
        
        print(f"✓ Prompt learner created")
        print(f"  Normal prompt shape: {normal_prompt.shape}")
        print(f"  Abnormal prompt shape: {abnormal_prompt.shape}")
        print(f"  Compound prompts: {len(compound_prompts)}")
        
        # 測試參數獲取
        params = prompt_learner.get_learnable_parameters()
        total_params = sum(p.numel() for p in params)
        print(f"  Learnable parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt learner test failed: {e}")
        return False


def test_loss_functions():
    """測試loss函數"""
    print("\nTesting loss functions...")
    
    try:
        from losses.hybrid_loss import CrossAttentionLoss
        
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
        
        # 測試loss計算
        loss_fn = CrossAttentionLoss()
        total_loss, loss_dict = loss_fn(predictions, targets)
        
        print(f"✓ Loss functions working")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Loss components: {len(loss_dict)}")
        
        # 測試backward pass
        total_loss.backward()
        print("✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False


def test_model_components():
    """測試模型組件 (不需要完整D2D環境)"""
    print("\nTesting model components...")
    
    try:
        # 只測試prompt learner部分
        from models.prompt_learner import MinimalPromptLearner
        
        # Mock text encoder
        class MockTextEncoder:
            def __init__(self):
                self.config = type('Config', (), {'hidden_size': 768})()
                self.tokenizer = self.MockTokenizer()
                
            class MockTokenizer:
                def __call__(self, text, **kwargs):
                    return type('TokenizerOutput', (), {
                        'input_ids': torch.randint(0, 1000, (1, 77))
                    })()
                    
            def get_input_embeddings(self):
                return nn.Embedding(1000, 768)
        
        mock_encoder = MockTextEncoder()
        prompt_learner = MinimalPromptLearner(mock_encoder, {
            "Prompt_length": 12,
            "learnabel_text_embedding_depth": 9,
            "learnabel_text_embedding_length": 4
        })
        
        # 測試訓練模式
        prompt_learner.train()
        normal_prompt, abnormal_prompt, compound_prompts = prompt_learner()
        
        # 測試optimizer能夠工作
        optimizer = torch.optim.Adam(prompt_learner.get_learnable_parameters(), lr=1e-3)
        
        # 創建dummy loss並測試backward
        dummy_loss = torch.sum(normal_prompt) + torch.sum(abnormal_prompt)
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()
        
        print("✓ Model components basic test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        return False


def test_training_script():
    """測試訓練腳本的基本import"""
    print("\nTesting training script imports...")
    
    try:
        # 嘗試import training相關模組
        from config.base_config import get_debug_config
        from losses.hybrid_loss import CrossAttentionLoss
        
        # 創建debug配置
        config = get_debug_config()
        print(f"✓ Training configuration ready")
        print(f"  Debug epochs: {config.training.epochs}")
        print(f"  Batch size: {config.training.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False


def main():
    """運行所有基本測試"""
    print("=== Hybrid Anomaly Detection Basic Tests ===")
    
    tests = [
        ("Configuration", test_config),
        ("Prompt Learner", test_prompt_learner),
        ("Loss Functions", test_loss_functions),
        ("Model Components", test_model_components),
        ("Training Script", test_training_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        success = test_func()
        results.append((test_name, success))
    
    # 總結結果
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All basic tests passed! Ready for full implementation.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the issues above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)