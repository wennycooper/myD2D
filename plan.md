# 基於D2D的Learnable Prompt異常檢測方案

## 專案概述

本專案旨在結合D2D (Diffusion-based Anomaly Detection) 與 AnomalyCLIP 的優勢，創建一個更強大的圖像缺陷檢測系統。核心理念是利用現成的D2D cross-attention機制，加入最小化的AnomalyCLIP learnable prompt，在cross-attention level設計loss來優化prompt embeddings。

## 核心設計理念

- **主體架構**: 利用現成的D2D cross-attention機制  
- **核心創新**: 加入最小化的AnomalyCLIP learnable prompt
- **優化目標**: 在cross-attention level設計loss來優化prompt embeddings

## 技術方案詳細設計

### 1. D2D Cross Attention機制分析

#### 資料流程:
```python
# UNet Forward過程中：
latent_input: (batch, 4, 64, 64)  
text_embeddings: (batch, 77, 768)

# 在UNet的各層cross-attention中：
query = from latent features: (batch*heads, spatial_tokens, head_dim)
key/value = from text embeddings: (batch*heads, 77, head_dim) 

# P2PCrossAttnProcessor captures:
attention_probs: (batch*heads, spatial_tokens, 77)
# 例如: (8, 4096, 77) for 64x64 resolution
```

#### AttentionStore收集機制:
```python
attention_store = {
    "down_cross": [layer1_attn, layer2_attn, ...],  # 下採樣階段
    "mid_cross": [...],                             # 中間階段
    "up_cross": [...]                               # 上採樣階段
}
```

### 2. 最小化AnomalyCLIP Prompt Learner整合

#### 核心組件設計:
```python
class MinimalPromptLearner(nn.Module):
    def __init__(self, text_encoder, design_details):
        super().__init__()
        
        # 基本設定
        self.n_ctx = design_details["Prompt_length"]  # 12
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]  # 4
        
        # Learnable context vectors
        ctx_dim = text_encoder.config.hidden_size  # 768
        self.ctx_normal = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        self.ctx_abnormal = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        
        # Deep compound prompts
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
            for _ in range(self.compound_prompts_depth - 1)
        ])
```

#### 關鍵特點:
- **直接產生embeddings**: learned prompts在embedding space，無法轉回text
- **最小參數量**: 只包含essential的learnable components
- **相容D2D**: 輸出格式直接可feed到D2D pipeline

### 3. Cross Attention Level的Loss Function

#### 核心Loss組件:
```python
class CrossAttentionLoss(nn.Module):
    def forward(self, predictions, targets):
        normal_attention = predictions['normal_attention']     # (batch, H, W)
        abnormal_attention = predictions['abnormal_attention'] # (batch, H, W)
        
        # 1. 組合成similarity-like format
        attention_stack = torch.stack([normal_attention, abnormal_attention], dim=1)
        
        # 2. Pixel-level losses (直接用AnomalyCLIP的loss！)
        focal_loss = self.focal_loss(attention_stack, gt_masks)
        dice_abnormal = self.dice_loss(abnormal_attention, gt_masks)
        dice_normal = self.dice_loss(normal_attention, 1 - gt_masks)
        
        # 3. Image-level loss (基於attention statistics)
        image_logits = self.compute_image_level_logits(normal_attention, abnormal_attention)
        image_loss = self.ce_loss(image_logits, gt_labels)
        
        # 4. Cross-attention specific losses
        sharpness_loss = self.compute_sharpness_loss(abnormal_attention, gt_masks)
        coverage_loss = self.compute_coverage_loss(abnormal_attention, gt_masks)
        
        return focal_loss + dice_loss + image_loss + sharpness_loss + coverage_loss
```

#### Loss設計特點:
- **重用AnomalyCLIP loss**: 直接應用proven的loss functions
- **Attention-specific losses**: 針對cross-attention特性設計的額外loss
- **End-to-end optimization**: 梯度直接flow回prompt parameters

### 4. 整合架構設計

```python
class HybridAnomalyDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # D2D pipeline (凍結所有參數)
        self.pipeline = EasonADPipeline.from_pretrained(config.sd_model_path)
        for param in self.pipeline.parameters():
            param.requires_grad = False
            
        # 最小化prompt learner (只有這部分會被訓練)
        self.prompt_learner = MinimalPromptLearner(
            self.pipeline.text_encoder, config.prompt_params
        )
        
    def forward(self, images):
        # 1. 生成learnable prompt embeddings
        normal_embeds, abnormal_embeds, compound_prompts = self.prompt_learner()
        
        # 2. 用normal prompt跑D2D獲取attention
        normal_attention = self.extract_attention_with_embeds(images, normal_embeds)
        
        # 3. 用abnormal prompt跑D2D獲取attention  
        abnormal_attention = self.extract_attention_with_embeds(images, abnormal_embeds)
        
        return {
            'normal_attention': normal_attention,
            'abnormal_attention': abnormal_attention
        }
```

## 訓練與測試流程

### 訓練流程

#### 訓練設置:
- **凍結參數**: 所有D2D parameters (VAE, UNet, Text Encoder)
- **訓練參數**: 只有prompt learner的parameters (~50MB)
- **Optimizer**: Adam, lr=1e-3
- **記憶體需求**: 大幅降低，約2-3GB

#### 訓練步驟:
1. **Forward**: 生成learnable embeddings → D2D attention extraction
2. **Loss計算**: Cross-attention maps + ground truth → loss
3. **Backward**: 梯度只flow回prompt learner parameters
4. **Update**: 優化prompt embeddings

### 測試評估流程

#### 測試腳本設計 (test.py):
基於D2D原有的test.py架構，整合混合模型的測試流程：

```python
class HybridTestPipeline:
    def __init__(self, checkpoint_path, config):
        # 載入D2D pipeline (凍結)
        self.pipeline = EasonADPipeline.from_pretrained(config.sd_model_path)
        
        # 載入訓練好的prompt learner
        self.prompt_learner = MinimalPromptLearner(config)
        self.prompt_learner.load_state_dict(torch.load(checkpoint_path))
        
    def test_single_image(self, image):
        # 1. 生成learned prompt embeddings
        normal_embeds, abnormal_embeds = self.prompt_learner()
        
        # 2. 提取normal/abnormal attention maps  
        normal_attention = self.extract_attention(image, normal_embeds)
        abnormal_attention = self.extract_attention(image, abnormal_embeds)
        
        # 3. 計算異常分數和anomaly map
        image_score = self.compute_image_score(normal_attention, abnormal_attention)
        anomaly_map = abnormal_attention
        
        return image_score, anomaly_map
```

#### 評估指標:
- **Image-level metrics**: AUROC, Average Precision
- **Pixel-level metrics**: AUROC, AUPRO
- **統計分析**: 每個object category的詳細結果
- **結果格式**: 與D2D相同的表格輸出格式

#### 測試流程:
1. **模型載入**: 載入預訓練D2D + 訓練好的prompt learner
2. **資料處理**: 使用相同的dataset和transform pipeline
3. **批次推理**: 
   - 對每張測試圖像運行混合模型
   - 提取normal/abnormal attention maps
   - 計算image-level和pixel-level預測
4. **指標計算**: 
   - 聚合所有預測結果
   - 計算AUROC, AUPRO, AP等指標
   - 生成per-category和overall結果表格
5. **結果輸出**: 
   - 標準化的性能報告
   - 可選的視覺化輸出
   - 詳細結果數據保存

#### 與原D2D test.py的差異:
- **模型架構**: 原D2D → 混合模型 (D2D + learnable prompts)
- **Attention提取**: 固定prompt → learned prompt embeddings
- **分數計算**: 單一prompt → normal/abnormal prompt對比
- **評估指標**: 保持相同的評估框架和指標

#### 推理優化:
```python
def inference_single_image(model, test_image):
    with torch.no_grad():
        # 生成learned embeddings
        normal_embeds, abnormal_embeds = model.prompt_learner()
        
        # 並行提取attention (可選優化)
        attention_results = model.parallel_attention_extraction(
            test_image, [normal_embeds, abnormal_embeds]
        )
        
        # 計算最終預測
        image_pred = compute_image_level_score(attention_results)
        anomaly_map = attention_results['abnormal_attention']
        
    return {
        'image_prediction': image_pred,
        'anomaly_map': anomaly_map,
        'confidence': compute_confidence_score(attention_results)
    }
```

## 方案優勢

### 1. 最小化修改
- **95% 現有代碼**: 主要利用D2D existing pipeline
- **5% 新增代碼**: 只需minimal prompt learner
- **無複雜轉換**: 不需要feature projection或維度轉換

### 2. 技術優雅  
- **Natural integration**: Cross-attention本質就是similarity computation
- **Direct embedding feed**: Learnable prompts直接輸入text encoder
- **Attention = Anomaly maps**: 輸出直接作為異常圖

### 3. 訓練高效
- **參數量小**: 只訓練prompt parameters (vs. 完整model)
- **記憶體友好**: 大幅降低GPU記憶體需求
- **收斂快速**: 小parameter space，容易優化

### 4. 理論基礎
- **數學等價**: `spatial_tokens × text_tokens` ≈ `patch_features @ text_features`
- **Proven components**: 結合兩個已驗證的方法
- **清晰目標**: 學習optimal prompt for anomaly detection

## 預期成果

### 性能提升:
- **更佳Prompt**: 學習到optimal normal/abnormal描述
- **更強特徵**: 利用SD VAE的powerful visual representation  
- **更高效率**: 訓練成本大幅降低
- **更好泛化**: 結合兩者優勢

### 技術創新:
- **首次結合**: D2D + AnomalyCLIP learnable prompts
- **新穎loss design**: Cross-attention level的prompt optimization
- **Efficient training**: 凍結大模型，只訓練prompts

## 實施計劃

### Phase 1: 基礎實現 (Week 1-2)
- [ ] 實作 MinimalPromptLearner
- [ ] 修改 EasonADPipeline 支援 prompt_embeds
- [ ] 基礎 attention extraction 功能
- [ ] 建立基本的模組化程式碼架構

### Phase 2: Loss Function設計 (Week 3)
- [ ] 實作 CrossAttentionLoss
- [ ] 整合 AnomalyCLIP loss components
- [ ] 設計 attention-specific losses
- [ ] 驗證loss function的梯度流

### Phase 3: 訓練流程實現 (Week 4)
- [ ] 完整訓練流程實現 (train.py)
- [ ] 訓練監控和checkpoint機制
- [ ] 初步訓練驗證
- [ ] 訓練穩定性調優

### Phase 4: 測試評估系統 (Week 5)
- [ ] 實作測試腳本 (test.py)
- [ ] 整合評估指標計算 (metrics.py)
- [ ] 視覺化工具 (visualization.py)
- [ ] 與原D2D結果對比驗證

### Phase 5: 完整驗證與優化 (Week 6-7)
- [ ] 在MVTec dataset完整測試
- [ ] 性能對比分析 (vs D2D, vs AnomalyCLIP)
- [ ] 超參數調優
- [ ] 多資料集驗證 (VisA等)

### Phase 6: 最終整合與文檔 (Week 8)
- [ ] 程式碼最終整理和文檔
- [ ] 完整的使用說明和範例
- [ ] 效能優化和部署準備
- [ ] 實驗結果總結報告

## 技術風險與對策

### 主要風險:
1. **Attention quality**: Cross-attention maps品質是否足夠好
2. **Prompt optimization**: 是否能學到meaningful prompts
3. **Training stability**: 小parameter space的訓練穩定性

### 對策:
1. **Baseline comparison**: 與原D2D和AnomalyCLIP詳細對比
2. **Ablation studies**: 驗證各component的貢獻
3. **Gradual training**: 從簡單case開始，逐步增加複雜度

## 資源需求

### 硬體需求:
- **GPU**: RTX 3090/4090 或同等級 (12GB+ VRAM)
- **RAM**: 32GB+
- **Storage**: 100GB+ (模型+資料集)

### 軟體環境:
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **Diffusers**: Latest
- **其他**: transformers, accelerate, xformers

### 資料集:
- **MVTec AD**: 主要測試資料集
- **VisA**: 額外驗證資料集
- **自定義資料**: 特定domain測試

## 成功指標

### 量化指標:
- **Pixel AUROC**: > 95% on MVTec
- **Image AUROC**: > 90% on MVTec  
- **Training time**: < 原AnomalyCLIP 50%
- **Memory usage**: < 原方法 60%
- **Test inference speed**: 與原D2D相當或更快

### 質化指標:
- **Attention quality**: 視覺化驗證attention maps合理性
- **Prompt meaningfulness**: 學習到的prompts是否有語義意義
- **Generalization**: 在不同defect types上的表現
- **Code maintainability**: 模組化設計和程式碼品質

### 測試評估標準:
- **與D2D對比**: 使用相同測試集和評估指標
- **與AnomalyCLIP對比**: 在相同實驗設定下比較
- **Ablation studies**: 驗證各組件的貢獻度
- **統計顯著性**: 多次實驗的穩定性分析
- **測試腳本相容性**: 與原D2D test.py架構保持一致

## 結論

本方案通過minimal modification of existing excellent work (D2D + AnomalyCLIP)，創造出一個more powerful, more efficient的異常檢測系統。核心創新在於將prompt learning帶入diffusion-based anomaly detection，並設計合適的loss functions來優化cross-attention level的prompt embeddings。

技術方案既practical又innovative，充分利用現有優秀工作的基礎，通過最小化修改達到最大效果。預期能在保持或提升檢測精度的同時，大幅降低訓練成本和複雜度。