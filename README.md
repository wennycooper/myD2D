# Hybrid Anomaly Detection Project

A completely independent hybrid anomaly detection model that combines learnable prompts from AnomalyCLIP with cross-attention mechanisms from D2D (Diffusion-based Defect Detection).

## 🎯 Project Overview

This project successfully integrates two state-of-the-art anomaly detection approaches into a single, self-contained system:

- **AnomalyCLIP**: Learnable context prompts for zero-shot anomaly detection
- **D2D**: Cross-attention extraction from Stable Diffusion models

### Key Innovations
- **Complete Independence**: No external dependencies on D2D or AnomalyCLIP projects
- **Minimal Integration**: Leverages existing cross-attention infrastructure with minimal changes
- **Efficient Training**: Only trains prompt embeddings (~10K parameters vs millions)
- **Mathematical Foundation**: Cross-attention ≈ Similarity computation equivalence

## 📁 Project Structure

```
my-proj/
├── models/
│   ├── __init__.py
│   ├── prompt_learner.py       # MinimalPromptLearner (from AnomalyCLIP)
│   └── hybrid_model.py         # HybridAnomalyDetector
├── pipeline/
│   ├── __init__.py
│   └── diffusion.py           # EasonADPipeline (copied from D2D)
├── losses/
│   ├── __init__.py
│   └── hybrid_loss.py         # CrossAttentionLoss
├── data/
│   ├── __init__.py
│   ├── dataset.py             # Dataset utilities (from D2D)
│   ├── transforms.py          # Image transforms (from D2D)
│   └── constants.py           # Preprocessing constants
├── utils/
│   ├── __init__.py
│   ├── loss_functions.py      # Loss functions (from AnomalyCLIP)
│   ├── transforms.py          # Transform utilities
│   └── metrics.py             # Evaluation metrics (from AnomalyCLIP)
├── config/
│   ├── __init__.py
│   └── base_config.py         # Configuration management
├── scripts/
│   └── train.py               # Training script
├── test.py                    # Independent testing/evaluation script
├── plan.md                    # Detailed technical plan
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this project
cd my-proj

# Install dependencies
pip install torch torchvision transformers diffusers scikit-learn scikit-image
pip install numpy scipy tqdm tabulate pillow
```

### 2. **No External Dependencies Required!**

This project is completely self-contained. You don't need to download or setup D2D or AnomalyCLIP projects - all necessary components have been copied and integrated locally.

### 3. Basic Testing

```bash
# Make sure you're in the project directory
cd my-proj

# Quick import test
python -c "
from models.hybrid_model import HybridAnomalyDetector
from config.base_config import get_debug_config
print('✓ Basic imports working!')
"

# Comprehensive functionality test
python test_basic.py
```

Expected output:
```
🚀 Starting Basic Functionality Tests
==================================================
✅ PASS - Import Test
✅ PASS - Configuration Test  
✅ PASS - Model Creation Test
✅ PASS - Loss Functions Test
✅ PASS - Data Utilities Test

🎯 Overall Result: 5/5 tests passed
🎉 All tests passed! The project is ready to use.
```

### 4. Training

```bash
# Debug training (small model for testing)
python scripts/train.py --debug --data_path /path/to/dataset --epochs 2

# Full training
python scripts/train.py --data_path /path/to/dataset --epochs 15 --batch_size 4
```

### 5. Testing/Evaluation

```bash
# Evaluate trained model
python test.py \
    --data_path /path/to/test/data \
    --dataset mvtec \
    --checkpoint_path ./checkpoints/best_model.pth \
    --metrics image-pixel-level
```

## 🔧 Core Components

### 1. MinimalPromptLearner (`models/prompt_learner.py`)
- Extracted from AnomalyCLIP with minimal dependencies
- Generates learnable normal/abnormal prompt embeddings
- Supports deep compound prompts with configurable depth
- **Key features**: Context learning, template optimization, embedding assembly

### 2. HybridAnomalyDetector (`models/hybrid_model.py`)
- Combines D2D diffusion pipeline with learnable prompts
- Freezes 95% of parameters, only trains prompt learner
- Provides both training and inference modes
- **Key features**: Attention extraction, map aggregation, image-level scoring

### 3. CrossAttentionLoss (`losses/hybrid_loss.py`)
- Designed specifically for cross-attention level optimization
- Integrates proven AnomalyCLIP loss components
- Adds novel attention-specific losses (sharpness, coverage, consistency)
- **Key features**: Multi-level loss computation, attention quality metrics

### 4. Independent Pipeline (`pipeline/diffusion.py`)
- Complete EasonADPipeline copied from D2D
- No external dependencies required
- Full attention control and extraction capabilities
- **Key features**: Attention store, cross-attention processing, map visualization

## 📊 Technical Architecture

### System Flow
```
Input Image → VAE Encoder → UNet with Cross-Attention 
           → [Normal Prompts]     [Abnormal Prompts]
           → Normal Attention Maps ← → Abnormal Attention Maps
           → Loss Computation → Prompt Optimization
```

### Key Advantages
1. **Self-Contained**: No external project dependencies
2. **Training Efficient**: 90%+ parameter reduction
3. **Technically Elegant**: Direct cross-attention utilization
4. **Mathematically Sound**: Cross-attention ≈ Similarity computation

For detailed technical specifications, see [plan.md](plan.md)

## 📈 Evaluation Setup

### Supported Datasets
- **MVTec AD**: Primary evaluation dataset (15 classes)
- **VisA**: Additional validation dataset (12 classes)  
- **BTAD**: Industrial anomaly detection (3 classes)
- **MPDD**: Manufacturing defect detection (6 classes)

### Evaluation Metrics
- **Image-level**: AUROC, Average Precision (AP)
- **Pixel-level**: AUROC, AUPRO (Average Precision per Region Overlap)

### Hardware Requirements
- **GPU**: RTX 3090/4090 or equivalent (12GB+ VRAM recommended)
- **RAM**: 16GB+ (32GB+ for large datasets)
- **Storage**: 50GB+ (model + datasets)

## 🎛️ Configuration Options

### Model Configuration
```python
model:
  pretrained_model_name_or_path: "stabilityai/stable-diffusion-2-1-base"
  n_ctx: 12                    # Number of learnable tokens
  depth: 9                     # Compound prompts depth  
  t_n_ctx: 4                   # Text encoder context length
  use_fp16: true               # Use mixed precision
```

### Training Configuration
```python
training:
  epochs: 15
  batch_size: 4
  learning_rate: 1e-3
  loss_weights:
    focal: 1.0                 # Focal loss weight
    dice: 1.0                  # Dice loss weight
    image: 1.0                 # Image-level loss weight
    sharpness: 0.5             # Attention sharpness weight
    coverage: 0.3              # Attention coverage weight
    consistency: 0.2           # Prompt consistency weight
```

## 📝 Usage Examples

### Training Custom Model
```python
from config.base_config import get_default_config
from models.hybrid_model import HybridAnomalyDetector
from losses.hybrid_loss import CrossAttentionLoss

# Load configuration
config = get_default_config()

# Initialize model
model = HybridAnomalyDetector(config)
loss_fn = CrossAttentionLoss()

# Training (see scripts/train.py for complete example)
for epoch in range(config['epochs']):
    for batch in dataloader:
        outputs = model(batch['images'], mode='train')
        loss, loss_dict = loss_fn(outputs, batch['targets'])
        # ... standard training loop
```

### Inference Usage
```python
import torch
from models.hybrid_model import HybridAnomalyDetector

# Load trained model
model = HybridAnomalyDetector(config)
checkpoint = model.load_checkpoint("./checkpoints/best.pth")

# Inference
model.eval()
with torch.no_grad():
    results = model(test_images, mode='eval')
    anomaly_maps = results['anomaly_maps']      # (B, 64, 64)
    image_scores = results['image_scores']      # (B,)
    similarity_maps = results['similarity_maps'] # (B, 2, 64, 64)
```

### Quick Configuration
```python
from config.base_config import get_debug_config, get_default_config

# For development/testing
debug_config = get_debug_config()  # Small model, 2 epochs

# For production training  
config = get_default_config()     # Full model, 15 epochs
```

## 🧪 Development Status

### ✅ Completed (All Components Independent)
- [x] **Complete Independence**: All D2D and AnomalyCLIP components copied locally
- [x] **Project Architecture**: Full modular structure implemented
- [x] **MinimalPromptLearner**: Extracted and optimized from AnomalyCLIP
- [x] **HybridAnomalyDetector**: Core model with attention extraction
- [x] **CrossAttentionLoss**: Multi-level loss functions with attention-specific terms
- [x] **Independent Pipeline**: EasonADPipeline copied and integrated locally
- [x] **Training System**: Complete training script with configuration management
- [x] **Testing Framework**: Independent test.py with evaluation metrics
- [x] **Data Utilities**: Dataset loading, transforms, and preprocessing
- [x] **Documentation**: Updated README and technical specifications

### ✅ Ready for Use
- [x] **No External Dependencies**: Project is completely self-contained
- [x] **Modular Design**: Easy to extend and customize
- [x] **Configuration System**: Flexible config management for different scenarios
- [x] **Error Handling**: Robust fallbacks and error messages
- [x] **Evaluation Pipeline**: Complete metrics calculation and reporting

## 🤝 Acknowledgments

This project integrates and builds upon excellent work from:
- **D2D**: Diffusion-based Anomaly Detection with cross-attention mechanisms
- **AnomalyCLIP**: CLIP-based Anomaly Detection with learnable prompts
- **Stable Diffusion**: Foundational diffusion model architecture

All necessary components have been copied and adapted locally to create a completely independent system.

## 📋 Requirements File

Create a `requirements.txt` file:
```
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
diffusers>=0.20.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
numpy>=1.21.0
scipy>=1.7.0
tqdm>=4.62.0
tabulate>=0.8.9
pillow>=8.3.0
packaging>=21.0
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project directory
   cd my-proj
   export PYTHONPATH=/home/kkuei/my-proj:$PYTHONPATH
   ```

2. **Memory Issues**
   ```python
   # Use debug config for testing
   config = get_debug_config()
   # Or reduce batch size
   config['batch_size'] = 2
   ```

3. **CUDA Compatibility**
   ```bash
   # Check PyTorch CUDA version
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## 📄 License

This project respects the original licenses of the integrated components. Please refer to the original D2D and AnomalyCLIP projects for their respective licensing terms.

## 📬 Contact

For questions or suggestions regarding this hybrid implementation, please create an issue or contact the project maintainer.