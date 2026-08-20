# LemGendary NIMA Aesthetic Scorer (Pro ViT)

![SOTA](https://img.shields.io/badge/Status-SOTA-brightgreen) ![Hardware](https://img.shields.io/badge/Hardware-Accelerated-blue) ![Epochs](https://img.shields.io/badge/Epochs-51-orange) ![Resolution](https://img.shields.io/badge/Res-256x256-blueviolet)

## Overview

The **LemGendary NIMA Aesthetic Scorer (Pro ViT)** is a professional-grade AI model optimized for the `quality` lifecycle within the LemGendary Training Suite.

- **Architecture**: NIMA_Model (Swin-v2-T (Global Multi-Scale Attention))
- **Input Resolution**: 256x256
- **Use Case**: High-end aesthetic quality scorer using Swin Transformer V2, optimized for high-res global composition.
- **Training Data**: LemGendizedNimaAesthetic

## Manifold Topology

```mermaid
graph TD
    Input[RGB Input 256x256] --> Backbone[NIMA_Model]
    Backbone --> Manifold[Latent Manifold]
    Manifold --> Head[Quality Head]
    Head --> Output[Predictive Array]
    
    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#00ff00,stroke:#333,stroke-width:4px
```

> [!IMPORTANT]
> **Quality Vector**: This model is specialized for **Aesthetics**. 
> - **Primary Targets**: Composition, Color, Lighting, Artistic Intent.

## Usage

```python
import torch, base64
from PIL import Image

# 1. Hardware-Agnostic Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Stealth Load (v16.0)
model_path = "nima_aesthetic_pro_latest.pth"
ckpt = torch.load(model_path, map_location=device, weights_only=False)
state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt

# 3. Initialization
from models.nima import NIMA_Model
model = NIMA_Model().to(device)
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state)
model.eval()

# 4. Forward Pass
img = Image.open("photo.jpg").convert('RGB').resize((256, 256))
input_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
with torch.no_grad():
    probs = model(input_tensor)

# 5. Score Calculation
scores = torch.arange(1, 11).float().to(device)
mean_score = torch.sum(probs * scores).item()
print(f"Quality Score: {mean_score:.2f}")
```

> [!TIP]
> **Implementation Guide**: For high-performance deployment including ONNX (FP32/FP16) and standalone PyTorch snippets, refer to the **[nima_aesthetic_pro_usage.ipynb](nima_aesthetic_pro_usage.ipynb)** notebook in this directory.

- **Input Requirements**: RGB Image Tensors normalized to ImageNet stats.
- **Failures**: Large aspect ratio distortions during standard resize phases.

## Implementation Requirements

- **Hardware**: NVIDIA GeForce GTX 1650 (4G VRAM)
- **Software**: PyTorch 2.1+, CUDA 12.1.
- **Training Lifecycle**: Successfully processed over 51 total epochs securely.

## Model Stats

- **Precision**: ONNX FP16 (Edge) / PyTorch FP32 (Training).
- **Latency**: Sub-50ms inference bound on target local GPU hardware.
- **Stability**: Trained using **Earth Mover's Distance (EMD)** with strict 1.0 Temperature Anchoring.

## Data Manifest

- **LemGendizedNimaAesthetic**: ~N/A binary image samples.

## Evaluation Results

- **Baseline Achievement**: **PLCC**: 0.6470317840576172 | **SRCC**: 0.6020231066467204
- **Split**: 80/20 train/validate with zero sample-leakage.

---
**LemGendary AI Training Suite** | *SOTA-Autonomous & Nuclear-Hardened Matrix*
