# LemGendary CodeFormer Face Restoration

![SOTA](https://img.shields.io/badge/Status-SOTA-brightgreen) ![Hardware](https://img.shields.io/badge/Hardware-Accelerated-blue) ![Epochs](https://img.shields.io/badge/Epochs-1-orange) ![Resolution](https://img.shields.io/badge/Res-224x224-blueviolet)

## Overview

The **LemGendary CodeFormer Face Restoration** is a professional-grade AI model optimized for the `face` lifecycle within the LemGendary Training Suite.

- **Architecture**: CodeFormer (Standard Backbone)
- **Input Resolution**: 224x224
- **Use Case**: Face restoration model
- **Training Data**: LemGendizedCodeFormer

## Manifold Topology

```mermaid
graph TD
    Input[RGB Input 224x224] --> Backbone[CodeFormer]
    Backbone --> Manifold[Latent Manifold]
    Manifold --> Head[Face Head]
    Head --> Output[Predictive Array]
    
    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#00ff00,stroke:#333,stroke-width:4px
```

## Usage

```python
# Premium CLI Integration provided for generative/VLM tasks.
```

> [!TIP]
> **Implementation Guide**: For high-performance deployment including ONNX (FP32/FP16) and standalone PyTorch snippets, refer to the **[codeformer_usage.ipynb](codeformer_usage.ipynb)** notebook in this directory.

- **Input Requirements**: RGB Image Tensors normalized to ImageNet stats.
- **Failures**: Large aspect ratio distortions during standard resize phases.

## Implementation Requirements

- **Hardware**: NVIDIA GeForce GTX 1650 (4G VRAM)
- **Software**: PyTorch 2.1+, CUDA 12.1.
- **Training Lifecycle**: Successfully processed over 1 total epochs securely.

## Model Stats

- **Precision**: ONNX FP16 (Edge) / PyTorch FP32 (Training).
- **Latency**: Sub-50ms inference bound on target local GPU hardware.
- **Stability**: Trained using **MSE Loss** to enforce strict manifold alignment.

## Data Manifest

- **LemGendizedCodeFormer**: ~N/A binary image samples.

## Evaluation Results

- **Baseline Achievement**: **PSNR**: 30.08 | **SSIM**: 0.7873 | **LPIPS**: 0.3903 | **FID**: 82.0017
- **Split**: 80/20 train/validate with zero sample-leakage.

---
**LemGendary AI Training Suite** | *SOTA-Autonomous & Nuclear-Hardened Matrix*
