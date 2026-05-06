# LemGendary Model Hub (v2026 SOTA)

Welcome to the official repository for high-fidelity pretrained models from the LemGendary AI Training Suite.

## Repository Structure

Each model directory contains a standardized triple-matrix deployment:

1.  **`LemGendary[Model].pt`**: Standalone PyTorch model (Architecture + Weights). Ideal for research and further training.
2.  **`LemGendary[Model]_FP32.onnx`**: High-precision ONNX matrix with external `.data` weighting. Designed for desktop and high-accuracy CPU inference.
3.  **`LemGendary[Model].onnx`**: Production-ready FP16 ONNX matrix with embedded weights. Optimized for **WebGPU**, mobile, and low-latency edge deployment.
4.  **`[model]_usage.ipynb`**: Interactive implementation guide with ready-to-run code snippets for all formats.
5.  **`metrics.csv`**: Full audit trail of the training process, including epoch-by-epoch SOTA progress.

## Quick Start

To use a model, navigate to its directory and open the `_usage.ipynb` notebook. It provides snippets for:
- PyTorch (FP32)
- ONNX Runtime (FP32 & FP16)

## Training Suite

These models are trained and maintained using the [LemGendary Training Suite](https://github.com/lemgenda/lemgendary-training-suite).

---
*© 2026 LemGendary AI - SOTA Matrix Manifold Management*
