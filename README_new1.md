# Neural Texture Compression

> Real-time PBR Texture Compression with Convolutional AutoEncoders

![Project Banner](./figures/results_preview.png)

## 📌 Abstract

Modern video games and real-time applications demand photorealistic rendering using high-resolution physically based rendering (PBR) materials. These materials—color (albedo), normal, roughness maps—occupy significant storage and VRAM. This project introduces a neural texture compression framework using convolutional autoencoders, achieving a **4× compression ratio** with minimal perceptual degradation, optimized for real-time performance.

---

## 🧠 Problem Statement

Traditional texture formats (e.g., JPEG, PNG) reduce storage size but are not optimized for GPU pipeline integration. As games scale up to 4K and beyond, storing and accessing uncompressed multi-channel textures becomes a bottleneck. This work explores **learned latent representations** for texture data that can be decoded efficiently either on load or at runtime.

---

## 🧩 Inference Modes

We evaluate two neural decoding modes:

| Mode               | VRAM Usage     | Compute Overhead   | Description |
|--------------------|----------------|---------------------|-------------|
| **Inference-on-Load** | High (decoded texture stored) | Low (decode once) | Decode once, use like normal texture |
| **Inference-on-Sample** | Low (latent only) | High (decode every frame) | Runtime shader-based neural decoding |

> This implementation uses the **inference-on-load** strategy.

---

## 🧬 Architecture

Our model is a symmetric **convolutional autoencoder**:

- **Encoder**: 3 × Conv + ReLU + MaxPooling
- **Bottleneck**: Fully connected latent layer (dim = 128)
- **Decoder**: Transposed convolutions + upsampling

This architecture significantly reduces the number of parameters compared to fully connected networks, while preserving essential visual structures.

![Autoencoder Architecture](./figures/Encoder_new.png)

---

## 🛠️ Training Details

- **Dataset**: [HuggingFace: dream-textures/color-normal-1k](https://huggingface.co/datasets/dream-textures/textures-color-normal-1k)
- **Image Resolution**: 512 × 512
- **Channels**: RGB + Normal maps
- **Batch Size**: 16
- **Learning Rate**: 0.001
- **Latent Dimension**: 128

### 🧮 Hybrid Loss Function

Weighted composite loss:
- `0.75 × L1`
- `0.20 × MSE`
- `0.05 × MS-SSIM`
- `0.00 × SSIM`

> This combination balances pixel accuracy and perceptual structure.

---

## 📉 Loss Curve

Training loss across 50 epochs shows stable convergence:

![Loss Curves](./figures/loss_by_epoch.png)

---

## 📊 Results

Color and normal maps are reconstructed with high perceptual fidelity:

| Texture Type | Original | Reconstruction |
|--------------|----------|----------------|
| Color Map    | ![](./figures/sample_color_gt.png) | ![](./figures/sample_color_pred.png) |
| Normal Map   | ![](./figures/sample_normal_gt.png) | ![](./figures/sample_normal_pred.png) |

> Achieved ~4× compression with PSNR > 30 and perceptually stable SSIM scores.

---

## 🧪 Usage

To train and evaluate the model:

```bash
# Train the model
python train.py

# Evaluate trained model
python eval.py
