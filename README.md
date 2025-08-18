# Cross-Modal Retrieval with Cauchy-Schwarz Divergence

> We introduce **Cauchy–Schwarz (CS) divergence** as a stable, hyperparameter-free alternative to KL/MMD for cross-modal retrieval and propose **Generalized CS (GCS) divergence** that aligns **N modalities** jointly via a **bidirectional circular matching scheme**—no pairwise comparisons required.
> 

This project implements a multi-modal neural network that integrates image, text, and audio features. It supports contrastive learning and reconstruction objectives, designed for cross-modal retrieval tasks.

## 🚀 Highlights

- **CS divergence**: stable, **hyperparameter-free** distribution alignment for bi-modal retrieval.
- **GCS divergence**: **linearly scalable** joint alignment for **3+ modalities**
- **Plug-and-Play**: drop-in replacement for KL/MMD losses in existing frameworks (e.g., JFSE, LAVIMO, DRCL, CMPM).
- Works across **image / text / audio / video / motion / ...**

## Baseline Integration
### 🔹 Bi-Modal Retrieval (Image–Text)
 - Original Alignment: KL divergence
 - Ours: Replace KL with  with Cauchy–Schwarz (CS) divergence.
 - ✅ No other changes → significant performance gains.
### 🔹 Tri-Modal Retrieval (Video–Text–Motion / Image–Text–Audio)
 - Original Alignment: Pairwise KL divergence
 - Ours: Replace with Generalized CS (GCS) divergence.
 - ✅ Eliminates pairwise alignment
 - ✅ Supports 3+ modalities seamlessly

## Project Structure

├── data
├── dataloader.py
├── model.py
├── train.py
├── README.md 
└── requirements.txt


## 🧠 Model Overview

### 🔹 Modalities

- Video
- Text
- Motion

### 🔹 Fusion Strategy

- Cross-modal attention is used to enable inter-modal interaction.
- The fused representation is supervised by auxiliary tasks such as audio reconstruction to encourage semantic alignment and knowledge sharing across modalities.

---

## ⚙️ Environment Setup

```bash
conda create -n multimodal python=3.9
conda activate multimodal
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

This project uses the Flickr30k dataset. You can download it from:

Dataset: KITML [KITML]
video
text
motion

After downloading, organize the files into the following structure:

data/KITML/
├── KITML/
│   ├── images/              # Original image files (.jpg)
│   ├── audio_features/      # Pre-extracted audio features (.npy)
├── flickr30k_train_captions.jsonl
└── flickr30k_test_captions.jsonl


Note: Audio features should be pre-extracted using pretrained models such as Wav2Vec2, saved in .npy format. Filenames must align with image_id and annotator_id.

## 🚀 Getting Started

To start training:

```bash
python train.py
```
