# Multi-Modal Feature Learning with Image, Text, and Audio

This project implements a multi-modal neural network that integrates image, text, and audio features. It supports contrastive learning and reconstruction objectives, designed for cross-modal retrieval tasks.

## Project Structure

├── data
├── dataloader.py
├── model.py
├── train.py
├── README.md 
└── requirements.txt


## 🧠 Model Overview

### 🔹 Modalities

- Image: Visual embeddings are extracted using CLIP ViT-B/32 [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) 
- Text: Semantic representations are obtained via DistilBERT. [DistilBERT](https://huggingface.co/distilbert-base-uncased) 
- Audio: Pre-extracted features (e.g., from Wav2Vec2 or VGGish) are linearly projected into a shared embedding space.[Wav2vec2.0](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20.)

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

Dataset: Flickr30k (https://google.github.io/localized-narratives/index.html)

After downloading, organize the files into the following structure:

data/Flickr30k/
├── Flickr30k/
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