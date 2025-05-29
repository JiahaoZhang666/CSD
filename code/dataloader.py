import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image


class MultiFeatureDataset_train(Dataset):
    def __init__(self, jsonlines_path, image_dir, audio_features_dir, tokenizer, max_length=512):
        self.image_dir = image_dir
        self.audio_features_dir = audio_features_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.items = []
        with open(jsonlines_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    image_id = item.get("image_id")
                    annotator_id = item.get("annotator_id")
                    captions = item.get("caption")
                    image_path = f"{self.image_dir}/{image_id}.jpg"
                    audio_path = f"{self.audio_features_dir}/flickr30k_train_{str(image_id).zfill(16)}_{annotator_id}.npy"
                    if os.path.exists(image_path) and os.path.exists(audio_path):
                        self.items.append((image_id, annotator_id, captions))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_id, annotator_id, captions = self.items[idx]

        # Load and preprocess image
        image_path = f"{self.image_dir}/{image_id}.jpg"
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)

        # Tokenize text
        text_inputs = self.tokenizer.encode_plus(
            captions,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        text = {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0)
        }

        # Load audio features
        audio_path = f"{self.audio_features_dir}/flickr30k_train_{str(image_id).zfill(16)}_{annotator_id}.npy"
        audio_features = torch.tensor(np.load(audio_path), dtype=torch.float32)

        return {
            'image': image,
            'text': text,
            'audio': audio_features,
            'labels': idx
        }


class MultiFeatureDataset_test(Dataset):
    def __init__(self, jsonlines_path, image_dir, audio_features_dir, tokenizer, max_length=512):
        self.image_dir = image_dir
        self.audio_features_dir = audio_features_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.items = []
        with open(jsonlines_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    image_id = item.get("image_id")
                    annotator_id = item.get("annotator_id")
                    captions = item.get("caption")
                    image_path = f"{self.image_dir}/{image_id}.jpg"
                    audio_path = f"{self.audio_features_dir}/flickr30k_test_{str(image_id).zfill(16)}_{annotator_id}.npy"
                    if os.path.exists(image_path) and os.path.exists(audio_path):
                        self.items.append((image_id, annotator_id, captions))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_id, annotator_id, captions = self.items[idx]

        # Load and preprocess image
        image_path = f"{self.image_dir}/{image_id}.jpg"
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)

        # Tokenize text
        text_inputs = self.tokenizer.encode_plus(
            captions,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        text = {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0)
        }

        # Load audio features
        audio_path = f"{self.audio_features_dir}/flickr30k_test_{str(image_id).zfill(16)}_{annotator_id}.npy"
        audio_features = torch.tensor(np.load(audio_path), dtype=torch.float32)

        return {
            'image': image,
            'text': text,
            'audio': audio_features,
            'labels': idx
        }
