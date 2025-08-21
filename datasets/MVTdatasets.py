


import os
import json
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MVTDataset(Dataset):
    def __init__(self, root: str, json_file: str, max_motion=500, max_video=50):
        self.root = root
        self.max_motion = max_motion
        self.max_video = max_video

        with open(os.path.join(root, json_file), 'r') as f:
            self.items = pd.DataFrame(json.load(f)).T.reset_index(drop=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items.loc[idx]

        motion = self._load_motion(item)
        video = self._load_video(item)
        text = random.choice([a['text'] for a in item['annotations']]) if item['annotations'] else ""
        label = idx

        motion = motion[:self.max_motion]            # (T, J, 3)
        video = video[:self.max_video]              # (N, 512, 512, 3)
        motion = np.transpose(motion, (1, 2, 0))     # → (J, 3, T)

        return torch.tensor(motion, dtype=torch.float32), \
               torch.tensor(video, dtype=torch.uint8), \
               text, \
               torch.tensor(label, dtype=torch.long)

    def _load_motion(self, item):
        try:
            arr = np.load(os.path.join(self.root, item['path'] + '.npz'))
            return arr['poses'].reshape(-1, 24, 3)
        except:
            return np.zeros((1, 24, 3), dtype=np.float32)

    def _load_video(self, item):
        
        path = item['path'].replace('KIT', 'KITrender').replace('_poses', '.mp4')
        cap = cv2.VideoCapture(os.path.join(self.root, path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (512, 512))
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            return np.zeros((1, 512, 512, 3), dtype=np.uint8)

        return np.array(frames, dtype=np.uint8)
        


def collate_fn(batch):
    motions, videos, texts, labels = zip(*batch)
    return (
        torch.stack(motions),
        torch.stack(videos),
        list(texts),
        torch.stack(labels)
    )


def get_dataloaders(root='data', train_json='kitml-train.json', val_json='kitml-val.json',
                    batch_size=4, num_workers=4):
    split_train_val('data/kitml.json', 'data/kitml-train.json', 'data/kitml-val.json')
    
    train_set = MVTDataset(root, train_json)
    val_set = MVTDataset(root, val_json)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader


def split_train_val(json_path, train_path, val_path, train_ratio=0.7, seed=42):
    with open(json_path, 'r') as f:
        data = json.load(f)

    keys = list(data.keys())
    random.Random(seed).shuffle(keys)
    split = int(len(keys) * train_ratio)

    with open(train_path, 'w') as f:
        json.dump({k: data[k] for k in keys[:split]}, f)
    with open(val_path, 'w') as f:
        json.dump({k: data[k] for k in keys[split:]}, f)


split_train_val('data/kitml.json', 'data/kitml-train.json', 'data/kitml-val.json')