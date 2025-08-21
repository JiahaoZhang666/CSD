import os
import time
import itertools
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from datasets import get_dataloaders
from metric import Loss
from model import MultiModalModel as Model

import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
# save_dir = Path('./TSNE-visualize-with-multi-color-on-testset')
# save_dir.mkdir(parents=True, exist_ok=True)  # 确保路径存在


def tsne_(img_feats, txt_feats, aud_feats, fus_feats, labels=None, name=None):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    img_2d = tsne.fit_transform(img_feats)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    txt_2d = tsne.fit_transform(txt_feats)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    aud_2d = tsne.fit_transform(aud_feats)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    fus_2d = tsne.fit_transform(fus_feats)
    plt.figure(figsize=(8, 6))

    plt.scatter(img_2d[:, 0], img_2d[:, 1], c='r', label='Image', alpha=0.5, marker='o', s=20)
    plt.scatter(txt_2d[:, 0], txt_2d[:, 1], c='b', label='Text', alpha=0.5, marker='x', s=20)
    plt.scatter(aud_2d[:, 0], aud_2d[:, 1], c='g', label='Audio', alpha=0.5, marker='^', s=20)
    plt.scatter(fus_2d[:, 0], fus_2d[:, 1], c='orange', label='Fusion', alpha=0.5, marker='s', s=20)

    plt.legend(loc='lower left', fontsize=12, frameon=True)
    plt.savefig(save_dir / f'TSNE-{name}-multi-color.png')
    plt.close()


def setup_logger(log_file="train.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


class Config:
    def __init__(self):
        self.video_dim = 512
        self.text_dim = 768
        self.motion_dim = 512
        self.num_classes = 2051
        self.resume = False
        self.feature_size = 512
        self.embed_dim = 512
        self.num_heads = 8
        self.tau = 0.05
        self.lr = 1e-4
        self.batch_accum = 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model, compute_loss, train_loader, optimizer, scheduler, config, logger, epoch
):
    model.train()
    compute_loss.train()
    device = config.device

    total_loss, video_precision, text_precision, motion_precision = 0, 0, 0, 0
    video_list, text_list, motion_list, recon_motion_list, label_list = [], [], [], [], []
    loss_count = 0

    for step, (motions, videos, texts, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        motions, videos, labels = motions.to(device), videos.to(device), labels.to(device)
        bs, _, _, nframes = motions.shape
        lengths = torch.full((bs,), nframes, device=device)
        mask = torch.ones((bs, nframes), dtype=torch.bool, device=device)
        batch = {'x': motions, 'mask': mask, 'lengths': lengths, 'y': torch.zeros(bs, dtype=torch.long).to(device)}

        # 前向提取特征
        v_embed, t_embed, m_embed, recon_m_embed = model(batch, videos, texts)

        video_list.append(v_embed)
        text_list.append(t_embed)
        motion_list.append(m_embed)
        recon_motion_list.append(recon_m_embed)
        label_list.append(labels)

        if step % config.batch_accum == 0 and step != 0:
            v_all = torch.cat(video_list, dim=0)
            t_all = torch.cat(text_list, dim=0)
            m_all = torch.cat(motion_list, dim=0)
            recon_m_all = torch.cat(recon_motion_list, dim=0)
            y_all = torch.stack(label_list, dim=0)

            loss, vp, tp, mp = compute_loss(v_all, t_all, m_all, recon_m_all, y_all)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            video_precision += vp
            text_precision += tp
            motion_precision += mp
            loss_count += 1

            video_list.clear()
            text_list.clear()
            motion_list.clear()
            recon_motion_list.clear()
            label_list.clear()

            if step % 40 == 0:
                logger.info(f"[Step {step}] Loss: {loss.item():.4f}, "
                            f"Video: {vp:.4f}, Text: {tp:.4f}, Motion: {mp:.4f}")

    avg_loss = total_loss / max(1, loss_count)
    vp = video_precision / max(1, loss_count)
    tp = text_precision / max(1, loss_count)
    mp = motion_precision / max(1, loss_count)
    return avg_loss, vp, tp, mp


def train(model, compute_loss, train_loader, val_loader, config, logger, epochs=50):
    model = model.to(config.device)
    compute_loss = compute_loss.to(config.device)

    optimizer = Adam(itertools.chain(model.parameters(), compute_loss.parameters()),
                     lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs
    )

    best_loss = float('inf')
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(epochs):
        start_time = time.time()

        val_loss, val_vp, val_tp, val_mp = validate_one_epoch(
            model, compute_loss, val_loader, config, logger, epoch
        )

        avg_loss, vp, tp, mp = train_one_epoch(
            model, compute_loss, train_loader, optimizer, scheduler, config, logger, epoch
        )

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("ckpts", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/best_model_epoch{epoch}.pth")

        elapsed = time.time() - start_time
        logger.info(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | "
                    f"Video: {vp:.4f}, Text: {tp:.4f}, Motion: {mp:.4f} || "
                    f"Val Loss: {val_loss:.4f} | Video: {val_vp:.4f}, Text: {val_tp:.4f}, Motion: {val_mp:.4f} "
                    f"| Time: {elapsed:.1f}s")


@torch.no_grad()
def validate_one_epoch(model, compute_loss, val_loader, config, logger, epoch):
    model.eval()
    compute_loss.eval()
    device = config.device

    total_loss, video_precision, text_precision, motion_precision = 0, 0, 0, 0
    loss_count = 0
    max_size = len(val_loader)
    feature_size = 512
    v_embeds = torch.zeros((max_size, feature_size)).cuda()
    t_embeds = torch.zeros((max_size, feature_size)).cuda()
    m_embeds = torch.zeros((max_size, feature_size)).cuda()
    rm_embeds = torch.zeros((max_size, feature_size)).cuda()
    index = 0
    for step, (motions, videos, texts, labels) in enumerate(tqdm(val_loader, desc=f"[Val Epoch {epoch}]")):
        motions, videos, labels = motions.to(device), videos.to(device), labels.to(device)
        bs, _, _, nframes = motions.shape
        lengths = torch.full((bs,), nframes, device=device)
        mask = torch.ones((bs, nframes), dtype=torch.bool, device=device)
        batch = {'x': motions, 'mask': mask, 'lengths': lengths, 'y': torch.zeros(bs, dtype=torch.long).to(device)}

        v_embed, t_embed, m_embed, recon_m_embed = model(batch, videos, texts)
        v_embeds[index:index+bs] = v_embed
        t_embeds[index:index+bs] = t_embed
        m_embeds[index:index+bs] = m_embed
        rm_embeds[index:index+bs] = recon_m_embed

        index += bs
        loss, vp, tp, mp = compute_loss(v_embed, t_embed, m_embed, recon_m_embed, labels)

        total_loss += loss.item()
        video_precision += vp
        text_precision += tp
        motion_precision += mp
        loss_count += 1
    v_embeds = v_embeds[:index]
    t_embeds = t_embeds[:index]
    m_embeds = m_embeds[:index]
    rm_embeds = rm_embeds[:index]

    # tsne_(v_embeds.cpu().numpy(), t_embeds.cpu().numpy(), m_embeds.cpu().numpy(), rm_embeds.cpu().numpy(), name=f'TSET-{epoch}')
    avg_loss = total_loss / max(1, loss_count)
    vp = video_precision / max(1, loss_count)
    tp = text_precision / max(1, loss_count)
    mp = motion_precision / max(1, loss_count)

    logger.info(f"[Val Epoch {epoch}] Loss: {avg_loss:.4f} | "
                f"Video: {vp:.4f}, Text: {tp:.4f}, Motion: {mp:.4f}")
    return avg_loss, vp, tp, mp


if __name__ == "__main__":
    logger = setup_logger()
    config = Config()
    train_loader, val_loader = get_dataloaders()

    model = Model()
    compute_loss = Loss(config)

    train(model, compute_loss, train_loader, val_loader, config, logger, epochs=50)
