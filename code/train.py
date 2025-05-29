import itertools
import time
import torch
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer
from dataloader import MultiFeatureDataset_train, MultiFeatureDataset_test
from model import Model
from utils.metric import Loss, compute_topk, AverageMeter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

save_dir = Path('./TSNE-visualize-with-multi-color-on-testset-coral')
save_dir.mkdir(parents=True, exist_ok=True)

def tsne_(img_feats, txt_feats, aud_feats, fus_feats, labels, name):
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

class Config:
    def __init__(self):
        self.image_dim = 512
        self.text_dim = 768
        self.audio_dim = 512
        self.num_classes = 30546
        self.resume = False
        self.feature_size = 512
        self.embed_dim = 512
        self.num_heads = 8
        self.tau = 0.05

config = Config()
model = Model(
    image_dim=config.image_dim,
    text_dim=config.text_dim,
    audio_dim=config.audio_dim,
    embed_dim=config.embed_dim,
    num_heads=config.num_heads,
    tau=config.tau
)

def train(epochs, train_loader, model, compute_loss, test_loader, config):
    compute_loss.cuda()
    model = model.cuda()
    optimizer = Adam(itertools.chain(model.parameters(), compute_loss.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)
    print(f"可训练参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in compute_loss.parameters() if p.requires_grad)}")

    min_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        image_precision_epoch = 0
        text_precision_epoch = 0
        audio_precision_epoch = 0

        if (epoch + 1) % 1 == 0:
            evaluate(test_loader, model, config, epoch)

        model.train()
        for data in tqdm(train_loader):
            images = data["image"].cuda()
            text_input_ids = data['text']['input_ids'].cuda()
            text_attention_mask = data['text']['attention_mask'].cuda()
            audios = data["audio"].cuda()
            labels = data['labels'].cuda()

            image_embeddings, text_embeddings, audio_embeddings, reconstructed_audio_embeddings = model(images, text_input_ids, text_attention_mask, audios)
            loss, image_precision, text_precision, audio_precision = compute_loss(image_embeddings, text_embeddings, audio_embeddings, reconstructed_audio_embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            image_precision_epoch += image_precision
            text_precision_epoch += text_precision
            audio_precision_epoch += audio_precision

        epoch_loss /= len(train_loader)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), f'model_output_cs_L/model_epoch{epoch}.pth')

        end_time = time.time()
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s, Image Precision: {image_precision_epoch/len(train_loader):.4f}, Text Precision: {text_precision_epoch/len(train_loader):.4f}, Audio Precision: {audio_precision_epoch/len(train_loader):.4f}")

def evaluate(data_loader, network, args, epoch):
    batch_time = AverageMeter()
    network.eval()

    max_size = 64 * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    audio_bank = torch.zeros((max_size, args.feature_size)).cuda()
    fuse_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()

    index = 0
    with torch.no_grad():
        end = time.time()
        for data in tqdm(data_loader):
            images = data['image'].cuda()
            text_input_ids = data['text']['input_ids'].cuda()
            text_attention_mask = data['text']['attention_mask'].cuda()
            labels = data['labels'].cuda()
            audio_input = data['audio'].cuda()

            interval = images.shape[0]
            image_embeddings, text_embeddings, audio_embeddings, fuse_audio_embeddings = network(images, text_input_ids, text_attention_mask, audio_input)

            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            audio_bank[index: index + interval] = audio_embeddings
            fuse_bank[index: index + interval] = fuse_audio_embeddings
            labels_bank[index: index + interval] = labels

            batch_time.update(time.time() - end)
            end = time.time()
            index += interval

        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        audio_bank = audio_bank[:index]
        fuse_bank = fuse_bank[:index]
        labels_bank = labels_bank[:index]

        tsne_(images_bank.cpu().numpy(), text_bank.cpu().numpy(), audio_bank.cpu().numpy(), fuse_bank.cpu().numpy(), labels_bank.cpu().numpy(), f'TEST-{epoch}')

        ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1, 10], True)
        ac_top1_i2a, ac_top10_i2a, ac_top1_a2i, ac_top10_a2i = compute_topk(images_bank, audio_bank, labels_bank, labels_bank, [1, 10], True)
        ac_top1_a2t, ac_top10_a2t, ac_top1_t2a, ac_top10_t2a = compute_topk(audio_bank, text_bank, labels_bank, labels_bank, [1, 10], True)

        print(f'-------- Evaluation @ Epoch {epoch} --------')
        print("ac_top1_i2t:", ac_top1_i2t.item())
        print("ac_top10_i2t:", ac_top10_i2t.item())
        print("ac_top1_t2i:", ac_top1_t2i.item())
        print("ac_top10_t2i:", ac_top10_t2i.item())
        print("ac_top1_i2a:", ac_top1_i2a.item())
        print("ac_top10_i2a:", ac_top10_i2a.item())
        print("ac_top1_a2i:", ac_top1_a2i.item())
        print("ac_top10_a2i:", ac_top10_a2i.item())
        print("ac_top1_a2t:", ac_top1_a2t.item())
        print("ac_top10_a2t:", ac_top10_a2t.item())
        print("ac_top1_t2a:", ac_top1_t2a.item())
        print("ac_top10_t2a:", ac_top10_t2a.item())
        print('-----------------------------------------')

def main():
    train_dataset = MultiFeatureDataset_train(
        "data/flickr30k/flickr30k_train_captions.jsonl",
        "data/flickr30k/flickr30k/images",
        "data/flickr30k/data2audio/train",
        DistilBertTokenizer.from_pretrained('models/distilbert-base-uncased'),
        512
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    test_dataset = MultiFeatureDataset_test(
        "data/flickr30k/flickr30k_test_captions.jsonl",
        "data/flickr30k/flickr30k/images",
        "data/flickr30k/data2audio/test",
        DistilBertTokenizer.from_pretrained('models/distilbert-base-uncased'),
        512
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    compute_loss = Loss(config)
    train(100, train_loader, model, compute_loss, test_loader, config)

if __name__ == '__main__':
    main()
