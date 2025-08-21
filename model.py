import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoProcessor, AutoModelForZeroShotImageClassification
from src.models.get_model import get_model
import clip

def get_clip_model():
    parameters = json.load(open('parameters.json', 'r'))
    parameters['device'] = 'cpu'
    clip_model, _ = clip.load("ViT-B/32", device=parameters['device'], jit=False)

    for domain in parameters.get('clip_training', '').split('_'):
        clip_num_layers = parameters.get('clip_layers', 12)
        if domain == 'text':
            clip_model.initialize_parameters()
            clip_model.transformer.resblocks = clip_model.transformer.resblocks[:clip_num_layers]
        if domain == 'image':
            clip_model.initialize_parameters()
            clip_model.visual.transformer = clip_model.transformer.resblocks[:clip_num_layers]

    if parameters.get('clip_training', '') == '':
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

    return get_model(parameters, clip_model)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
        self.temporal_embedding = nn.Linear(768, 768)
        self.temporal_transformer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=256, dropout=0.1)
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )

    def forward(self, videos, texts):
        device = videos.device
        B, T, _, _, _ = videos.shape
        embeddings = []
        for b in range(B):
            inputs = self.processor(text=texts[b], images=videos[b], return_tensors="pt", padding=True).to(device)
            vision_out = self.model.vision_model(inputs['pixel_values'])
            embed = vision_out['pooler_output']
            embed = self.temporal_embedding(embed)
            embed = self.temporal_transformer(embed)
            embed = self.projector(embed.mean(dim=0).unsqueeze(0))
            embeddings.append(embed)
        return torch.cat(embeddings)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )

    def forward(self, texts, device):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.model.distilbert(**tokens)
        cls_embed = outputs['last_hidden_state'][:, 0, :]
        return self.projector(cls_embed)


class CrossModalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_attn = nn.MultiheadAttention(512, 8)
        self.image_attn = nn.MultiheadAttention(512, 8)

    def forward(self, query, key1, key2):
        attn1, _ = self.text_attn(query, key1, key1)
        attn2, _ = self.image_attn(query, key2, key2)
        return (query + 0.5 * (attn1 + attn2)).squeeze(1)


class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(512, 8)
        self.norm = nn.LayerNorm(512)

    def forward(self, text, image, motion):
        x = torch.stack([text, image, motion], dim=1).permute(1, 0, 2)
        fused, _ = self.attn(x, x, x)
        return self.norm(fused.permute(1, 0, 2).mean(dim=1))


class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

    def forward(self, fused):
        return self.decoder(fused)


class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = VideoEncoder()
        self.text_encoder = TextEncoder()
        self.clip_model = get_clip_model()
        self.attn = CrossModalAttention()
        self.fusion = FusionModule()
        self.reconstructor = Reconstructor()

    def forward(self, motion_inputs, video_inputs, text_inputs):
        device = video_inputs.device
        video_feat = self.video_encoder(video_inputs, text_inputs)
        text_feat = self.text_encoder(text_inputs, device)
        motion_feat = self.clip_model.encoder(motion_inputs)["mu"]

        img_att = self.attn(video_feat, text_feat, motion_feat)
        txt_att = self.attn(text_feat, video_feat, motion_feat)
        mot_att = self.attn(motion_feat, text_feat, video_feat)

        fused = self.fusion(txt_att, img_att, mot_att)
        recon = self.reconstructor(fused)
        return img_att, txt_att, mot_att, recon