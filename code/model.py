import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import CLIPProcessor, CLIPModel

class Model(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, audio_dim=512, embed_dim=512, num_heads=8, tau=0.05):
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained("models/clip-vit-base-patch32", torch_dtype=torch.float32)
        self.clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")

        self.bert_model = DistilBertModel.from_pretrained('models/distilbert-base-uncased', torch_dtype=torch.float32)
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert-base-uncased')

        self.audio_encoder = nn.Linear(audio_dim, embed_dim)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.text_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.image_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.audio_attn = nn.MultiheadAttention(embed_dim, num_heads)

        self.fusion_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.recon_decoder = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, audio_dim)
        )

    def forward(self, images, text_input_ids, text_attention_mask, audios):
        image_embeddings = self.clip_model.get_image_features(pixel_values=images).float()
        text_embeddings = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0, :]
        audio_embeddings = self.audio_encoder(audios)

        text_emb = self.text_proj(text_embeddings).unsqueeze(1)
        image_emb = self.image_proj(image_embeddings).unsqueeze(1)
        audio_emb = self.audio_proj(audio_embeddings)

        text_emb_fea = self._cross_attn(text_emb, image_emb, audio_emb)
        image_emb_fea = self._cross_attn(image_emb, text_emb, audio_emb)
        audio_emb_fea = self._cross_attn(audio_emb, text_emb, image_emb)

        fused_emb = self.fuse_features(text_emb_fea, image_emb_fea, audio_emb_fea)
        reconstructed_audio = self.reconstruct(fused_emb)

        return text_emb_fea, image_emb_fea, audio_emb_fea, reconstructed_audio

    def _cross_attn(self, query, key1, key2):
        attn1, _ = self.text_attn(query, key1, key1)
        attn2, _ = self.image_attn(query, key2, key2)
        emb_final = query + 0.5 * (attn1 + attn2)
        return emb_final.squeeze(1)

    def fuse_features(self, text, image, audio):
        features = torch.stack([text, image, audio], dim=1)
        features = features.permute(1, 0, 2)
        fused, _ = self.fusion_attn(features, features, features)
        fused = fused.permute(1, 0, 2)
        return self.fusion_norm(fused.mean(dim=1))

    def reconstruct(self, fused_emb):
        return self.recon_decoder(fused_emb)
