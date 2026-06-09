import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from encoder import VisionEncoder
from projection import ProjectionLayer


class VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionEncoder()
        self.projection = ProjectionLayer()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, images, input_ids, attention_mask):
        # images:         [B, 3, 224, 224]
        # input_ids:      [B, seq_len]
        # attention_mask: [B, seq_len]

        # 1. Extract visual features and project into GPT-2's embedding space
        visual_features = self.encoder(images)           # [B, 2048]
        visual_token = self.projection(visual_features)  # [B, 768]
        visual_token = visual_token.unsqueeze(1)         # [B, 1, 768]

        # 2. Get GPT-2 token embeddings for the caption
        token_embeddings = self.gpt2.transformer.wte(input_ids)  # [B, seq_len, 768]

        # 3. Prepend visual token to caption embeddings
        inputs_embeds = torch.cat([visual_token, token_embeddings], dim=1)  # [B, seq_len+1, 768]

        # 4. Extend attention mask to cover the visual token (always attend to it)
        visual_mask = torch.ones(images.size(0), 1, device=attention_mask.device)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)  # [B, seq_len+1]

        # 5. Run GPT-2 — use inputs_embeds instead of input_ids since we already embedded
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        return outputs.logits  # [B, seq_len+1, vocab_size]
