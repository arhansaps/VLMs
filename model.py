import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from encoder import VisionEncoder
from projection import ProjectionLayer


class VLM(nn.Module):
    def __init__(self, spatial_visual_tokens=False):
        super().__init__()

        #load the 3 components
        self.encoder = VisionEncoder(spatial=spatial_visual_tokens)
        self.projection = ProjectionLayer()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, images, input_ids, attention_mask):
        # images:         [B, 3, 224, 224]
        # input_ids:      [B, seq_len]
        # attention_mask: [B, seq_len]

        # 1. Extract visual features and project into GPT-2's embedding space
        visual_features = self.encoder(images)             # [B, 2048] or [B, 49, 2048]
        visual_tokens = self.projection(visual_features)   # [B, 768] or [B, 49, 768]
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(1)      # [B, 1, 768]

        #unsqueeze as gpt expects sequence_length in between with that, one image is represented as N tokens
        num_visual_tokens = visual_tokens.size(1)

        #Get GPT-2 token embeddings for the caption
        token_embeddings = self.gpt2.transformer.wte(input_ids)  # [B, seq_len, 768]

        #Prepend visual token(s) to caption embeddings, this is done so that the LLM has some info about the actual image as well and that info can be used in refining
        inputs_embeds = torch.cat([visual_tokens, token_embeddings], dim=1)  # [B, seq_len+N, 768]

        #Extend attention mask to cover the visual token(s) (always attend to them)
        visual_mask = torch.ones(images.size(0), num_visual_tokens, device=attention_mask.device)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)  # [B, seq_len+N]

        #Run GPT-2 — use inputs_embeds instead of input_ids since we already embedded
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        return outputs.logits  # [B, seq_len+N, vocab_size]
        #these logits are basically a vector(obv) which includes a score indicating how likely GPT thinks each vocab token is the most likely to come next at that pos.
