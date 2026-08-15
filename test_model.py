import torch

from model import VLM


def test_vlm_forward_single_token_shape():
    model = VLM(spatial_visual_tokens=False)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (2, 10))
    attention_mask = torch.ones(2, 10)
    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (2, 11, 50257)


def test_vlm_forward_spatial_tokens_shape():
    model = VLM(spatial_visual_tokens=True)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (2, 10))
    attention_mask = torch.ones(2, 10)
    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (2, 59, 50257)  # 49 visual tokens + 10 caption tokens
