import torch

from encoder import VisionEncoder


def test_vision_encoder_default_output_shape():
    encoder = VisionEncoder()
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 2048)


def test_vision_encoder_spatial_output_shape():
    encoder = VisionEncoder(spatial=True)
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 49, 2048)
