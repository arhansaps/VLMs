import torch.nn as nn

"""Projection Layer : take the 2048 dim vector as input and convert it into a 768 dim vector
                        for the llm to work with of course."
"""

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=2048, output_dim=768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # x: [B, 2048]  →  [B, 768]
        return self.projection(x)
