%%writefile MiniGPT/src/transformer.py

import torch.nn as nn

from src.attention import MultiHeadAttention
from src.mlp import FeedForward
from src.config import MiniGPTConfig


class TransformerBlock(nn.Module):

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadAttention(config)

        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = FeedForward(config)

    def forward(self, x):

        # Attention + Residual
        x = x + self.attention(self.ln1(x))

        # MLP + Residual
        x = x + self.mlp(self.ln2(x))

        return x