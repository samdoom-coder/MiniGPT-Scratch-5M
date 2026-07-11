%%writefile MiniGPT/src/mlp.py

import torch.nn as nn

from src.config import MiniGPTConfig


class FeedForward(nn.Module):
    """
    Feed Forward Network (MLP)
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                config.embedding_dim,
                config.ffn_dim
            ),

            nn.GELU(),

            nn.Dropout(config.dropout),

            nn.Linear(
                config.ffn_dim,
                config.embedding_dim
            ),

            nn.Dropout(config.dropout)

        )

    def forward(self, x):
        return self.net(x)