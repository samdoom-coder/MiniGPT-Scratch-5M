%%writefile MiniGPT/src/attention.py

import math
import torch
import torch.nn as nn

from src.config import MiniGPTConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        assert self.embedding_dim % self.num_heads == 0

        # Query, Key, Value projections
        self.query = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False
        )

        # Output projection (Wo)
        self.out_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        batch_size, seq_length, _ = x.shape

        # -------------------------------------------------
        # Project input into Query, Key and Value
        # -------------------------------------------------
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # -------------------------------------------------
        # Split into multiple heads
        # (B, T, C) -> (B, T, H, D)
        # -------------------------------------------------
        q = q.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )

        v = v.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )

        # -------------------------------------------------
        # (B, T, H, D) -> (B, H, T, D)
        # -------------------------------------------------
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -------------------------------------------------
        # Scaled Dot Product Attention
        # -------------------------------------------------
        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # -------------------------------------------------
        # Causal Mask
        # -------------------------------------------------
        mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask == 1,
            float("-inf")
        )

        # -------------------------------------------------
        # Softmax
        # -------------------------------------------------
        attention = torch.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        # -------------------------------------------------
        # Weighted sum of Values
        # -------------------------------------------------
        output = attention @ v

        # -------------------------------------------------
        # Combine Heads
        # (B, H, T, D) -> (B, T, H, D)
        # -------------------------------------------------
        output = output.transpose(1, 2)

        # -------------------------------------------------
        # (B, T, H, D) -> (B, T, C)
        # -------------------------------------------------
        output = output.contiguous().view(
            batch_size,
            seq_length,
            self.embedding_dim
        )

        # -------------------------------------------------
        # Final Linear Projection
        # -------------------------------------------------
        output = self.out_proj(output)

        output = self.dropout(output)

        return output