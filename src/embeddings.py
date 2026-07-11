%%writefile MiniGPT/src/embeddings.py

import torch
import torch.nn as nn

from src.config import MiniGPTConfig


class PositionalEmbedding(nn.Module):

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        self.embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim
        )

    def forward(self, x):
        batch_size, seq_length = x.shape

        positions = torch.arange(
            seq_length,
            device=x.device
        ).unsqueeze(0)

        return self.embedding(positions)


class GPTEmbedding(nn.Module):

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )

        self.position_embedding = PositionalEmbedding(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(input_ids)

        embeddings = token_embeddings + position_embeddings

        return self.dropout(embeddings)