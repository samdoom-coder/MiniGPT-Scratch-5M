from dataclasses import dataclass


@dataclass
class MiniGPTConfig:
    # Tokenizer
    vocab_size: int = 8000
    tokenizer_path: str = "/content/MiniGPT/tokenizer"

    # Model
    context_length: int = 128
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    # Feed Forward Network
    ffn_dim: int = 1024

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 20

    # Special Tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def __post_init__(self):
        assert self.embedding_dim % self.num_heads == 0, \
            "embedding_dim must be divisible by num_heads"

        self.head_dim = self.embedding_dim // self.num_heads