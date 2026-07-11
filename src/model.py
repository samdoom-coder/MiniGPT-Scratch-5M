import torch
import torch.nn as nn

from src.config import MiniGPTConfig
from src.embeddings import GPTEmbedding
from src.transformer import TransformerBlock


class MiniGPT(nn.Module):
    """
    MiniGPT Language Model
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        # Save config for generate()
        self.config = config

        self.embedding = GPTEmbedding(config)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

        # Weight Tying (GPT-2 Style)
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    ):
        """
        Generate text using Temperature + Top-k + Top-p (Nucleus) Sampling.
        """

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the last context window
            input_crop = input_ids[:, -self.config.context_length:]

            # Forward pass
            logits = self(input_crop)

            # Take logits for the last token only
            logits = logits[:, -1, :]

            # -------------------------
            # Temperature
            # -------------------------
            logits = logits / temperature

            # -------------------------
            # Top-k
            # -------------------------
            if top_k is not None:

                values, indices = torch.topk(
                    logits,
                    k=min(top_k, logits.size(-1))
                )

                filtered_logits = torch.full_like(
                    logits,
                    float("-inf")
                )

                filtered_logits.scatter_(
                    dim=-1,
                    index=indices,
                    src=values
                )

                logits = filtered_logits

            # -------------------------
            # Convert to probabilities
            # -------------------------
            probs = torch.softmax(
                logits,
                dim=-1
            )

            # -------------------------
            # Top-p (Nucleus Sampling)
            # -------------------------
            if top_p is not None:

                sorted_probs, sorted_indices = torch.sort(
                    probs,
                    descending=True
                )

                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p

                # Always keep at least one token
                sorted_indices_to_remove[..., 1:] = \
                    sorted_indices_to_remove[..., :-1].clone()

                sorted_indices_to_remove[..., 0] = False

                sorted_probs[sorted_indices_to_remove] = 0

                # Renormalize
                sorted_probs = sorted_probs / sorted_probs.sum(
                    dim=-1,
                    keepdim=True
                )

                sampled = torch.multinomial(
                    sorted_probs,
                    num_samples=1
                )

                next_token = sorted_indices.gather(
                    dim=-1,
                    index=sampled
                )
                
                # Stop if EOS token is generated
                if next_token.item() == self.config.eos_token_id:
                    break

            else:

                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

            # Append token
            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1
            )

        return input_ids