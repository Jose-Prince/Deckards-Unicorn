import pandas as pd
from datasets import Dataset, DatasetDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam 
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer
from tqdm import tqdm

# Deep Learning model
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_heads=8, n_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_heads,
                dim_feedforward=4*n_embd,
                activation="gelu",
                batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, attention_mask=None):
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Create causal mask (prevents attending to future tokens)
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        
        # Create padding mask (True for padding tokens that should be ignored)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)  # True where we should ignore
        else:
            padding_mask = None
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(
                x, 
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask
            )
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
