import pandas as pd
from datasets import Dataset, DatasetDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam 
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer
from tqdm import tqdm

# Load dataset
train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Tokenize the dialogs from the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["dialog"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(
    lambda x: {"labels": x["input_ids"]}, batched=True
)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8)

# Deep Learning model
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_heads=8, n_layers=4, block_size=128):
        super().__init__()
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
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Training
device = "cuda" if torch.cuda.is_available() else "cpu" # use cuda when possible
model = GPT(vocab_size=len(tokenizer), n_embd=256, n_heads=8, n_layers=4, block_size=128).to(device)

optimizer = Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for batch in train_progress:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        logits = model(inputs, attention_mask=attention_mask)

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_progress.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

# Evaluate model
    model.eval()
    val_loss = 0
    val_progress = tqdm(val_loader, desc=f"Validating {epoch+1}/{epochs}", leave=False)
    with torch.no_grad():
        for batch in val_progress:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(inputs, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            val_loss += loss.item()
            val_progress.set_postfix({"val_loss": loss.item()})

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

torch.save(model.state_dict(), "gpt_dialog_model.pt")
