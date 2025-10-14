import pandas as pd
from datasets import Dataset, DatasetDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer

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

train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8)

print(next(iter(train_loader)))
