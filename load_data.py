import pandas as pd
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def clean_text(text):
    """Clean and normalize dialog text."""
    text = str(text)  # Ensure it's a string
    text = text.strip("[]")
    text = text.replace("\\n", " ").replace("\n", " ")
    text = text.replace("'", "").replace('"', "")
    text = text.replace(" ,", ",").replace(" ?", "?")
    text = text.replace(" .", ".").replace(" !", "!")
    text = " ".join(text.split())  # Remove extra whitespace
    return text

def tokenize_function(examples):
    """Tokenize dialog text with proper padding."""
    return tokenizer(
        examples["dialog"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True
    )

def load_dataset():
    train_df = pd.read_csv("data/UnifiedDataset/train.csv")
    valid_df = pd.read_csv("data/UnifiedDataset/validation.csv")
    test_df = pd.read_csv("data/UnifiedDataset/test.csv")

    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    # Tokenize datasets
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"labels": x["input_ids"]},
        batched=True
    )

    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8)
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=8)

    return train_loader, val_loader, test_loader

