import pandas as pd
from datasets import Dataset, DatasetDict

from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def load_dataset():
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
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"labels": x["input_ids"]}, batched=True
    )
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True), DataLoader(tokenized_datasets["validation"], batch_size=8)

def tokenize_function(examples):
    return tokenizer(
        examples["dialog"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

