import pandas as pd
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # para evitar error de padding

def clean_text(text):
    text = text.strip("[]")
    text = text.replace("\\n", " ").replace("\n", " ")
    text = text.replace("'", "").replace('"', "")
    text = text.replace(" ,", ",").replace(" ?", "?").replace(" .", ".").replace(" !", "!")
    text = " ".join(text.split())
    return text

def tokenize_function(examples):
    return tokenizer(
        examples["dialog"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def load_dataset():
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/validation.csv")
    test_df = pd.read_csv("data/test.csv")

    for df in [train_df, valid_df, test_df]:
        df["dialog"] = df["dialog"].apply(clean_text)

    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(tokenized["train"], batch_size=2, shuffle=True)
    val_loader = DataLoader(tokenized["validation"], batch_size=2)

    return train_loader, val_loader
