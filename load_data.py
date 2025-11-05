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
    """Load and prepare the conversational dataset."""
    print("Loading datasets...")
    
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/validation.csv")
    test_df = pd.read_csv("data/test.csv")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    # Clean text data
    for df in [train_df, valid_df, test_df]:
        df["dialog"] = df["dialog"].apply(clean_text)

    # Create HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["dialog"])
    
    # Add labels (same as input_ids for language modeling)
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    
    # Set format for PyTorch
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Create DataLoaders with better batch size
    # Increase batch size for better training stability
    batch_size = 8 if torch.cuda.is_available() else 4
    
    train_loader = DataLoader(
        tokenized["train"], 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        tokenized["validation"], 
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"Using batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader