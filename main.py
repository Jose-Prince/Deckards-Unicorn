import os
import torch
from transformers import GPT2Tokenizer
from gpt_model import GPT
from load_data import load_dataset
from train_model import train
from tcp_server import start_server

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPT(vocab_size=len(tokenizer), n_embd=256, n_heads=8, n_layers=4, block_size=128).to(device)

    model_path = "models/gpt_dialog_model.pt"

    if not os.path.exists(model_path):
        print("Training model...")
        train_loader, val_loader, _ = load_dataset()
        train(tokenizer, model, train_loader, val_loader)
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))

    start_server()

if __name__ == "__main__":
    main()

