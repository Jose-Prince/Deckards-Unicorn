from gpt_model import GPT
from load_data import load_dataset
from train_model import train

import os
import torch
from transformers import GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT(vocab_size=len(tokenizer), n_embd=256, n_heads=8, n_layers=4, block_size=128).to(device)

if not os.path.exists("./gpt_dialog_model.pt"):
    train_loader, val_loader = load_dataset()
    train(tokenizer, model, train_loader, val_loader)

model.load_state_dict(torch.load("gpt_dialog_model.pt"))

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens)
            logits = logits[:, -1, :] / temperature

            top_k_values, top_k_indices = torch.topk(logits, k=top_k)
            probs = torch.softmax(top_k_values, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs, num_samples=1)]
            next_token = next_token.unsqueeze(0)  # ← mantiene batch dimension correcta

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


prompt = "Hello"
response = generate_text(model, tokenizer, prompt, max_length=60)
print(response)
