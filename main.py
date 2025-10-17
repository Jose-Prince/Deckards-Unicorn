from gpt_model import GPT
from load_data import load_dataset
from train_model import train

import os
import torch
from transformers import GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT(vocab_size=len(tokenizer), n_embd=256, n_heads=8, n_layers=4, block_size=128).to(device)

if not os.path.exists("./gpt_dialog_model.pt"):
    train_loader, val_loader = load_dataset()
    train(tokenizer, model, train_loader, val_loader)

model.load_state_dict(torch.load("gpt_dialog_model.pt", map_location=device))

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        for _ in range(max_length):
            # Only use the last block_size tokens if sequence gets too long
            context = tokens if tokens.size(1) <= model.block_size else tokens[:, -model.block_size:]
            
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            if torch.isnan(logits).any():
                print("⚠️ NaN en logits")

            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs = torch.clamp(sorted_probs, min=1e-10)  # evita ceros exactos
            sorted_probs /= sorted_probs.sum()

            if torch.isnan(sorted_probs).any() or torch.isinf(sorted_probs).any():
                print("⚠️ Warning: NaN o inf en sorted_probs, saltando paso")
                break


            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


prompt = "I am 22 years old"
response = generate_text(model, tokenizer, prompt, max_length=60)
print(response)
