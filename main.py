from gpt_model import GPT
from load_data import load_dataset
from train_model import train

import os
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT(vocab_size=len(tokenizer), 
            n_embd=256, 
            n_heads=8, 
            n_layers=4, 
            block_size=128).to(device)

if not os.path.exists("./gpt_dialog_model.pt"):
    train_loader, val_loader = load_dataset()
    train(tokenizer, model, train_loader, val_loader)

model.load_state_dict(torch.load("gpt_dialog_model.pt", map_location=device))

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.6, top_k=50, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Only use the last block_size tokens if sequence gets too long
            context = tokens if tokens.size(1) <= model.block_size else tokens[:, -model.block_size:]
            
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("⚠️ NaN/Inf en logits, stopping generation")
                break

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_values)

            probs = F.softmax(logits, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                
                sorted_probs[sorted_indices_to_remove] = 0.0
                
                sorted_probs = sorted_probs / sorted_probs.sum(dim=1, keepdim=True)

                next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


prompt = "Who are you?"
response = generate_text(model, tokenizer, prompt, max_length=60)
print(response)
