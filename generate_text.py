import torch

def generate_text(model, tokenizer, prompt, max_length=60, temperature=0.8, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            context = tokens[:, -model.block_size:] if tokens.size(1) > model.block_size else tokens

            logits = model(context)
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()

            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)

            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(tokens[0], skip_special_tokens=True)

