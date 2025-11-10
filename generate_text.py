import torch
import torch.nn.functional as F
import random
import re

def generate_text(model, tokenizer, prompt, min_length=20, max_length=80, temperature=0.85, top_p=0.9, top_k=50):
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Determine block size
    block_size = getattr(model, "block_size", getattr(model.config, "n_positions", 1024))

    with torch.no_grad():
        for _ in range(max_length):
            context = tokens[:, -block_size:] if tokens.size(1) > block_size else tokens
            outputs = model(context)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits_filtered = torch.full_like(logits, float("-inf"))
                logits_filtered.scatter_(1, top_k_indices, top_k_values)
                logits = logits_filtered

            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 0] = False
                sorted_probs[sorted_indices_to_remove] = 0.0
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                next_token_idx = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices.gather(-1, next_token_idx)
            else:
                next_token = torch.multinomial(probs, 1)

            tokens = torch.cat([tokens, next_token], dim=1)
            next_id = next_token.item()

            if next_id == tokenizer.eos_token_id:
                break

            text_so_far = tokenizer.decode(tokens[0], skip_special_tokens=True)
            if re.search(r"(\b\w+\b)(?:\s+\1){2,}", text_so_far.lower()):
                break

            if len(tokens[0]) > min_length and text_so_far.strip().endswith(('.', '!', '?')):
                break

    generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
    if generated_text.lower().startswith(prompt.lower()):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    return response or generated_text


def generate_human_like_response(model, tokenizer, prompt, temperature=0.85):
    target_length = random.randint(25, 55)

    conversational_prompt = (
        "This is a natural, friendly and emotional chat between a human and an AI named Alex. "
        "Alex speaks casually and empathetically.\n\n"
        f"Human: {prompt}\nAI:"
    )

    response = generate_text(
        model,
        tokenizer,
        prompt,
        min_length=12,
        max_length=target_length,
        temperature=temperature,
        top_p=0.9,
        top_k=50
    )

    response = post_process_response(response, prompt)
    return response


def post_process_response(text, original_prompt):
    text = text.strip()
    for delim in ['\n', 'Human:', 'AI:']:
        if delim in text:
            text = text.split(delim)[0].strip()

    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ’ ", "'")

    # Evita respuestas ultra cortas
    if len(text.split()) < 6:
        fallback_phrases = [
            "That’s interesting.",
            "I get how you feel.",
            "That’s nice to hear.",
            "That sounds exciting!",
            "Tell me more about it."
        ]
        text += " " + random.choice(fallback_phrases)

    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text.strip()
