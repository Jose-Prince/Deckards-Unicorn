import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=60, temperature=0.85, top_p=0.92, top_k=50):
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    if hasattr(model, 'block_size'):
        block_size = model.block_size
    elif hasattr(model, 'config'):
        block_size = model.config.n_positions
    else:
        block_size = 1024

    with torch.no_grad():
        for _ in range(max_length):
            context = tokens[:, -block_size:] if tokens.size(1) > block_size else tokens

            outputs = model(context)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            logits = logits[:, -1, :] / temperature
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN/Inf detected in logits, stopping generation")
                break

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits_filtered = torch.full_like(logits, float('-inf'))
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
                
                next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    if generated_text.lower().startswith(prompt.lower()):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    return response if response else generated_text


def generate_human_like_response(model, tokenizer, prompt, max_new_tokens=35, temperature=0.85):
    response = generate_text(
        model, 
        tokenizer, 
        prompt, 
        max_length=max_new_tokens + len(tokenizer.encode(prompt)),
        temperature=temperature,
        top_p=0.92,
        top_k=50
    )
    
    response = post_process_response(response, prompt)
    
    return response


def post_process_response(text, original_prompt):
    text = text.strip()
    
    if text.lower().startswith(original_prompt.lower()):
        text = text[len(original_prompt):].strip()
    
    for delimiter in ['. ', '! ', '? ']:
        if delimiter in text:
            text = text.split(delimiter)[0] + delimiter.rstrip()
            break
    
    if '\n' in text:
        text = text.split('\n')[0].strip()
    
    incomplete_patterns = ['and your', 'and I', 'and the', 'and my']
    for pattern in incomplete_patterns:
        if text.strip().lower().startswith(pattern):
            text = text.replace(pattern, '').strip()
    
    words = text.split()
    if len(words) > 20:
        text = ' '.join(words[:20])
        if text[-1] not in '.!?':
            text += '...'
    
    if not text or len(text.strip()) < 2:
        return "Hey there!"
    
    return text.strip()