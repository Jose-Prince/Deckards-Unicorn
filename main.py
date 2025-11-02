from gpt_model import GPT
from load_data import load_dataset
from train_model import train

import os
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model_path = "./gpt_dialog_model"

if os.path.exists(model_path):
    print("Loading saved model...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
else:
    print("Loading base GPT-2 for fine-tuning...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    train_loader, val_loader = load_dataset()
    train(tokenizer, model, train_loader, val_loader, epochs=5)


model.to(device)
model.eval()

def generate_text(prompt, max_length=80, temperature=0.6, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "Who are you?"
response = generate_text(model, tokenizer, prompt, max_length=60)
print(response)
