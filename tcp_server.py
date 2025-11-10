import socket
import torch
from generate_text import generate_human_like_response
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def start_server(host="127.0.0.1", port=65432):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load fine-tuned GPT-2 model
    model_path = "models/gpt_dialog_model"
    
    if os.path.exists(model_path):
        # Load with from_pretrained (matches save_pretrained in training)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        print(f"✓ Fine-tuned GPT-2 loaded from {model_path}")
    else:
        # Fallback to base GPT-2 if fine-tuned model doesn't exist
        print("⚠️ Fine-tuned model not found, using base GPT-2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    model.to(device)
    model.eval()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"TCP Server listening on {host}:{port}")
    print(f"Device: {device.upper()}\n")

    while True:
        conn, addr = server.accept()
        print(f"Connection established with {addr}")
        with conn:
            while True:
                data = conn.recv(1024).decode("utf-8").strip()
                if not data:
                    break
                print(f"Client: {data}")

                response = generate_human_like_response(
                    model, 
                    tokenizer, 
                    data, 
                    max_new_tokens=35,
                    temperature=0.85
                )
                print(f"AI: {response}\n")
                conn.sendall(response.encode("utf-8"))

        print(f"Connection closed with {addr}\n")