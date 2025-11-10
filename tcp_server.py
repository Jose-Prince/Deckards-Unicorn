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

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    
    model_pt_path = "models/gpt_dialog_model.pt"
    
    if os.path.exists(model_pt_path):
        state_dict = torch.load(model_pt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Fine-tuned GPT-2 loaded from {model_pt_path}")
    else:
        print(f"Fine-tuned model not found at {model_pt_path}")
        print("Using base GPT-2 (not trained on your data)")
    
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
        
        conversation_history = []
        max_history_length = 5
        
        with conn:
            while True:
                data = conn.recv(1024).decode("utf-8").strip()
                if not data:
                    break
                
                print(f"Client: {data}")
                
                if conversation_history:
                    context = ""
                    for user_msg, ai_msg in conversation_history[-3:]:
                        context += f"{user_msg} {ai_msg} "
                    context += data
                else:
                    context = data

                response = generate_human_like_response(
                    model, 
                    tokenizer, 
                    context, 
                    max_new_tokens=20,
                    temperature=0.9      
                )
                
                if response.startswith("AI:"):
                    response = response[3:].strip()
                if response.startswith("User:"):
                    response = response.split("AI:")[-1].strip()
                
                print(f"AI: {response}\n")
                
                conversation_history.append((data, response))
                
                conn.sendall(response.encode("utf-8"))

        print(f"Connection closed with {addr}")
        print(f"Conversation had {len(conversation_history)} exchanges\n")