import socket
import torch
from generate_text import generate_text
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def start_server(host="127.0.0.1", port=5050):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(torch.load("models/gpt2_dialog_model.pt", map_location=device))
    model.eval()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"TCP Server listening in {host}:{port}")

    while True:
        conn, addr = server.accept()
        print(f"Connection stablished with {addr}")
        with conn:
            while True:
                data = conn.recv(1024).decode("utf-8")
                if not data:
                    break
                print(f"Client: {data}")

                response = generate_text(model, tokenizer, data, max_length=80)
                conn.sendall(response.encode("utf-8"))

        print(f"Connection closed with {addr}")

