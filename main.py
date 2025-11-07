import socket
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model_path = "./gpt_dialog_model"

if os.path.exists(model_path):
    print("Loading saved model...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
else:
    print("Model not found! Please train the model first.")
    exit()

model.to(device)
model.eval()


def generate_human_like_response(prompt, max_new_tokens=30, temperature=0.9, top_p=0.95, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Only generate 30 NEW tokens max
            min_new_tokens=5,               # At least 5 tokens
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    response = post_process_response(generated_text, prompt)
    return response

def post_process_response(text, original_prompt):
    """
    Clean up the response to make it more human-like:
    1. Remove the prompt if it's repeated
    2. Take only the first sentence or short answer
    3. Remove any follow-up questions that seem scripted
    """
    # Remove the original prompt if it appears in the response
    text = text.strip()
    if text.lower().startswith(original_prompt.lower()):
        text = text[len(original_prompt):].strip()
    
    # Split by common sentence endings
    sentences = []
    for delimiter in ['. ', '! ', '? ', '\n']:
        if delimiter in text:
            text = text.split(delimiter)[0] + delimiter.rstrip()
            break
    
    # Remove common scripted patterns
    scripted_phrases = [
        "Can I help you",
        "How can I assist",
        "What can I do for you",
        "Is there anything else",
        "Would you like to",
        "Do you want to"
    ]
    
    for phrase in scripted_phrases:
        if phrase in text:
            text = text.split(phrase)[0].strip()
            # Add proper ending punctuation if missing
            if text and text[-1] not in '.!?':
                text += '.'
            break
    
    # Limit to reasonable conversational length (max 15 words for casual responses)
    words = text.split()
    if len(words) > 15:
        text = ' '.join(words[:15])
        if text[-1] not in '.!?':
            text += '...'
    
    return text.strip()


def format_prompt_for_casual_response(question):
    """
    Format the prompt to encourage shorter, casual responses.
    Add context that suggests brief, human-like answers.
    """
    # For very short questions, keep them as is
    if len(question.split()) <= 5:
        return question
    
    return question

HOST = "127.0.0.1"
PORT = 65432

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

print(f"\n[SERVER] Listening on {HOST}:{PORT}")

while True:
    conn, addr = server_socket.accept()
    print(f"[SERVER] Connected by {addr}")
    
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                print(f"[SERVER] Connection closed")
                break
            
            message = data.decode().strip()
            print(f"[CLIENT] {message}")
            
            if message.lower() in ["quit", "exit"]:
                conn.sendall("Goodbye!".encode())
                break
            
            response = generate_human_like_response(message, max_new_tokens=20, temperature=0.9)
            print(f"[AI] {response}")
            conn.sendall(response.encode())