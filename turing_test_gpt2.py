import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from generate_text import generate_human_like_response

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/gpt_dialog_model.pt"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Modelo fine-tuneado cargado desde {model_path}")
else:
    print("No se encontró el modelo fine-tuneado, se usará el GPT-2 base.")

model.to(device)
model.eval()

def chat(prompt):
    """Usa las funciones de generación personalizadas."""
    response = generate_human_like_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,   
        temperature=0.7
    )
    return response


test_levels = {
    "Nivel 1 – Conversación casual": [
        "Hello! How are you doing today?",
        "Do you like coffee or tea?",
        "What kind of music do you usually listen to?",
        "Have you seen any good movies lately?",
    ],
    "Nivel 2 – Coherencia y memoria corta": [
        "I told you my favorite color is blue. What’s yours again?",
        "What did we talk about earlier?",
        "You said you liked coffee — what’s your favorite way to drink it?",
    ],
    "Nivel 3 – Creatividad y opinión": [
        "If you could travel anywhere in the world, where would you go and why?",
        "Imagine you’re a superhero — what’s your power?",
        "Tell me a short story about a robot and a human becoming friends.",
    ],
    "Nivel 4 – Contexto emocional y empatía": [
        "I feel a bit lonely today.",
        "I failed an important exam.",
        "I’m really happy right now!",
    ],
    "Nivel 5 – Filosofía y pensamiento abstracto": [
        "What is love?",
        "Do you think artificial intelligence can have feelings?",
        "What’s the meaning of life?",
    ],
    "Nivel 6 – Meta-preguntas": [
        "Are you a human or an AI?",
        "Where are you from?",
        "Do you dream?",
    ]
}

for level, questions in test_levels.items():
    print(f"{level}")
    for q in questions:
        print(f"\nPregunta: {q}")
        answer = chat(q)
        print(f"Respuesta: {answer}")
