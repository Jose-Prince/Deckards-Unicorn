import os
import shap
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt

print("🔍 Initializing SHAP Analysis for Conversational AI...")

# --- Model Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

model_pt_path = "models/gpt_dialog_model.pt"

if os.path.exists(model_pt_path):
    state_dict = torch.load(model_pt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ Fine-tuned GPT-2 loaded from {model_pt_path}")
else:
    print(f"⚠️ Fine-tuned model not found, using base GPT-2")

model.to(device)
model.eval()

print("\n" + "="*60)
print("SHAP EXPLAINABLE AI ANALYSIS")
print("="*60)


# --- Prediction Function for SHAP ---
def predict_probabilities(texts):
    """
    Predict probability distribution over vocabulary for next token.
    SHAP needs this to understand model behavior.
    """
    # Ensure input is list of strings
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        texts = [str(t) for t in texts]
    
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get logits for last token (next token prediction)
        last_logits = logits[:, -1, :]
        
        # Convert to probabilities
        probs = torch.softmax(last_logits, dim=-1)
        
        # Return top-k probabilities for efficiency
        top_k = 100  # Analyze top 100 most likely tokens
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        
        # Create sparse probability matrix
        batch_size = probs.shape[0]
        vocab_size = probs.shape[1]
        sparse_probs = torch.zeros((batch_size, vocab_size), device=probs.device)
        
        for i in range(batch_size):
            sparse_probs[i, top_indices[i]] = top_probs[i]
        
        return sparse_probs.cpu().numpy()


def predict_next_token(texts):
    """
    Simpler prediction function that returns probability of most likely token.
    Faster for SHAP analysis.
    """
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        texts = [str(t) for t in texts]
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Return max probability for each input
        max_probs, predicted_tokens = probs.max(dim=-1)
        
        return max_probs.cpu().numpy()


# --- Analyze Multiple Conversational Examples ---
prompts = [
    "Hello, how are you",
    "What is your name",
    "Do you like movies",
    "I am feeling happy today",
    "Tell me about yourself"
]

print("\n📊 Analyzing conversational prompts with SHAP...\n")

# Create SHAP explainer
print("Creating SHAP explainer (this may take a moment)...")
explainer = shap.Explainer(
    predict_next_token, 
    masker=shap.maskers.Text(tokenizer=tokenizer)
)

for i, prompt in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"Prompt {i+1}: '{prompt}'")
    print(f"{'='*60}")
    
    try:
        # Get SHAP values
        print("Computing SHAP values...")
        shap_values = explainer([prompt])
        
        # Display text plot
        print("\n📈 Token Importance Visualization:")
        shap.plots.text(shap_values[0], display=True)
        
        # Get predicted next token
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_token_id = outputs.logits[0, -1, :].argmax().item()
            predicted_token = tokenizer.decode([predicted_token_id])
        
        print(f"\n🎯 Predicted next token: '{predicted_token}'")
        
        # Show token contributions
        print("\n📊 Token Contribution Scores:")
        tokens = shap_values[0].data
        values = shap_values[0].values
        
        for token, value in zip(tokens, values):
            impact = "↑ Positive" if value > 0 else "↓ Negative" if value < 0 else "○ Neutral"
            print(f"  '{token}': {value:.4f} {impact}")
        
    except Exception as e:
        print(f"⚠️ Error analyzing prompt: {e}")
        continue

print("\n" + "="*60)


# --- Detailed Analysis: Compare Similar Prompts ---
print("\n🔬 DETAILED ANALYSIS: Impact of Specific Words")
print("="*60)

comparison_pairs = [
    ("I am happy", "I am sad"),
    ("Hello friend", "Hello stranger"),
    ("You are nice", "You are mean"),
]

for prompt1, prompt2 in comparison_pairs:
    print(f"\n📍 Comparing: '{prompt1}' vs '{prompt2}'")
    
    try:
        # Analyze both
        shap_values_1 = explainer([prompt1])
        shap_values_2 = explainer([prompt2])
        
        # Get predictions
        pred1 = predict_next_token([prompt1])[0]
        pred2 = predict_next_token([prompt2])[0]
        
        print(f"  '{prompt1}' → confidence: {pred1:.4f}")
        print(f"  '{prompt2}' → confidence: {pred2:.4f}")
        print(f"  Difference: {abs(pred1 - pred2):.4f}")
        
        # Show key word impact
        tokens1 = shap_values_1[0].data
        values1 = shap_values_1[0].values
        
        key_token = tokens1[-1]  # Last token is usually most different
        key_value = values1[-1]
        
        print(f"  Key token '{key_token}' impact: {key_value:.4f}")
        
    except Exception as e:
        print(f"  ⚠️ Error: {e}")


# --- Summary Statistics ---
print("\n" + "="*60)
print("📈 SUMMARY STATISTICS")
print("="*60)

print("\n✅ Analysis Complete!")
print("\nKey Insights:")
print("• Positive SHAP values → Token increases model confidence")
print("• Negative SHAP values → Token decreases model confidence")
print("• Larger absolute values → Greater impact on prediction")
print("\nUse this to understand:")
print("  - Which words make responses more/less confident")
print("  - How context affects next token prediction")
print("  - Model's sensitivity to specific phrases")