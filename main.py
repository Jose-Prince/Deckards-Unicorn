from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
    """
    Generate SHORT, human-like conversational responses.
    
    Key changes:
    - max_new_tokens instead of max_length (generates only NEW tokens, not counting prompt)
    - Higher temperature (0.9) for more natural variation
    - Stops at first sentence ending or newline
    - Limits response length aggressively
    """
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
            # Stop at sentence endings
            early_stopping=True
        )
    
    # Decode only the generated part (excluding the prompt)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Post-processing: Extract only the first response
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


# Test with your examples
print("\n" + "="*60)
print("HUMAN-LIKE CONVERSATIONAL AI (Short Responses)")
print("="*60)

test_prompts = [
    "Who are you?",
    "Hello, how are you?",
    "What's your name?",
    "Tell me about yourself",
    "What do you like to do?",
    "How old are you?",
    "Where are you from?",
    "What's up?",
]

print("\n--- TESTING DIFFERENT TEMPERATURE SETTINGS ---\n")

# Test with different settings
settings = [
    {"temp": 0.9, "max_tokens": 20, "desc": "Natural & Brief"},
    {"temp": 0.8, "max_tokens": 30, "desc": "Balanced"},
    {"temp": 1.0, "max_tokens": 15, "desc": "Very Casual"},
]

for setting in settings:
    print(f"\n*** {setting['desc']} (temp={setting['temp']}, max_tokens={setting['max_tokens']}) ***\n")
    
    for prompt in test_prompts[:4]:  # Test first 4 prompts
        response = generate_human_like_response(
            prompt, 
            max_new_tokens=setting['max_tokens'],
            temperature=setting['temp']
        )
        print(f"Q: {prompt}")
        print(f"A: {response}")
        print("-" * 60)


# Interactive mode with human-like responses
print("\n" + "="*60)
print("INTERACTIVE MODE - Try to distinguish from human!")
print("Type 'quit' to exit")
print("="*60)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if user_input:
        response = generate_human_like_response(
            user_input, 
            max_new_tokens=25,  # Very short responses
            temperature=0.9
        )
        print(f"AI: {response}")


# Tips for improvement
print("\n" + "="*60)
print("TIPS TO IMPROVE HUMAN-LIKENESS:")
print("="*60)
print("""
1. RETRAIN with better data:
   - Use Reddit conversations, Twitter threads, or chat logs
   - Avoid formal customer service dialogs
   - Include casual language, slang, and natural speech patterns

2. DATA PREPROCESSING:
   - Format as: "Q: question A: short_answer"
   - Keep answers under 20 words
   - Include conversational markers (um, well, actually, etc.)

3. FINE-TUNING APPROACH:
   - Start from a conversational model like DialoGPT or Blenderbot
   - Use smaller max_length during training (64 tokens instead of 128)
   - Add special tokens for turn-taking

4. GENERATION PARAMETERS:
   - max_new_tokens: 15-30 (forces brevity)
   - temperature: 0.85-0.95 (more natural variation)
   - Add stopping criteria for sentence endings

5. POST-PROCESSING:
   - Cut at first sentence
   - Remove repeated questions
   - Filter out overly formal language
""")