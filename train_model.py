import torch
import numpy as np
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path="gpt_dialog_model"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.save_path = save_path
        self.early_stop = False

    def __call__(self, val_loss, model, tokenizer):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save both model and tokenizer
            model.save_pretrained(self.save_path)
            tokenizer.save_pretrained(self.save_path)
            print(f"✓ Model saved (val_loss: {val_loss:.4f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping: no improvement in {self.patience} epochs.")
                self.early_stop = True

def train(tokenizer, model, train_loader, val_loader, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on: {device.upper()}")
    
    model.to(device)
    
    # Use AdamW with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    early_stopping = EarlyStopping(patience=3, save_path="gpt_dialog_model")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in train_progress:
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass - GPT2LMHeadModel expects labels for loss calculation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validating", leave=False)
            for batch in val_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model, tokenizer)
        if early_stopping.early_stop:
            print("Stopping training early.")
            break
    
    # Save final model if training completed without early stopping
    if not early_stopping.early_stop:
        print("\nTraining completed! Saving final model...")
        model.save_pretrained("gpt_dialog_model")
        tokenizer.save_pretrained("gpt_dialog_model")
        print("✓ Final model saved")