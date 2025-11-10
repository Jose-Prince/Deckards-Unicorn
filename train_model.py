import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path="gpt_dialog_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.save_path = save_path
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Model saved (val_loss: {val_loss:.4f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping: no improvement in {self.patience} epochs.")
                self.early_stop = True


def train(tokenizer, model, train_loader, val_loader, epochs=10, lr=3e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"\nTraining on: {device.upper()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    early_stopping = EarlyStopping(patience=5, save_path="models/gpt_dialog_model.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in train_progress:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            if attention_mask is not None:
                attn_mask = attention_mask[:, :-1]
            else:
                attn_mask = None

            # Forward pass
            outputs = model(inputs, attention_mask=attn_mask)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = loss_fn(
                logits.contiguous().view(-1, logits.size(-1)),
                targets.contiguous().view(-1)
            )

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf loss detected, skipping batch")
                continue

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            
            train_progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validating", leave=False)
            for batch in val_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None

                # Same splitting for validation
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                if attention_mask is not None:
                    attn_mask = attention_mask[:, :-1]
                else:
                    attn_mask = None

                outputs = model(inputs, attention_mask=attn_mask)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                loss = loss_fn(
                    logits.contiguous().view(-1, logits.size(-1)),
                    targets.contiguous().view(-1)
                )
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*60}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    print("\nTraining complete!")
    
    # Save final model
    if not early_stopping.early_stop:
        torch.save(model.state_dict(), "models/gpt_dialog_model.pt")
        print("Final model saved")