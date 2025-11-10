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


def train(tokenizer, model, train_loader, val_loader, epochs=5, lr=3e-4):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)
    print(f"\nTraining on: {device.upper()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    early_stopping = EarlyStopping(patience=3, save_path="gpt_dialog_model.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in train_progress:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    print("Training complete.")

