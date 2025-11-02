import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam 

from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path="gpt_model.pt"):
        """
        Args:
            patience (int, optional): number of epochs without upgrade before stopping.
            min_delta (float, optional): min upgrade in val_loss for considering progress.
        """
        
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
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping activated: without improvement in {self.patience} epochs.")
                self.early_stop = True

# Training
def train(tokenizer, model, train_loader, val_loader, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEntrenando en: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")


    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Early stopping initialize
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4, save_path="gpt_dialog_model.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch in train_progress:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = model(inputs[:, :-1], attention_mask=attention_mask[:, :-1])
            targets = inputs[:, 1:].contiguous()

            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        # Validate model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(inputs[:, :-1], attention_mask=attention_mask[:, :-1])
                targets = inputs[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Stopping training.")
            break

    if not early_stopping.early_stop:
        torch.save(model.state_dict(), "gpt_dialog_model.pt")