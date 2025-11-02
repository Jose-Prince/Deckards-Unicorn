import torch
import numpy as np
from tqdm import tqdm

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
            model.save_pretrained(self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping: no improvement in {self.patience} epochs.")
                self.early_stop = True

def train(tokenizer, model, train_loader, val_loader, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    early_stopping = EarlyStopping(patience=3, save_path="gpt_dialog_model")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in train_progress:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Stopping training.")
            break