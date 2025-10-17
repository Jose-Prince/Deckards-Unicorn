import torch
import torch.nn as nn

from torch.optim import Adam 

from tqdm import tqdm

# Training
def train(tokenizer, model, train_loader, val_loader, epochs=3):
    device = "cuda" if torch.cuda.is_available() else "cpu" # use cuda when possible

    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Evaluate model
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f"Validating {epoch+1}/{epochs}", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(inputs[:, :-1], attention_mask=attention_mask[:, :-1])
                targets = inputs[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), "gpt_dialog_model.pt")
