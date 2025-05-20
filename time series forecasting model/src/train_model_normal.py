import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset_builder import TempSequenceDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
VALIDATION_SPLIT = 0.2
WEIGHT_DECAY = 1e-4

dataset = TempSequenceDataset("data/processed", sequence_length=SEQUENCE_LENGTH)
input_size = dataset[0][0].shape[1]
output_size = dataset[0][1].shape[0]

val_size = int(len(dataset) * VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

print(f"Loaded dataset: {len(dataset)} samples")
print(f"Input shape: [batch, {SEQUENCE_LENGTH}, {input_size}], Target: [{output_size}]")

class TempLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc = nn.Linear(128, output_size)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.act(self.drop1(out))
        out, _ = self.lstm2(out)
        out = self.act(self.drop2(out))
        out = out[:, -1, :]
        return self.fc(out)

model = TempLSTM(input_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        preds = model(X)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            val_preds.append(pred)
            val_targets.append(y)
    val_preds = torch.cat(val_preds).numpy()
    val_targets = torch.cat(val_targets).numpy()

    val_mse = mean_squared_error(val_targets, val_preds)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    scheduler.step(val_mse)

    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.2f} | "
          f"Val MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")

    if val_mse < best_loss:
        best_loss = val_mse
        patience_counter = 0
        torch.save(model.state_dict(), "models/temp_lstm_final.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

print("Training complete. Best model saved to models/temp_lstm_final.pt")
