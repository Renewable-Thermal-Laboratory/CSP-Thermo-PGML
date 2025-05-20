import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset_builder_normal import TempSequenceDataset
from train_model_normal import TempLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# Reproducibility
torch.manual_seed(42)

# Load dataset
dataset = TempSequenceDataset("data/processed", sequence_length=10)
print(f"Dataset ready: {len(dataset)} total samples")

# Split into train/val/test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
print(f"Dataset: {len(dataset)} samples (Train: {train_size}, Val: {val_size}, Test: {test_size})")

# Data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

# Model setup
input_dim = len(dataset[0][0][0])
output_dim = len(dataset[0][1])
model = TempLSTM(input_size=input_dim, output_size=output_dim)
print(f"Input shape: [batch, 10, {input_dim}] | Target: [{output_dim}]")

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
best_val_loss = float('inf')
best_model_path = "models/temp_lstm_final.pt"
os.makedirs("models", exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_losses = []
    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            pred = model(X_val)
            val_preds.append(pred.numpy())
            val_targets.append(y_val.numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    val_mse = mean_squared_error(val_targets, val_preds)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)

    train_loss = np.mean(train_losses)
    print(f"Epoch {epoch}/{epochs} — Loss: {train_loss:.2f} | Val MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")

    # Save best model
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        torch.save(model.state_dict(), best_model_path)

print(f"Training complete. Best model saved to {best_model_path}")

model.load_state_dict(torch.load(best_model_path))
model.eval()

test_preds, test_targets = [], []
with torch.no_grad():
    for X_test, y_test in test_loader:
        pred = model(X_test)
        test_preds.append(pred.numpy())
        test_targets.append(y_test.numpy())

test_preds = np.vstack(test_preds)
test_targets = np.vstack(test_targets)

test_mse = mean_squared_error(test_targets, test_preds)
test_mae = mean_absolute_error(test_targets, test_preds)
test_r2 = r2_score(test_targets, test_preds)

print("\nTest Set Evaluation")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"R²: {test_r2:.4f}")
