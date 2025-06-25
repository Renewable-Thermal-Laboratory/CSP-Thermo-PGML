import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset_builder import TempSequenceDataset
from plot_residual_errors import plot_sensor_errors
from model import TempLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
VALIDATION_SPLIT = 0.2
WEIGHT_DECAY = 1e-4

# Initialize dataset
dataset = TempSequenceDataset("data/processed", sequence_length=SEQUENCE_LENGTH)
input_size = dataset[0][0].shape[1]
output_size = dataset[0][1].shape[0]

# Train/val split
val_size = int(len(dataset) * VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

print(f"Loaded dataset: {len(dataset)} samples")
print(f"Input shape: [batch, {SEQUENCE_LENGTH}, {input_size}], Target: [{output_size}]")

# Training loop
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

# ===== NEW TEST SET EVALUATION =====
print("\n=== Evaluating on Test Set ===")

# Load best model
model.load_state_dict(torch.load("models/temp_lstm_final.pt"))
model.eval()

# Get test samples and convert to tensors
test_samples = dataset.get_test_dataset()
test_X = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in test_samples])
test_y = torch.stack([torch.tensor(x[1], dtype=torch.float32) for x in test_samples])

# Create DataLoader for batch processing
test_loader = DataLoader(torch.utils.data.TensorDataset(test_X, test_y), 
                       batch_size=BATCH_SIZE)

# Run inference
test_preds, test_targets = [], []
with torch.no_grad():
    for X, y in test_loader:
        pred = model(X)
        test_preds.append(pred)
        test_targets.append(y)

test_preds = torch.cat(test_preds).numpy()
test_targets = torch.cat(test_targets).numpy()

# Inverse transform to original scale
test_preds_raw = dataset.thermal_scaler.inverse_transform(test_preds)
test_targets_raw = dataset.thermal_scaler.inverse_transform(test_targets)

# Calculate metrics
test_mse = mean_squared_error(test_targets_raw, test_preds_raw)
test_mae = mean_absolute_error(test_targets_raw, test_preds_raw)
test_r2 = r2_score(test_targets_raw, test_preds_raw)

print(f"\nTest Metrics (Original Scale):")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f} °C")
print(f"R²: {test_r2:.4f}")
plot_sensor_errors(test_preds_raw, test_targets_raw, dataset.thermal_scaler, label_prefix="10seq_")

plt.figure(figsize=(16, 20))
for i in range(10):  # For all 10 couples
        plt.subplot(5, 2, i+1)
        plt.plot(test_targets_raw[:100, i], label="Actual", color='blue', alpha=0.7)
        plt.plot(test_preds_raw[:100, i], label="Predicted", color='red', linestyle='--', alpha=0.7)
        plt.title(f"{dataset.thermal_scaler.feature_names_in_[i]}" if hasattr(dataset.thermal_scaler, 'feature_names_in_') 
                 else f"couple {i+1}")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        
        if i >= 8:  
            plt.xlabel("Time Steps")
    
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/all_couple_predictions.png", dpi=300)
print("\nSaved all channel predictions plot to results/all_couple_predictions.png")

# Residual Analysis
residuals = test_targets_raw - test_preds_raw
plt.figure(figsize=(12, 6))
plt.hist(residuals.flatten(), bins=50, edgecolor='k')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (°C)")
plt.savefig("results/error_distribution.png")