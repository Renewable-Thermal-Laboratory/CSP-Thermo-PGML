import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import mean_absolute_error
from plot_residual_errors import plot_sensor_errors
import matplotlib.pyplot as plt
import joblib
import os
from dataset_builder import TempSequenceDataset

# Configuration
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 3e-4
PATIENCE = 15
VALIDATION_SPLIT = 0.15
WEIGHT_DECAY = 1e-5
EPSILON = 1e-6

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Enhanced Model Architecture
class HighPrecisionTempModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Main LSTM Encoder
        self.encoder = nn.LSTM(input_size, 512, num_layers=3, 
                              batch_first=True, dropout=0.1)
        
        # Attention Mechanism
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Sensor-specific decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 1)
            ) for _ in range(output_size)
        ])
        
        # Temperature stabilization
        self.stabilizer = nn.Sequential(
            nn.Linear(output_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Encoder
        enc_out, _ = self.encoder(x)
        
        # Attention
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        
        # Sensor-specific predictions
        predictions = []
        for decoder in self.decoders:
            predictions.append(decoder(attn_out[:, -1, :]))
        
        combined = torch.cat(predictions, dim=1)
        
        # Stabilization
        stabilized = self.stabilizer(combined)
        
        return combined + 0.1 * stabilized  # Residual connection

# Physics-Informed Loss
class PrecisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = nn.L1Loss()
        self.temp_range = (20, 50)  # Expected temperature range
        
    def forward(self, preds, targets):
        # Base MAE
        loss = self.base_loss(preds, targets)
        
        # Temperature bounds penalty
        lower_penalty = torch.relu(self.temp_range[0] - preds).mean()
        upper_penalty = torch.relu(preds - self.temp_range[1]).mean()
        range_penalty = (lower_penalty + upper_penalty) * 0.3
        
        # Thermal gradient penalty (TC1 should be warmer than TC10)
        gradient_penalty = torch.relu(preds[:, -1] - preds[:, 0]).mean() * 0.2
        
        return loss + range_penalty + gradient_penalty

# Initialize dataset
dataset = TempSequenceDataset("data/processed", sequence_length=SEQUENCE_LENGTH)
input_size = dataset[0][0].shape[1]
output_size = dataset[0][1].shape[0]

# Data splits
train_size = int(0.85 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# Model and training setup
model = HighPrecisionTempModel(input_size, output_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = PrecisionLoss()

# Training loop
best_mae = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            val_preds.append(model(X))
            val_targets.append(y)
    
    val_preds = torch.cat(val_preds).numpy()
    val_targets = torch.cat(val_targets).numpy()
    
    # Inverse scaling
    val_preds_raw = dataset.thermal_scaler.inverse_transform(val_preds)
    val_targets_raw = dataset.thermal_scaler.inverse_transform(val_targets)
    
    # Calculate metrics
    val_mae = mean_absolute_error(val_targets_raw, val_preds_raw)
    sensor_maes = [
        mean_absolute_error(val_targets_raw[:, i], val_preds_raw[:, i]) 
        for i in range(output_size)
    ]
    
    scheduler.step(val_mae)
    
    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val MAE: {val_mae:.4f}°C")
    
    print("Per-sensor MAE:", [f"{mae:.3f}°C" for mae in sensor_maes])
    
    # Early stopping and model saving
    if all(mae < 1.0 for mae in sensor_maes):
        torch.save(model.state_dict(), "models/high_precision_model.pt")
        print("🎯 Target accuracy achieved! Model saved.")
        break
        
    if val_mae < best_mae:
        best_mae = val_mae
        patience_counter = 0
        torch.save(model.state_dict(), "models/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# Final Evaluation
model_path = "models/high_precision_model.pt" if os.path.exists("models/high_precision_model.pt") else "models/best_model.pt"
model.load_state_dict(torch.load(model_path))
model.eval()

test_preds, test_targets = [], []
with torch.no_grad():
    for X, y in test_loader:
        test_preds.append(model(X))
        test_targets.append(y)

test_preds = torch.cat(test_preds).numpy()
test_targets = torch.cat(test_targets).numpy()

# Inverse scaling
test_preds_raw = dataset.thermal_scaler.inverse_transform(test_preds)
test_targets_raw = dataset.thermal_scaler.inverse_transform(test_targets)

# Calculate final metrics
test_mae = mean_absolute_error(test_targets_raw, test_preds_raw)
sensor_maes = [
    mean_absolute_error(test_targets_raw[:, i], test_preds_raw[:, i]) 
    for i in range(output_size)
]

print("\n🔥 Final Test Results 🔥")
print(f"Overall MAE: {test_mae:.4f}°C")
print("Per-sensor MAE:")
for i, mae in enumerate(sensor_maes):
    print(f"  TC{i+1}: {mae:.4f}°C")

# Save results
results = {
    'overall_mae': test_mae,
    'per_sensor_mae': sensor_maes,
    'predictions': test_preds_raw,
    'targets': test_targets_raw
}
joblib.dump(results, "results/final_metrics.save")

# Plot results
plt.figure(figsize=(14, 8))
for i in range(min(output_size, 10)):  # Limit to 10 subplots
    plt.subplot(2, 5, i+1)
    plt.plot(test_targets_raw[:100, i], label='Actual', color='blue')
    plt.plot(test_preds_raw[:100, i], label='Predicted', color='red', linestyle='--')
    plt.title(f"TC{i+1} (MAE: {sensor_maes[i]:.2f}°C)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
plt.tight_layout()
plt.savefig("results/final_predictions.png", dpi=300)
plt.close()
print("\nSaved prediction plots to results/final_predictions.png")

plot_sensor_errors(test_preds_raw, test_targets_raw, dataset.thermal_scaler, label_prefix="10seq_")

# 2. Plot all sensor predictions
plt.figure(figsize=(16, 20))
for i in range(10):  # For all 10 sensors
    plt.subplot(5, 2, i+1)
    plt.plot(test_targets_raw[:100, i], label="Actual", color='blue', alpha=0.7)
    plt.plot(test_preds_raw[:100, i], label="Predicted", color='red', linestyle='--', alpha=0.7)
    
    # Use feature names if available
    sensor_name = dataset.thermal_scaler.feature_names_in_[i] if hasattr(dataset.thermal_scaler, 'feature_names_in_') else f"TC{i+1}"
    plt.title(f"{sensor_name} (MAE: {sensor_maes[i]:.2f}°C)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    
    if i >= 8:  
        plt.xlabel("Time Steps")

plt.tight_layout()
plt.savefig("results/all_couple_predictions.png", dpi=300)
plt.close()