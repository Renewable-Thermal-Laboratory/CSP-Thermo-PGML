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

# Fix for multiprocessing on macOS
import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set

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

# Enhanced Model Architecture with TC1 Focus
class HighPrecisionTempModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Main LSTM Encoder with increased capacity
        self.encoder = nn.LSTM(input_size, 768, num_layers=4, 
                              batch_first=True, dropout=0.15)
        
        # Attention Mechanism
        self.attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        
        # Special decoder for TC1 (problematic sensor)
        self.tc1_decoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Standard decoders for other sensors (TC2-TC10)
        self.other_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 256),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 1)
            ) for _ in range(output_size - 1)
        ])
        
        # Temperature stabilization with sensor-specific weights
        self.stabilizer = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        # Sensor-specific scaling factors (learnable)
        self.sensor_scales = nn.Parameter(torch.ones(output_size))
        self.sensor_biases = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        # Encoder
        enc_out, _ = self.encoder(x)
        enc_out = self.norm1(enc_out)
        
        # Attention
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        attn_out = self.norm2(attn_out + enc_out)  # Residual connection
        
        # Final representation
        final_repr = attn_out[:, -1, :]
        
        # TC1 special handling
        tc1_pred = self.tc1_decoder(final_repr)
        
        # Other sensors
        other_preds = []
        for decoder in self.other_decoders:
            other_preds.append(decoder(final_repr))
        
        # Combine all predictions
        combined = torch.cat([tc1_pred] + other_preds, dim=1)
        
        # Apply sensor-specific scaling
        scaled = combined * self.sensor_scales + self.sensor_biases
        
        # Stabilization
        stabilized = self.stabilizer(scaled)
        
        return scaled + 0.05 * stabilized  # Reduced residual weight

# Enhanced Physics-Informed Loss with TC1 Focus
class PrecisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = nn.SmoothL1Loss()  # More robust to outliers
        self.temp_range = (15, 55)  # Expanded range for TC1
        
        # Sensor-specific loss weights (higher weight for TC1)
        self.sensor_weights = torch.tensor([3.0] + [1.0] * 9)  # TC1 gets 3x weight
        
    def forward(self, preds, targets):
        # Weighted MAE per sensor
        losses = []
        for i in range(preds.size(1)):
            sensor_loss = F.l1_loss(preds[:, i], targets[:, i])
            losses.append(sensor_loss * self.sensor_weights[i])
        
        weighted_loss = torch.stack(losses).mean()
        
        # Temperature bounds penalty (more lenient for TC1)
        lower_bounds = torch.tensor([10.0] + [18.0] * 9).to(preds.device)
        upper_bounds = torch.tensor([60.0] + [52.0] * 9).to(preds.device)
        
        lower_penalty = torch.relu(lower_bounds - preds).mean()
        upper_penalty = torch.relu(preds - upper_bounds).mean()
        range_penalty = (lower_penalty + upper_penalty) * 0.2
        
        # Thermal gradient penalty (TC1 should be warmer than TC10)
        gradient_penalty = torch.relu(preds[:, -1] - preds[:, 0] + 5.0).mean() * 0.1
        
        # Smoothness penalty for TC1 (reduce erratic predictions)
        tc1_smoothness = torch.mean(torch.abs(preds[1:, 0] - preds[:-1, 0])) * 0.3
        
        return weighted_loss + range_penalty + gradient_penalty + tc1_smoothness

# Data preprocessing function
def preprocess_data(dataset):
    """Check and fix data quality issues, especially for TC1"""
    print("Analyzing data quality...")
    
    # Get a sample of data to analyze
    sample_size = min(1000, len(dataset))
    sample_data = []
    
    for i in range(sample_size):
        _, targets = dataset[i]
        sample_data.append(targets.numpy())
    
    sample_data = np.array(sample_data)
    
    # Analyze each sensor
    for i in range(sample_data.shape[1]):
        sensor_data = sample_data[:, i]
        print(f"TC{i+1}: mean={sensor_data.mean():.2f}, std={sensor_data.std():.2f}, "
              f"min={sensor_data.min():.2f}, max={sensor_data.max():.2f}")
        
        # Check for anomalies
        q1, q3 = np.percentile(sensor_data, [25, 75])
        iqr = q3 - q1
        outliers = np.sum((sensor_data < q1 - 1.5*iqr) | (sensor_data > q3 + 1.5*iqr))
        if outliers > 0:
            print(f"  ⚠️  TC{i+1} has {outliers} potential outliers ({outliers/len(sensor_data)*100:.1f}%)")

# Initialize dataset with preprocessing
dataset = TempSequenceDataset("data/processed", sequence_length=SEQUENCE_LENGTH)
preprocess_data(dataset)

input_size = dataset[0][0].shape[1]
output_size = dataset[0][1].shape[0]
print(f"Input size: {input_size}, Output size: {output_size}")

# Data splits
train_size = int(0.85 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible splits
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

# Model and training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HighPrecisionTempModel(input_size, output_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Cosine annealing scheduler for better convergence
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

criterion = PrecisionLoss().to(device)

# Training loop with improved monitoring
best_mae = float('inf')
patience_counter = 0
train_losses = []
val_maes = []

print("\n🚀 Starting training...")

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    batch_count = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        train_loss += loss.item()
        batch_count += 1
    
    scheduler.step()
    avg_train_loss = train_loss / batch_count
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            val_preds.append(model(X).cpu())
            val_targets.append(y.cpu())
    
    val_preds = torch.cat(val_preds).numpy()
    val_targets = torch.cat(val_targets).numpy()
    
    # Inverse scaling
    val_preds_raw = dataset.thermal_scaler.inverse_transform(val_preds)
    val_targets_raw = dataset.thermal_scaler.inverse_transform(val_targets)
    
    # Calculate metrics
    val_mae = mean_absolute_error(val_targets_raw, val_preds_raw)
    val_maes.append(val_mae)
    
    sensor_maes = [
        mean_absolute_error(val_targets_raw[:, i], val_preds_raw[:, i]) 
        for i in range(output_size)
    ]
    
    # Enhanced progress reporting
    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f}°C | "
              f"TC1 MAE: {sensor_maes[0]:.4f}°C | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        print("  Per-sensor MAE:", [f"{mae:.3f}" for mae in sensor_maes])
    
    # Early stopping with TC1-aware criteria
    tc1_mae = sensor_maes[0]
    other_maes = sensor_maes[1:]
    
    # Success criteria: TC1 < 2.0°C AND all others < 1.0°C
    if tc1_mae < 2.0 and all(mae < 1.0 for mae in other_maes):
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_mae': val_mae,
            'sensor_maes': sensor_maes
        }, "models/high_precision_model.pt")
        print(f"🎯 Target accuracy achieved at epoch {epoch+1}! Model saved.")
        break
        
    if val_mae < best_mae:
        best_mae = val_mae
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_mae': val_mae,
            'sensor_maes': sensor_maes
        }, "models/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Load best model for final evaluation
model_path = "models/high_precision_model.pt" if os.path.exists("models/high_precision_model.pt") else "models/best_model.pt"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n📊 Loading model from epoch {checkpoint.get('epoch', 'unknown')}")

# Final Evaluation
test_preds, test_targets = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        test_preds.append(model(X).cpu())
        test_targets.append(y.cpu())

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
    status = "✅" if (i == 0 and mae < 2.0) or (i > 0 and mae < 1.0) else "❌"
    print(f"  TC{i+1}: {mae:.4f}°C {status}")

# Detailed TC1 analysis
tc1_errors = np.abs(test_preds_raw[:, 0] - test_targets_raw[:, 0])
print(f"\n🔍 TC1 Detailed Analysis:")
print(f"  Mean error: {tc1_errors.mean():.4f}°C")
print(f"  Median error: {np.median(tc1_errors):.4f}°C")
print(f"  95th percentile error: {np.percentile(tc1_errors, 95):.4f}°C")
print(f"  Max error: {tc1_errors.max():.4f}°C")

# Save results
results = {
    'overall_mae': test_mae,
    'per_sensor_mae': sensor_maes,
    'predictions': test_preds_raw,
    'targets': test_targets_raw,
    'train_losses': train_losses,
    'val_maes': val_maes
}
joblib.dump(results, "results/final_metrics.save")

# Enhanced plotting
plt.figure(figsize=(16, 12))

# Training curves
plt.subplot(3, 1, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(val_maes, label='Validation MAE', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss / MAE')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

# TC1 specific plot (most important)
plt.subplot(3, 1, 2)
sample_range = slice(0, 200)  # Show more samples
plt.plot(test_targets_raw[sample_range, 0], label='TC1 Actual', color='blue', linewidth=2)
plt.plot(test_preds_raw[sample_range, 0], label='TC1 Predicted', color='red', linestyle='--', linewidth=2)
plt.title(f"TC1 Predictions (MAE: {sensor_maes[0]:.4f}°C)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Time Steps")
plt.legend()
plt.grid(True, alpha=0.3)

# Error distribution for TC1
plt.subplot(3, 1, 3)
plt.hist(tc1_errors, bins=50, alpha=0.7, color='red', edgecolor='black')
plt.axvline(tc1_errors.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {tc1_errors.mean():.3f}°C')
plt.axvline(np.median(tc1_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(tc1_errors):.3f}°C')
plt.xlabel('Absolute Error (°C)')
plt.ylabel('Frequency')
plt.title('TC1 Error Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/enhanced_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# Original plots
plt.figure(figsize=(14, 8))
for i in range(min(output_size, 10)):
    plt.subplot(2, 5, i+1)
    plt.plot(test_targets_raw[:100, i], label='Actual', color='blue')
    plt.plot(test_preds_raw[:100, i], label='Predicted', color='red', linestyle='--')
    plt.title(f"TC{i+1} (MAE: {sensor_maes[i]:.2f}°C)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
plt.tight_layout()
plt.savefig("results/final_predictions.png", dpi=300)
plt.close()

plot_sensor_errors(test_preds_raw, test_targets_raw, dataset.thermal_scaler, label_prefix="fixed_10seq_")

print(f"\n💾 Results saved to results/ directory")
print("🎨 Enhanced analysis plot saved as results/enhanced_analysis.png")
