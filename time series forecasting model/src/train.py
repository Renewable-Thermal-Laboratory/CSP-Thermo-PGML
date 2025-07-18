import torch
from torch.utils.data import DataLoader
from dataset_builder import TempSequenceDataset
from model import ImprovedTempModel
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
import json

# Import updated physics loss functions
from physics_loss import (
    compute_energy_storage_loss_20sec,
    compute_energy_storage_loss_debug_20sec,
    get_adaptive_physics_weight,
    track_physics_metrics
)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config
config = {
    "sequence_length": 20,
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "patience": 25,  # Increased patience for physics-informed learning
    "initial_lambda_physics": 2.0,  # Starting weight, will be adaptive
    "delta_t": 20.0,  # 20-second intervals
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = TempSequenceDataset(
    "data/processed_H6",
    sequence_length=config["sequence_length"]
)

train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

model = ImprovedTempModel(
    input_size=dataset[0][0].shape[-1],
    output_size=dataset[0][1].shape[-1]
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)
criterion = nn.HuberLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=8,
    factor=0.5
)

# PHYSICS CONSTANTS
rho = 1836.31  # kg/m³
cp = 1512      # J/(kg·K)
diameter = 0.1035  # m
radius = diameter / 2  # m
delta_t = config["delta_t"]  # 20 seconds

# Training metrics tracking
training_metrics = {
    'epochs': [],
    'train_loss': [],
    'data_loss': [],
    'physics_loss': [],
    'adaptive_weight': [],
    'val_mse': [],
    'val_mae': [],
    'val_r2': [],
    'physics_metrics': []
}

best_metrics = {"mse": float('inf'), "mae": float('inf'), "r2": -float('inf')}
best_physics_loss = float('inf')

print("Starting training with adaptive physics weight and 20-second delta T...")
print(f"Initial configuration: {config}")

for epoch in range(config["epochs"]):
    model.train()
    epoch_train_loss = 0
    epoch_data_loss = 0
    epoch_physics_loss = 0
    epoch_adaptive_weights = []
    
    batch_count = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        # Get predictions
        outputs = model(X)
        
        # Data loss
        loss_data = criterion(outputs, y)
        
        # Get initial temperatures (20 seconds ago) - this is the key change
        # X shape: [batch, sequence_length, features]
        # We want T at t=0 (20 seconds ago) instead of t=19 (1 second ago)
        initial_temps = X[:, 0, :10]  # T at beginning of sequence (t=0)
        
        # Extract scaled h_total
        h_scaled = X[:, -1, 10]
        
        # Inverse-transform h_total to get real height in meters
        zeros_params = torch.zeros((h_scaled.shape[0], 4), device=h_scaled.device)
        scaled_params = torch.cat([h_scaled.unsqueeze(1), zeros_params], dim=1)
        
        # Inverse-transform using param_scaler
        h_total_raw_np = dataset.param_scaler.inverse_transform(
            scaled_params.detach().cpu().numpy()
        )
        h_total_raw = torch.tensor(
            h_total_raw_np[:, 0], device=h_scaled.device, dtype=torch.float32
        )
        
        # Calculate physics loss with 20-second intervals
        if epoch == 0 and batch_idx == 0:
            print(f"\nFirst batch debug - using 20-second intervals:")
            print(f"Initial temps shape: {initial_temps.shape}")
            print(f"Current temps shape: {y.shape}")
            print(f"Predicted temps shape: {outputs.shape}")
            
            # Debug call for first batch
            loss_physics = compute_energy_storage_loss_debug_20sec(
                outputs,
                y,
                initial_temps,
                dataset.thermal_scaler,
                rho,
                cp,
                h_total_raw,
                radius,
                delta_t
            )
        else:
            # Regular physics loss calculation
            loss_physics, _ = compute_energy_storage_loss_20sec(
                outputs,
                y,
                initial_temps,
                dataset.thermal_scaler,
                rho,
                cp,
                h_total_raw,
                radius,
                delta_t
            )
        
        # Get adaptive physics weight
        adaptive_weight = get_adaptive_physics_weight(
            loss_physics.item(),
            loss_data.item(),
            epoch
        )
        
        # Combined loss
        loss = loss_data + adaptive_weight * loss_physics
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        epoch_train_loss += loss.item()
        epoch_data_loss += loss_data.item()
        epoch_physics_loss += loss_physics.item()
        epoch_adaptive_weights.append(adaptive_weight)
        batch_count += 1
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                  f"Data Loss: {loss_data.item():.4f}, "
                  f"Physics Loss: {loss_physics.item():.2f}, "
                  f"Adaptive Weight: {adaptive_weight:.2f}")
    
    # Average metrics for the epoch
    avg_train_loss = epoch_train_loss / batch_count
    avg_data_loss = epoch_data_loss / batch_count
    avg_physics_loss = epoch_physics_loss / batch_count
    avg_adaptive_weight = np.mean(epoch_adaptive_weights)
    
    # Validation
    model.eval()
    X_val, y_val = dataset.get_val_data()
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    with torch.no_grad():
        val_preds = model(X_val).cpu().numpy()
        val_targets = y_val.cpu().numpy()
    
    val_preds_raw = dataset.thermal_scaler.inverse_transform(val_preds)
    val_targets_raw = dataset.thermal_scaler.inverse_transform(val_targets)
    
    val_mse = mean_squared_error(val_targets_raw, val_preds_raw)
    val_mae = mean_absolute_error(val_targets_raw, val_preds_raw)
    val_r2 = r2_score(val_targets_raw, val_preds_raw)
    
    # Calculate validation physics metrics
    val_initial_temps = X_val[:, 0, :10]  # T at beginning of sequence
    val_h_scaled = X_val[:, -1, 10]
    
    val_zeros_params = torch.zeros((val_h_scaled.shape[0], 4), device=val_h_scaled.device)
    val_scaled_params = torch.cat([val_h_scaled.unsqueeze(1), val_zeros_params], dim=1)
    val_h_total_raw_np = dataset.param_scaler.inverse_transform(
        val_scaled_params.detach().cpu().numpy()
    )
    val_h_total_raw = torch.tensor(
        val_h_total_raw_np[:, 0], device=val_h_scaled.device, dtype=torch.float32
    )
    
    val_physics_metrics = track_physics_metrics(
        torch.tensor(val_preds).to(device),
        y_val,
        val_initial_temps,
        dataset.thermal_scaler,
        rho, cp, val_h_total_raw, radius, delta_t
    )
    
    # Store metrics
    training_metrics['epochs'].append(epoch + 1)
    training_metrics['train_loss'].append(avg_train_loss)
    training_metrics['data_loss'].append(avg_data_loss)
    training_metrics['physics_loss'].append(avg_physics_loss)
    training_metrics['adaptive_weight'].append(avg_adaptive_weight)
    training_metrics['val_mse'].append(val_mse)
    training_metrics['val_mae'].append(val_mae)
    training_metrics['val_r2'].append(val_r2)
    training_metrics['physics_metrics'].append(val_physics_metrics)
    
    # Print epoch summary
    print(f"\n=== EPOCH {epoch+1}/{config['epochs']} SUMMARY ===")
    print(f"Train Loss: {avg_train_loss:.4f} (Data: {avg_data_loss:.4f}, Physics: {avg_physics_loss:.2f})")
    print(f"Adaptive Weight: {avg_adaptive_weight:.2f}")
    print(f"Validation - MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    print(f"Physics Metrics - Loss: {val_physics_metrics['total_physics_loss']:.2f}, "
          f"Energy Balance: {val_physics_metrics['energy_balance_ratio']:.3f}")
    print(f"Max Bin Error: {val_physics_metrics['max_bin_error']:.2f}, "
          f"Avg Bin Error: {val_physics_metrics['avg_bin_error']:.2f}")
    
    # Learning rate scheduling
    scheduler.step(val_mse)
    
    # Save best model based on validation MSE
    if val_mse < best_metrics["mse"]:
        best_metrics = {"mse": val_mse, "mae": val_mae, "r2": val_r2}
        best_physics_loss = val_physics_metrics['total_physics_loss']
        torch.save(model.state_dict(), "models/best_model.pth")
        print("*** SAVED NEW BEST MODEL ***")
    
    # Early stopping check
    if epoch >= config["patience"]:
        recent_mse = training_metrics['val_mse'][-config["patience"]:]
        if all(mse >= best_metrics["mse"] for mse in recent_mse):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print(f"\n=== TRAINING COMPLETE ===")
print(f"Best validation metrics: MSE={best_metrics['mse']:.2f}, MAE={best_metrics['mae']:.2f}, R²={best_metrics['r2']:.4f}")
print(f"Best physics loss: {best_physics_loss:.2f}")

# Save training metrics
os.makedirs("results", exist_ok=True)
with open("results/training_metrics.json", "w") as f:
    json.dump(training_metrics, f, indent=2)

# Final test set evaluation
print("\n=== FINAL TEST EVALUATION ===")
model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

X_test, y_test = dataset.get_test_data()
X_test, y_test = X_test.to(device), y_test.to(device)

with torch.no_grad():
    test_preds = model(X_test).cpu().numpy()
    test_targets = y_test.cpu().numpy()

test_preds_raw = dataset.thermal_scaler.inverse_transform(test_preds)
test_targets_raw = dataset.thermal_scaler.inverse_transform(test_targets)

test_mse = mean_squared_error(test_targets_raw, test_preds_raw)
test_mae = mean_absolute_error(test_targets_raw, test_preds_raw)
test_r2 = r2_score(test_targets_raw, test_preds_raw)

print(f"Test MSE: {test_mse:.2f} | MAE: {test_mae:.2f} | R²: {test_r2:.4f}")

# Calculate final test physics metrics
test_initial_temps = X_test[:, 0, :10]  # T at beginning of sequence
test_h_scaled = X_test[:, -1, 10]

test_zeros_params = torch.zeros((test_h_scaled.shape[0], 4), device=test_h_scaled.device)
test_scaled_params = torch.cat([test_h_scaled.unsqueeze(1), test_zeros_params], dim=1)
test_h_total_raw_np = dataset.param_scaler.inverse_transform(
    test_scaled_params.detach().cpu().numpy()
)
test_h_total_raw = torch.tensor(
    test_h_total_raw_np[:, 0], device=test_h_scaled.device, dtype=torch.float32
)

final_test_physics_metrics = track_physics_metrics(
    torch.tensor(test_preds).to(device),
    y_test,
    test_initial_temps,
    dataset.thermal_scaler,
    rho, cp, test_h_total_raw, radius, delta_t
)

print(f"\nFinal Test Physics Metrics:")
print(f"Physics Loss: {final_test_physics_metrics['total_physics_loss']:.2f}")
print(f"Energy Balance Ratio: {final_test_physics_metrics['energy_balance_ratio']:.3f}")
print(f"Max Bin Error: {final_test_physics_metrics['max_bin_error']:.2f}")
print(f"Average Bin Error: {final_test_physics_metrics['avg_bin_error']:.2f}")

# Plot training metrics
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(training_metrics['epochs'], training_metrics['train_loss'], 'b-', label='Total Loss')
plt.plot(training_metrics['epochs'], training_metrics['data_loss'], 'g-', label='Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(training_metrics['epochs'], training_metrics['physics_loss'], 'r-', label='Physics Loss')
plt.xlabel('Epoch')
plt.ylabel('Physics Loss')
plt.title('Physics Loss Evolution')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(training_metrics['epochs'], training_metrics['adaptive_weight'], 'purple', label='Adaptive Weight')
plt.xlabel('Epoch')
plt.ylabel('Physics Weight')
plt.title('Adaptive Physics Weight')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(training_metrics['epochs'], training_metrics['val_mse'], 'orange', label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Validation MSE')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(training_metrics['epochs'], training_metrics['val_r2'], 'cyan', label='Validation R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('Validation R²')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
energy_balance_ratios = [m['energy_balance_ratio'] for m in training_metrics['physics_metrics']]
plt.plot(training_metrics['epochs'], energy_balance_ratios, 'brown', label='Energy Balance Ratio')
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
plt.xlabel('Epoch')
plt.ylabel('Ratio')
plt.title('Energy Balance Ratio')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/training_metrics_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Continue with existing prediction graph generation...
print("\nEvaluating on test set CSVs...")

from predicted_graph import process_file, load_model_and_scalers, print_numerical_results, plot_depth_profile

all_results = []
output_dir = "results/predicted_graph_plots"
os.makedirs(output_dir, exist_ok=True)

for filepath in dataset.test_files:
    try:
        result = process_file(
            filepath,
            model,
            dataset.thermal_scaler,
            dataset.param_scaler,
            config["sequence_length"]
        )
        all_results.append(result)
    except Exception as e:
        print(f"Error in {filepath}: {e}")

all_results.sort(key=lambda x: x['avg_residual'])

print("\nResults for ALL Test Files:")
for r in all_results:
    print(f"{r['filename']:<45} {r['avg_residual']:.3f}")

for i, result in enumerate(all_results):
    print_numerical_results(result)
    save_path = os.path.join(output_dir, f"{result['filename'].replace('.csv', '')}.png")
    plot_depth_profile(result, save_path)
    print(f"Saved profile plot to {save_path}")

print(f"\nTraining complete! Check results/ directory for detailed metrics and plots.")