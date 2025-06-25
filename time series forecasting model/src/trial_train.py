import torch
from torch.utils.data import DataLoader
from trial_dataset import TempSequenceDataset
from trial_model import ImprovedTempModel
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
from predicted_graph import process_file, load_model_and_scalers  # Import from prediction module
import glob

# Configuration
config = {
    "sequence_length": 20,
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "patience": 15
}

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TempSequenceDataset(
    "data/processed_H6",
    sequence_length=config["sequence_length"]
)

# Data loaders
train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

# Model
model = ImprovedTempModel(
    input_size=dataset[0][0].shape[-1],
    output_size=dataset[0][1].shape[-1]
).to(device)

# Optimizer and loss
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)
criterion = nn.HuberLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5
)

# Training loop
best_metrics = {"mse": float('inf'), "mae": float('inf'), "r2": -float('inf')}

for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    X_test, y_test = dataset.get_test_data()
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        test_preds = model(X_test).cpu().numpy()
        test_targets = y_test.cpu().numpy()

    # Inverse transform
    test_preds_raw = dataset.thermal_scaler.inverse_transform(test_preds)
    test_targets_raw = dataset.thermal_scaler.inverse_transform(test_targets)

    # Metrics
    mse = mean_squared_error(test_targets_raw, test_preds_raw)
    mae = mean_absolute_error(test_targets_raw, test_preds_raw)
    r2 = r2_score(test_targets_raw, test_preds_raw)

    print(f"Epoch {epoch+1}/{config['epochs']}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Test MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

    # Update scheduler
    scheduler.step(mse)

    # Save best model
    if mse < best_metrics["mse"]:
        best_metrics = {"mse": mse, "mae": mae, "r2": r2}
        torch.save(model.state_dict(), "models/best_model.pth")
        print("Saved new best model")

    # Early stopping
    if epoch > config["patience"] and mse >= best_metrics["mse"]:
        print("Early stopping triggered")
        break

# ====== Visualization Code ======
os.makedirs("results", exist_ok=True)

# 1. Plot sensor errors
from plot_residual_errors import plot_sensor_errors
plot_sensor_errors(test_preds_raw, test_targets_raw, dataset.thermal_scaler, label_prefix="10seq_")

# 2. Plot all sensor predictions
plt.figure(figsize=(16, 20))
for i in range(10):  # For all 10 couples
    plt.subplot(5, 2, i+1)
    plt.plot(test_targets_raw[:100, i], label="Actual", color='blue', alpha=0.7)
    plt.plot(test_preds_raw[:100, i], label="Predicted", color='red', linestyle='--', alpha=0.7)

    # Use feature names if available
    if hasattr(dataset.thermal_scaler, 'feature_names_in_'):
        plt.title(dataset.thermal_scaler.feature_names_in_[i])
    else:
        plt.title(f"Sensor {i+1}")

    plt.ylabel("Temperature (°C)")
    plt.legend()

    if i >= 8:
        plt.xlabel("Time Steps")

plt.tight_layout()
plt.savefig("results/all_couple_predictions.png", dpi=300)
plt.close()
print("Saved all channel predictions plot to results/all_couple_predictions.png")

# 3. Residual analysis
residuals = test_targets_raw - test_preds_raw
plt.figure(figsize=(12, 6))
plt.hist(residuals.flatten(), bins=50, edgecolor='k')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (°C)")
plt.savefig("results/error_distribution.png")
plt.close()
print("Saved error distribution plot to results/error_distribution.png")

# ========== Post-Training Evaluation on CSV Test Files ==========
print("\nEvaluating on individual CSVs...")
model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device("cpu")))
model.eval()

all_results = []
data_dir = "data/processed_H6"
output_dir = "results/predicted_graph_plots"
os.makedirs(output_dir, exist_ok=True)

for filepath in glob.glob(os.path.join(data_dir, "*.csv")):
    try:
        result = process_file(filepath, model, dataset.thermal_scaler, dataset.param_scaler, config["sequence_length"])
        all_results.append(result)
    except Exception as e:
        print(f"Error in {filepath}: {e}")

all_results.sort(key=lambda x: x['avg_residual'])

print("\nTop 5 Best Predictions:")
print("Filename".ljust(45), "Avg Residual (°C)")
for r in all_results[:5]:
    print(f"{r['filename']:<45} {r['avg_residual']:.3f}")

from predicted_graph import print_numerical_results, plot_depth_profile

for i, result in enumerate(all_results[:5]):
    print_numerical_results(result)
    save_path = os.path.join(output_dir, f"{result['filename'].replace('.csv', '')}.png")
    plot_depth_profile(result, save_path)
    print(f"Saved profile plot to {save_path}")

print("\nTraining complete. Best metrics:")
print(f"MSE: {best_metrics['mse']:.2f}")
print(f"MAE: {best_metrics['mae']:.2f}")
print(f"R²: {best_metrics['r2']:.4f}")
