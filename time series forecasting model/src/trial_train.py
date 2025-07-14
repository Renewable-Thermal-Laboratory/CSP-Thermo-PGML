import torch
from torch.utils.data import DataLoader
from trial_dataset import TempSequenceDataset
from trial_model import ImprovedTempModel
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random

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
    "patience": 15
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = TempSequenceDataset(
    "data/processed_theoretical",
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
    patience=5
)

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

    # Evaluate on validation set
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

    print(f"Epoch {epoch+1}/{config['epochs']}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val MSE: {val_mse:.2f} | MAE: {val_mae:.2f} | R²: {val_r2:.4f}")

    scheduler.step(val_mse)

    if val_mse < best_metrics["mse"]:
        best_metrics = {"mse": val_mse, "mae": val_mae, "r2": val_r2}
        torch.save(model.state_dict(), "models_theoretical/best_model.pth")
        print("Saved new best model")

    if epoch > config["patience"] and val_mse >= best_metrics["mse"]:
        print("Early stopping triggered")
        break

# Final test set evaluation
model.load_state_dict(torch.load("models_theoretical/best_model.pth"))
model.to(device)
model.eval()

X_test, y_test = dataset.get_test_data()
X_test, y_test = X_test.to(device), y_test.to(device)

with torch.no_grad():
    test_preds = model(X_test).cpu().numpy()
    test_targets = y_test.cpu().numpy()

test_preds_raw = dataset.thermal_scaler.inverse_transform(test_preds)
test_targets_raw = dataset.thermal_scaler.inverse_transform(test_targets)

mse = mean_squared_error(test_targets_raw, test_preds_raw)
mae = mean_absolute_error(test_targets_raw, test_preds_raw)
r2 = r2_score(test_targets_raw, test_preds_raw)

print("\n===== FINAL TEST METRICS =====")
print(f"Test MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

# Plotting unchanged from your code
os.makedirs("results", exist_ok=True)

# Plot sensor errors
from plot_residual_errors import plot_sensor_errors
plot_sensor_errors(test_preds_raw, test_targets_raw, dataset.thermal_scaler, label_prefix="10seq_")

# Plot all sensor predictions
plt.figure(figsize=(16, 20))
for i in range(10):
    plt.subplot(5, 2, i+1)
    plt.plot(test_targets_raw[:100, i], label="Actual", color='blue', alpha=0.7)
    plt.plot(test_preds_raw[:100, i], label="Predicted", color='red', linestyle='--', alpha=0.7)
    if hasattr(dataset.thermal_scaler, 'feature_names_in_'):
        plt.title(dataset.thermal_scaler.feature_names_in_[i])
    else:
        plt.title(f"Sensor {i+1}")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    if i >= 8:
        plt.xlabel("Time Steps")

plt.tight_layout()
plt.savefig("results/all_couple_predictions_theoretical.png", dpi=300)
plt.close()

# Residuals
residuals = test_targets_raw - test_preds_raw
plt.figure(figsize=(12, 6))
plt.hist(residuals.flatten(), bins=50, edgecolor='k')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (°C)")
plt.savefig("results/error_distribution_theoretical.png")
plt.close()
print("Saved plots.")

# Per-file evaluation unchanged
print("\nEvaluating on individual CSVs...")
from predicted_graph import process_file, load_model_and_scalers, print_numerical_results, plot_depth_profile

all_results = []
data_dir = "data/processed_theoretical"
output_dir = "results/predicted_graph_plots_theoretical"
os.makedirs(output_dir, exist_ok=True)

for filepath in dataset.test_files:
    try:
        result = process_file(filepath, model, dataset.thermal_scaler, dataset.param_scaler, config["sequence_length"])
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

print("\nTraining complete. Best validation metrics:")
print(f"MSE: {best_metrics['mse']:.2f}")
print(f"MAE: {best_metrics['mae']:.2f}")
print(f"R²: {best_metrics['r2']:.4f}")
