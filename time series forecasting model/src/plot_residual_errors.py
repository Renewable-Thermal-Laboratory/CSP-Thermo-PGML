
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def plot_sensor_errors(preds_raw, targets_raw, thermal_scaler, output_dir="results_errors", label_prefix=""):
    os.makedirs(output_dir, exist_ok=True)

    # Sensor names
    sensor_labels = thermal_scaler.feature_names_in_ if hasattr(thermal_scaler, 'feature_names_in_') else [f"TC{i+1}" for i in range(preds_raw.shape[1])]

    # Compute residuals
    residuals = preds_raw - targets_raw
    sensor_mse = np.mean(residuals ** 2, axis=0)
    sensor_mae = np.mean(np.abs(residuals), axis=0)

    # === Plot MSE and MAE bar charts ===
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.bar(sensor_labels, sensor_mse, color='coral')
    plt.title("Per-Sensor MSE")
    plt.ylabel("MSE")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(sensor_labels, sensor_mae, color='skyblue')
    plt.title("Per-Sensor MAE")
    plt.ylabel("MAE (°C)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{label_prefix}sensor_error_summary.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved per-sensor residual plot to: {out_path}")


# === Example Usage ===
# from plot_residual_errors import plot_sensor_errors
# plot_sensor_errors(preds_raw, targets_raw, thermal_scaler, label_prefix="10seq_")
