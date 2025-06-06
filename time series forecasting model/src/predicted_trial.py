import torch
import joblib
import numpy as np
import pandas as pd
import re
import os
import glob
from trial_model import ImprovedTempModel

# Thermal and parameter feature definitions
THERMAL_COLS = ["TC1_tip", "TC2", "TC3", "TC4", "TC5", 
                "TC6", "TC7", "TC8", "TC9", "TC10"]
PARAM_PATTERN = r"h(\d+).*?flux(\d+).*?abs(\d+).*?surf.*?(\d+).*?(\d+)s"

# Mappings from preprocessing
h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
flux_map = {88: 25900, 78: 21250, 73: 19400}
abs_map = {0: 3, 92: 100}
surf_map = {0: 0.98, 1: 0.76}

def extract_params_from_filename(filename):
    match = re.search(PARAM_PATTERN, filename)
    if not match:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")
    h = h_map.get(int(match.group(1)), int(match.group(1)))
    flux = flux_map.get(int(match.group(2)), int(match.group(2)))
    abs_val = abs_map.get(int(match.group(3)), int(match.group(3)))
    surf = surf_map.get(int(match.group(4)), int(match.group(4)))
    start_time = int(match.group(5))
    return [h, flux, abs_val, surf, start_time]

def load_model_and_scalers():
    model = ImprovedTempModel(input_size=15, output_size=10)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("models/thermal_scaler.save")
    param_scaler = joblib.load("models/param_scaler.save")
    return model, scaler, param_scaler

def predict_with_residuals(csv_path, model, scaler, param_scaler):
    filename = os.path.basename(csv_path)
    try:
        static_params = extract_params_from_filename(filename)
    except ValueError:
        return None

    try:
        df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        if len(df) < 11:
            return None

        sequence = df[THERMAL_COLS].iloc[-11:-1]
        actual_values = df[THERMAL_COLS].iloc[-1].values
        sequence_scaled = scaler.transform(sequence)
        static_scaled = param_scaler.transform([static_params])[0]

        X = [list(row) + static_scaled.tolist() for row in sequence_scaled]
        X_tensor = torch.tensor([X], dtype=torch.float32)

        with torch.no_grad():
            prediction_scaled = model(X_tensor).numpy()

        predicted_values = scaler.inverse_transform(prediction_scaled)[0]
        residuals = actual_values - predicted_values

        return {
            'filename': filename,
            'predicted_values': predicted_values,
            'actual_values': actual_values,
            'residuals': residuals,
            'mae': np.mean(np.abs(residuals)),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'max_residual': np.max(np.abs(residuals)),
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual_raw': np.max(residuals)
        }
    except Exception as e:
        print(f"Error in {filename}: {e}")
        return None

def evaluate_all(data_dir="data/processed", top_n=5):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        print("No CSVs found.")
        return

    model, scaler, param_scaler = load_model_and_scalers()
    results = [predict_with_residuals(f, model, scaler, param_scaler) for f in files]
    results = [r for r in results if r is not None]
    if not results:
        print("No successful evaluations.")
        return

    results.sort(key=lambda x: x['max_residual'])
    max_res = [r['max_residual'] for r in results]

    print("\n=== Max Residual Stats ===")
    print(f"Min: {min(max_res):.2f} | Max: {max(max_res):.2f} | Mean: {np.mean(max_res):.2f} | Median: {np.median(max_res):.2f}")

    print(f"\n=== Top {top_n} Files ===")
    for i, r in enumerate(results[:top_n], 1):
        print(f"\n#{i} File: {r['filename']}")
        print(f"Max Residual: {r['max_residual']:.2f} | MAE: {r['mae']:.2f} | RMSE: {r['rmse']:.2f}")
        print("Sensor       Pred      Actual    Residual  AbsResidual")
        print("------------------------------------------------------")
        for j, sensor in enumerate(THERMAL_COLS):
            abs_res = abs(r['residuals'][j])
            print(f"{sensor:<12}{r['predicted_values'][j]:<10.2f}{r['actual_values'][j]:<10.2f}{r['residuals'][j]:<10.2f}{abs_res:<10.2f}")

if __name__ == "__main__":
    evaluate_all("data/processed_H6", top_n=5)
