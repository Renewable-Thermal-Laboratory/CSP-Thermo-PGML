import torch
import joblib
import numpy as np
import pandas as pd
import re
import os
from model import TempLSTM

# Thermal and parameter feature definitions
THERMAL_COLS = ["TC1_tip", "TC2", "TC3", "TC4", "TC5", 
                "TC6", "TC7", "TC8", "TC9", "TC10"]
PARAM_PATTERN = r"h(\d+).*?flux(\d+).*?abs(\d+).*?surf.*?(\d+)"

# Mappings from your preprocessing
h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
flux_map = {88: 25900, 78: 21250, 73: 19400}
abs_map = {0: 3, 92: 100}
surf_map = {0: 0.98, 1: 0.76}

def extract_params_from_filename(filename):
    match = re.search(PARAM_PATTERN, filename)
    if not match:
        raise ValueError(f"Filename doesn't contain expected pattern: {filename}")
    h = h_map.get(int(match.group(1)), int(match.group(1)))
    flux = flux_map.get(int(match.group(2)), int(match.group(2)))
    abs_val = abs_map.get(int(match.group(3)), int(match.group(3)))
    surf = surf_map.get(int(match.group(4)), int(match.group(4)))
    return [h, flux, abs_val, surf]

def load_model_and_scalers():
    model = TempLSTM(input_size=14, output_size=10)  # 10 thermal + 4 physical params
    model.load_state_dict(torch.load("models/temp_lstm_final.pt"))
    model.eval()
    
    scaler = joblib.load("models/scaler.save")
    param_scaler = joblib.load("models/scaler_params.save")
    return model, scaler, param_scaler

def predict_from_csv(csv_path):
    filename = os.path.basename(csv_path)
    static_params = extract_params_from_filename(filename)

    # Load and clean thermal data
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=THERMAL_COLS).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Not enough data: at least 10 time steps are required.")
    sequence = df[THERMAL_COLS].iloc[-10:]

    # Load model and scalers
    model, scaler, param_scaler = load_model_and_scalers()

    # Scale thermal and parameter data
    sequence_scaled = scaler.transform(sequence)
    static_scaled = param_scaler.transform([static_params])[0]

    # Combine each timestep with the static parameters
    X = [list(row) + static_scaled.tolist() for row in sequence_scaled]
    X_tensor = torch.tensor([X], dtype=torch.float32)

    with torch.no_grad():
        prediction_scaled = model(X_tensor).numpy()

    # Inverse scale back to original °C units
    prediction = scaler.inverse_transform(prediction_scaled)[0]
    return prediction

# Example usage
if __name__ == "__main__":
    csv_path = "data/processed/h2_flux88_abs0_surf0_431s - Sheet1.csv"  # Change to your file path
    predicted_temps = predict_from_csv(csv_path)
    print("🌡️ Predicted temperatures at t+1:")
    for sensor, value in zip(THERMAL_COLS, predicted_temps):
        print(f"{sensor}: {value:.2f} °C")
