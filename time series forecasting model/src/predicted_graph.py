import os
import glob
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import joblib
from new_model import ImprovedTempModel

# Expected column names
THERMAL_COLS = ["TC1_tip", "TC2", "TC3", "TC4", "TC5",
                "TC6", "TC7", "TC8", "TC9", "TC10"]
PARAM_COLS = ["h", "flux", "abs", "surf", "Time"]
INPUT_FEATURES = len(THERMAL_COLS) + len(PARAM_COLS)  # Should be 15

def extract_params_from_filename(filename):
    """Extract parameters using regex pattern matching."""
    match = re.search(r"h(\d+).*?flux(\d+).*?abs(\d+).*?surf.*?(\d+).*?(\d+)s", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
    flux_map = {88: 25900, 78: 21250, 73: 19400}
    abs_map = {0: 3, 92: 100}
    surf_map = {0: 0.98, 1: 0.76}
    h = h_map.get(int(match.group(1)), int(match.group(1)))
    flux = flux_map.get(int(match.group(2)), int(match.group(2)))
    abs_val = abs_map.get(int(match.group(3)), int(match.group(3)))
    surf = surf_map.get(int(match.group(4)), int(match.group(4)))
    start_time = int(match.group(5))
    return {"h": h, "flux": flux, "abs": abs_val, "surf": surf, "Time": start_time}

def load_model_and_scalers(model_path="models/best_model.pth", scaler_dir="models"):
    thermal_scaler = joblib.load(os.path.join(scaler_dir, "thermal_scaler.save"))
    param_scaler = joblib.load(os.path.join(scaler_dir, "param_scaler.save"))
    model = ImprovedTempModel(input_size=INPUT_FEATURES, output_size=len(THERMAL_COLS))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, thermal_scaler, param_scaler

def process_file(filepath, model, thermal_scaler, param_scaler, sequence_length=20):
    df = pd.read_csv(filepath)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    if len(df) < sequence_length + 1:
        raise ValueError(f"Not enough data in {filepath}")

    try:
        params = extract_params_from_filename(os.path.basename(filepath))
    except Exception as e:
        raise ValueError(f"Failed to extract parameters from {filepath}: {e}")

    input_data = df[THERMAL_COLS].iloc[-sequence_length-1:-1]
    time_steps = np.linspace(0, params["Time"], sequence_length)
    param_matrix = np.array([[params["h"], params["flux"], params["abs"], params["surf"], t] for t in time_steps])
    
    thermal_scaled = thermal_scaler.transform(input_data)
    param_scaled = param_scaler.transform(param_matrix)
    
    X = np.hstack([thermal_scaled, param_scaled])
    if X.shape != (sequence_length, INPUT_FEATURES):
        raise ValueError(f"Input shape mismatch: expected ({sequence_length}, {INPUT_FEATURES}), got {X.shape}")
    
    X_tensor = torch.FloatTensor(X).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()[0]

    pred = thermal_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
    actual = df[THERMAL_COLS].iloc[-1].values
    residuals = np.abs(pred - actual)

    return {
        "filename": os.path.basename(filepath),
        "params": params,
        "predicted": pred,
        "actual": actual,
        "residuals": residuals,
        "avg_residual": np.mean(residuals),
        "thermal_cols": THERMAL_COLS
    }

def print_numerical_results(result):
    """Print the predicted, actual, and residual values for each thermocouple."""
    print(f"\nNumerical Results for {result['filename']}:")
    print("-" * 70)
    print(f"{'TC':<6} {'Predicted (°C)':<15} {'Actual (°C)':<15} {'Residual (°C)':<15}")
    print("-" * 70)
    
    for i, tc_name in enumerate(result['thermal_cols']):
        tc_num = tc_name.replace('_tip', '')  # Clean TC name for display
        predicted = result['predicted'][i]
        actual = result['actual'][i]
        residual = result['residuals'][i]
        print(f"{tc_num:<6} {predicted:<15.3f} {actual:<15.3f} {residual:<15.3f}")
    
    print("-" * 70)
    print(f"{'AVG':<6} {'':<15} {'':<15} {result['avg_residual']:<15.3f}")
    print("-" * 70)

def plot_depth_profile(result, save_path=None):
    # TC10 at top (0), TC1 at bottom (-0.016), equidistant spacing
    tc_numbers = list(range(10, 0, -1))  # TC10 to TC1 (top to bottom)
    depths = np.linspace(0, -0.016, len(tc_numbers))  # 0 to -0.016 m
    
    # Reorder data to match TC10 to TC1 order (reverse the original order)
    actual_ordered = result["actual"][::-1]  # Reverse to get TC10 first
    predicted_ordered = result["predicted"][::-1]  # Reverse to get TC10 first

    plt.figure(figsize=(7, 6))  
    plt.plot(actual_ordered, depths, 'o-', label='Actual', color='blue', markersize=6)
    plt.plot(predicted_ordered, depths, 'x--', label='Predicted', color='red', markersize=8)
    
    # Add TC labels
    for i, (act, pred, tc) in enumerate(zip(actual_ordered, predicted_ordered, tc_numbers)):
        plt.text(act, depths[i], f'TC{tc}', ha='right', va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue', alpha=0.7))
        plt.text(pred, depths[i], f'TC{tc}', ha='left', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='lightcoral', alpha=0.7))
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.title(f'Vertical Profile: {result["filename"]}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    model, thermal_scaler, param_scaler = load_model_and_scalers()
    sequence_length = 20
    data_dir = "data/processed_theoretical_H6"
    output_dir = "results/theoretical_profiles_graph (no training)" \
    ""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for filepath in glob.glob(os.path.join(data_dir, "*.csv")):
        try:
            result = process_file(filepath, model, thermal_scaler, param_scaler, sequence_length)
            results.append(result)
        except Exception as e:
            print(f"{filepath}: {e}")

    results.sort(key=lambda x: x["avg_residual"])

    print("\nTop 5 Best Predictions:")
    print("Filename".ljust(45), "Avg Residual (°C)")
    for r in results:
        print(f"{r['filename']:<45} {r['avg_residual']:.3f}")

    for i, result in enumerate(results):
        # Print numerical results
        print_numerical_results(result)
        
        # Generate and save plot
        save_path = os.path.join(output_dir, f"{result['filename'].replace('.csv', '')}.png")
        plot_depth_profile(result, save_path)
        print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    main()