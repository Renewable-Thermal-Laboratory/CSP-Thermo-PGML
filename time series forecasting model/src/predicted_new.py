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
    model = ImprovedTempModel(input_size=14, output_size=10)  # 10 thermal + 4 physical params
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()
    
    scaler = joblib.load("models/thermal_scaler.save")
    param_scaler = joblib.load("models/param_scaler.save")
    return model, scaler, param_scaler

def predict_from_csv_with_residuals(csv_path, model, scaler, param_scaler):
    filename = os.path.basename(csv_path)
    
    try:
        static_params = extract_params_from_filename(filename)
    except ValueError:
        return None  # Skip files that don't match pattern
    
    try:
        # Load and clean thermal data
        df = pd.read_csv(csv_path)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=THERMAL_COLS).reset_index(drop=True)
        
        if len(df) < 11:  # Need at least 11 rows: 10 for sequence + 1 for actual comparison
            return None
        
        # Use bottom 10 from second-to-last position (rows -11 to -2)
        sequence = df[THERMAL_COLS].iloc[-11:-1]
        
        # Last row is the actual values for comparison
        actual_values = df[THERMAL_COLS].iloc[-1].values
        
        # Scale thermal and parameter data
        sequence_scaled = scaler.transform(sequence)
        static_scaled = param_scaler.transform([static_params])[0]
        
        # Combine each timestep with the static parameters
        X = [list(row) + static_scaled.tolist() for row in sequence_scaled]
        X_tensor = torch.tensor([X], dtype=torch.float32)
        
        with torch.no_grad():
            prediction_scaled = model(X_tensor).numpy()
        
        # Inverse scale back to original °C units
        predicted_values = scaler.inverse_transform(prediction_scaled)[0]
        
        # Calculate residuals (actual - predicted)
        residuals = actual_values - predicted_values
        
        # Calculate performance metrics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        max_residual = np.max(np.abs(residuals))  # Maximum absolute residual
        
        return {
            'filename': filename,
            'predicted_values': predicted_values,
            'actual_values': actual_values,
            'residuals': residuals,
            'mae': mae,
            'rmse': rmse,
            'max_residual': max_residual,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual_raw': np.max(residuals)
        }
    
    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")
        return None

def process_all_files_for_min_max_residual(data_dir="data/processed_H6", top_n=5):
    """Process all files and show only the ones with the smallest maximum residuals"""
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files. Processing...")
    
    # Load model and scalers once
    model, scaler, param_scaler = load_model_and_scalers()
    
    results = []
    
    # Process each file
    for csv_path in csv_files:
        result = predict_from_csv_with_residuals(csv_path, model, scaler, param_scaler)
        if result is not None:
            results.append(result)
    
    if not results:
        print("No valid results obtained from any files.")
        return
    
    print(f"Successfully processed {len(results)} files.")
    
    # Sort by maximum absolute residual (ascending - smallest max residual first)
    results_sorted = sorted(results, key=lambda x: x['max_residual'])
    
    # Show summary statistics
    max_residuals = [r['max_residual'] for r in results]
    print("\n" + "="*80)
    print("📊 MAXIMUM RESIDUAL STATISTICS")
    print("="*80)
    print(f"Total files processed: {len(results)}")
    print(f"Max Residual range: {min(max_residuals):.3f} - {max(max_residuals):.3f} °C")
    print(f"Mean Max Residual: {np.mean(max_residuals):.3f} °C")
    print(f"Median Max Residual: {np.median(max_residuals):.3f} °C")
    
    # Show top performers (smallest max residuals)
    print(f"\n" + "="*80)
    print(f"🏆 TOP {top_n} FILES WITH SMALLEST MAXIMUM RESIDUALS")
    print("="*80)
    
    for rank, result in enumerate(results_sorted[:top_n], 1):
        print(f"\n🥇 Rank {rank}: {result['filename']}")
        print(f"   Max Residual: {result['max_residual']:.3f} °C")
        print(f"   MAE: {result['mae']:.3f} °C | RMSE: {result['rmse']:.3f} °C")
        print(f"   Mean Residual: {result['mean_residual']:.3f} °C | Std Residual: {result['std_residual']:.3f} °C")
        print(f"   Residual Range: [{result['min_residual']:.3f}, {result['max_residual_raw']:.3f}] °C")
        
        print(f"\n   {'Sensor':<10} {'Predicted':<12} {'Actual':<12} {'Residual':<12} {'Abs Residual':<12}")
        print("   " + "-"*65)
        for i, sensor in enumerate(THERMAL_COLS):
            abs_res = abs(result['residuals'][i])
            marker = " ⚠️" if abs_res == result['max_residual'] else ""
            print(f"   {sensor:<10} {result['predicted_values'][i]:<12.2f} {result['actual_values'][i]:<12.2f} {result['residuals'][i]:<12.2f} {abs_res:<12.2f}{marker}")
    
    print("\n" + "="*80)
    print("⚠️ = Sensor with maximum absolute residual")

# Example usage
if __name__ == "__main__":
    # You can change top_n to see more or fewer results
    process_all_files_for_min_max_residual("data/processed", top_n=5)