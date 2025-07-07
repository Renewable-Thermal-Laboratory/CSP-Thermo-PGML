import os
import re
import pandas as pd

# Maps and pattern as you provided
h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
flux_map = {88: 25900, 78: 21250, 73: 19400}
abs_map = {0: 3, 92: 100}
surf_map = {0: 0.98, 1: 0.76}
pattern = r"h(\d+)_flux(\d+)_abs(\d+)(?:_[A-Za-z0-9]+)*_surf([01])(?:_[A-Za-z0-9]+)*[\s_]+(\d+)s\b"

cleaned_input_dir = r"D:\Research Assistant work\Github Organization\ml models\ml_models\PhysicsGuidedNeuralNetwork\data/all_processed"
time_reset_output_dir = r"D:\Research Assistant work\Github Organization\ml models\ml_models\PhysicsGuidedNeuralNetwork\data/new_processed"

os.makedirs(time_reset_output_dir, exist_ok=True)

def extract_params_from_filename(filename):
    match = re.search(pattern, filename)
    if not match:
        print(f"Filename pattern did not match: {filename}")
        return None, None, None, None
    
    h_raw = int(match.group(1))
    flux_raw = int(match.group(2))
    abs_raw = int(match.group(3))
    surf_raw = int(match.group(4))
    
    h = h_map.get(h_raw, h_raw)
    flux = flux_map.get(flux_raw, flux_raw)
    abs_ = abs_map.get(abs_raw, abs_raw)
    surf = surf_map.get(surf_raw, None)
    
    if surf is None:
        print(f"Surface value missing in mapping for surf code: {surf_raw} in file: {filename}")
    
    return h, flux, abs_, surf

for fname in os.listdir(cleaned_input_dir):
    if not fname.endswith(".csv"):
        continue
    
    file_path = os.path.join(cleaned_input_dir, fname)
    df = pd.read_csv(file_path)
    
    h, flux, abs_, surf = extract_params_from_filename(fname)
    if None in (h, flux, abs_, surf):
        print(f"Skipping file due to missing parameters: {fname}")
        continue
    
    # Add new columns to the dataframe
    df["h"] = h
    df["flux"] = flux
    df["abs"] = abs_
    df["surf"] = surf
    
    # Save the new file to output directory
    output_path = os.path.join(time_reset_output_dir, fname)
    df.to_csv(output_path, index=False)
    print(f"Processed and saved: {fname}")

print("All files processed.")
