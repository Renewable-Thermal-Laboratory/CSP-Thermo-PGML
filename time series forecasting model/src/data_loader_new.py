import os
import re
import pandas as pd

RAW_DIR = "data/New_theoretical_data"
PROCESSED_DIR = "data/processed_New_theoretical_data"

# Regex pattern: h0.4_flux5000_abs15_surf90_600s
pattern = r"h([0-9.]+)_flux([0-9.]+)_abs([0-9.]+)_surf([0-9.]+)_([0-9]+)s"

def extract_params_from_filename(filename):
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")
    h = float(match.group(1))
    flux = float(match.group(2))
    abs_val = float(match.group(3))
    surf = float(match.group(4)) / 100  # convert surf to 0.x
    return h, flux, abs_val, surf

def preprocess_and_save(file_path):
    filename = os.path.basename(file_path)
    print(f"Processing: {filename}")
    
    h, flux, abs_val, surf = extract_params_from_filename(filename)
    
    df = pd.read_csv(file_path)

    # Add parameter columns
    df["h"] = h
    df["flux"] = flux
    df["abs"] = abs_val
    df["surf"] = surf

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

def batch_preprocess_all():
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".csv"):
            try:
                preprocess_and_save(os.path.join(RAW_DIR, fname))
            except Exception as e:
                print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    batch_preprocess_all()
