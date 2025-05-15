import pandas as pd
import os
import re

RAW_DIR = "data/Raw_normal"
PROCESSED_DIR = "data/processed"

# Optional mappings — used if keys exist
h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
flux_map = {88: 25900, 78: 21250, 73: 19400}
abs_val_map = {0: 3, 92: 100}
surf_map = {0: 0.98, 1: 0.76}

# Flexible pattern to handle various suffixes and spacing
pattern = r"h(\d+).*?flux(\d+).*?abs(\d+).*?surf.*?(\d+).*?([0-9]+)s"

def extract_params_from_filename(filename):
    match = re.search(pattern, filename)

    h_raw = int(match.group(1))
    flux_raw = int(match.group(2))
    abs_raw = int(match.group(3))
    surf_raw = int(match.group(4))
    start_time = int(match.group(5))

    # Use mapping if available, else raw
    h = h_map.get(h_raw, h_raw)
    flux = flux_map.get(flux_raw, flux_raw)
    abs_val = abs_val_map.get(abs_raw, abs_raw)
    surf = surf_map.get(surf_raw, surf_raw)

    return h, flux, abs_val, surf, start_time

def preprocess_and_save_with_params(file_path, save=True):
    filename = os.path.basename(file_path)
    print(f"Processing {filename}")

    params = extract_params_from_filename(filename)
    h, flux, abs_val, surf, start_time = params

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig", header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1", header=0)

    # Handle Time column
    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
        df = df[df["Time"] >= start_time].reset_index(drop=True)
    else:
        print(f"No 'Time' column in {filename}. Skipping time trim.")

    if "TC_9.5" in df.columns:
        df.drop(columns=["TC_9.5"], inplace=True)
        print(f"Dropped TC_9.5 from {filename}")

    # Add parameter columns (even if None)
    df["h"] = h
    df["flux"] = flux
    df["abs"] = abs_val
    df["surf"] = surf

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(os.path.join(PROCESSED_DIR, filename), index=False)
        print(f"Saved: {filename}")

    return df

def batch_preprocess_all():
    print(f"Processing all CSVs in: {RAW_DIR}")
    for file in os.listdir(RAW_DIR):
        if file.lower().endswith(".csv"):
            try:
                preprocess_and_save_with_params(os.path.join(RAW_DIR, file))
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    batch_preprocess_all()
