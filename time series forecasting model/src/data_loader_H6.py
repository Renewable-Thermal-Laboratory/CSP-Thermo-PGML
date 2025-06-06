import pandas as pd
import os
import re
import numpy as np

RAW_DIR = "data/H6"
PROCESSED_DIR = "data/processed_H6"

h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
flux_map = {88: 25900, 78: 21250, 73: 19400}
abs_val_map = {0: 3, 92: 100}
surf_map = {0: 0.98, 1: 0.76}

pattern = r"h(\d+).*?flux(\d+).*?abs(\d+).*?surf.*?(\d+).*?([0-9]+)s"

ALLOWED_COLS = [
    "Time", "TC1_tip", "TC2", "TC3", "TC4", "TC5",
    "TC6", "TC7", "TC8", "TC9", "TC10"
]

def extract_params_from_filename(filename):
    match = re.search(pattern, filename)
    h_raw = int(match.group(1))
    flux_raw = int(match.group(2))
    abs_raw = int(match.group(3))
    surf_raw = int(match.group(4))
    start_time = int(match.group(5))

    h = h_map.get(h_raw, h_raw)
    flux = flux_map.get(flux_raw, flux_raw)
    abs_val = abs_val_map.get(abs_raw, abs_raw)
    surf = surf_map.get(surf_raw, surf_raw)

    return h, flux, abs_val, surf, start_time

def preprocess_and_save_with_params(file_path, save=True):
    filename = os.path.basename(file_path)
    print(f"Processing {filename}")

    h, flux, abs_val, surf, start_time = extract_params_from_filename(filename)

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig", header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1", header=0)

    # Keep only allowed columns
    df = df[[col for col in ALLOWED_COLS if col in df.columns]]

    # Handle Time column
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df[df["Time"] >= start_time].reset_index(drop=True)
    df["Time"] = df["Time"] - df["Time"].iloc[0]  # Shift time to start at 0

    if "TC6" in df.columns:
        tc6 = df["TC6"].values
        keep_index = len(tc6)  # default: keep everything

        for i in range(len(tc6)):
            future = tc6[i:]
            if np.all(np.diff(future) <= 0):  # strictly decreasing or flat
                keep_index = i
                break

        df = df.iloc[:keep_index].reset_index(drop=True)

    # Drop unwanted
    if "TC_9.5" in df.columns:
        df.drop(columns=["TC_9.5"], inplace=True)

    # Add parameter columns
    df["h"] = h
    df["flux"] = flux
    df["abs"] = abs_val
    df["surf"] = surf

    # Save
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
