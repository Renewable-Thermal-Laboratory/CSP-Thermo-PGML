import pandas as pd
import os
import re

# Directories
INPUT_DIR = "data/raw_theoretical_H6"
OUTPUT_DIR = "data/processed_theoretical_H6"

# Mapping from raw codes to real values
h_map = {2: 0.0375, 3: 0.084, 6: 0.1575}
flux_map = {88: 25900, 78: 21250, 73: 19400}
abs_val_map = {0: 3, 92: 100}
surf_map = {0: 0.98, 1: 0.76}

# Regex pattern to extract params
pattern = r"h(\d+).*?flux(\d+).*?abs(\d+).*?surf.*?(\d+).*?([0-9]+)s"

# Mapping from Theoretical_Temps_X → TC names
rename_map = {
    "Theoretical_Temps_11": "TC1_tip",
    "Theoretical_Temps_10": "TC2",
    "Theoretical_Temps_9":  "TC3",
    "Theoretical_Temps_7":  "TC4",
    "Theoretical_Temps_6":  "TC5",
    "Theoretical_Temps_5":  "TC6",
    "Theoretical_Temps_4":  "TC7",
    "Theoretical_Temps_3":  "TC8",
    "Theoretical_Temps_2":  "TC9",
    "Theoretical_Temps_1":  "TC10",
}

# Desired final order
final_columns = ["Time"] + list(rename_map.values()) + ["h", "flux", "abs", "surf"]

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process all CSVs
for file in os.listdir(INPUT_DIR):
    if file.lower().endswith(".csv"):
        input_path = os.path.join(INPUT_DIR, file)
        print(f"Processing: {file}")
        try:
            # Extract parameters from filename
            match = re.search(pattern, file)
            if not match:
                print(f"Skipping {file} → filename pattern not matched.")
                continue

            h_raw = int(match.group(1))
            flux_raw = int(match.group(2))
            abs_raw = int(match.group(3))
            surf_raw = int(match.group(4))
            start_time = int(match.group(5))

            h = h_map.get(h_raw, h_raw)
            flux = flux_map.get(flux_raw, flux_raw)
            abs_val = abs_val_map.get(abs_raw, abs_raw)
            surf = surf_map.get(surf_raw, surf_raw)

            # Load CSV
            df = pd.read_csv(input_path)

            # -------------------------
            # STEP 1 - Keep only theoretical columns
            # -------------------------

            theoretical_cols = list(rename_map.keys())
            # Drop Theoretical_Temps_8
            if "Theoretical_Temps_8" in df.columns:
                df = df.drop(columns=["Theoretical_Temps_8"])

            # Keep only Time + theoretical cols
            cols_to_keep = ["Time"] + theoretical_cols
            cols_in_file = [c for c in cols_to_keep if c in df.columns]

            df = df[cols_in_file]

            # -------------------------
            # STEP 2 - Add static params
            # -------------------------
            df["h"] = h
            df["flux"] = flux
            df["abs"] = abs_val
            df["surf"] = surf

            # -------------------------
            # STEP 3 - Filter rows and reset Time
            # -------------------------
            df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
            df = df.dropna(subset=["Time"])

            df = df[df["Time"] >= start_time].reset_index(drop=True)

            if len(df) == 0:
                print(f"Skipping {file} → no data after start time.")
                continue

            # Reset time to start at 0
            df["Time"] = df["Time"] - df["Time"].iloc[0]

            # -------------------------
            # STEP 4 - Rename columns
            # -------------------------
            df = df.rename(columns=rename_map)

            # -------------------------
            # STEP 5 - Reorder columns
            # -------------------------
            # Ensure all expected columns exist
            missing_cols = [col for col in final_columns if col not in df.columns]
            if missing_cols:
                print(f"Skipping {file} → missing columns: {missing_cols}")
                continue

            df_final = df[final_columns]

            # Save
            output_path = os.path.join(OUTPUT_DIR, file)
            df_final.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing {file}: {e}")
