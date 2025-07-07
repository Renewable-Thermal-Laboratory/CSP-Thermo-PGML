import pandas as pd
import os
import glob

input_dir = "D:\Research Assistant work\Github Organization\data\Theoretical_VTDP"
output_dir = "D:/Research Assistant work/Github Organization/ml models/ml_models/PhysicsGuidedNeuralNetwork/data/all_processed"

os.makedirs(output_dir, exist_ok=True)

file_paths = glob.glob(os.path.join(input_dir, "*.csv"))

print(f"Found {len(file_paths)} CSV files in {input_dir}")

for file_path in file_paths:
    print(f"Processing: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Columns before dropping: {df.columns.tolist()}")
    depth_cols = [col for col in df.columns if col.startswith("Depth_")]
    df = df.drop(columns=depth_cols, errors='ignore')
    print(f"Columns after dropping Depth_: {df.columns.tolist()}")

    # Just to confirm filtering works, extract start time from filename
    import re
    filename = os.path.basename(file_path)
    match = re.search(r'(\d+)s', filename)
    if match:
        start_time = int(match.group(1))
        df = df[df["Time"] >= start_time]
        print(f"Filtered rows with Time >= {start_time}: {len(df)} rows remain")
    else:
        print("No starting time found in filename")

    # Save one file to confirm it works
    save_path = os.path.join(output_dir, "cleaned_" + filename)
    df.to_csv(save_path, index=False)
    print(f"Saved cleaned file to {save_path}")