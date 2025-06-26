# import pandas as pd
# import os
# import glob

# input_dir = "D:\Research Assistant work\Github Organization\data\Theoretical_VTDP"
# output_dir = "D:/Research Assistant work/Github Organization/ml models/ml_models/PhysicsGuidedNeuralNetwork/data/all_processed"

# os.makedirs(output_dir, exist_ok=True)

# file_paths = glob.glob(os.path.join(input_dir, "*.csv"))

# print(f"Found {len(file_paths)} CSV files in {input_dir}")

# for file_path in file_paths:
#     print(f"Processing: {file_path}")
#     df = pd.read_csv(file_path)
#     print(f"Columns before dropping: {df.columns.tolist()}")
#     depth_cols = [col for col in df.columns if col.startswith("Depth_")]
#     df = df.drop(columns=depth_cols, errors='ignore')
#     print(f"Columns after dropping Depth_: {df.columns.tolist()}")

#     # Just to confirm filtering works, extract start time from filename
#     import re
#     filename = os.path.basename(file_path)
#     match = re.search(r'(\d+)s', filename)
#     if match:
#         start_time = int(match.group(1))
#         df = df[df["Time"] >= start_time]
#         print(f"Filtered rows with Time >= {start_time}: {len(df)} rows remain")
#     else:
#         print("No starting time found in filename")

#     # Save one file to confirm it works
#     save_path = os.path.join(output_dir, "cleaned_" + filename)
#     df.to_csv(save_path, index=False)
#     print(f"Saved cleaned file to {save_path}")

import pandas as pd
import os
import glob

def reset_time_to_zero(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)

            if "Time" not in df.columns:
                print(f"Skipping {filename}: No 'Time' column found.")
                continue

            # Subtract min Time to reset start time to 0
            min_time = df["Time"].min()
            df["Time"] = df["Time"] - min_time

            # Save file to output directory
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False)
            print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# === Usage ===
cleaned_input_dir = "D:\Research Assistant work\Github Organization\ml models\ml_models\PhysicsGuidedNeuralNetwork\data/all_processed"   # Your cleaned CSV files folder
time_reset_output_dir = "D:\Research Assistant work\Github Organization\ml models\ml_models\PhysicsGuidedNeuralNetwork\data/all_processed_time_reset"

reset_time_to_zero(cleaned_input_dir, time_reset_output_dir)
