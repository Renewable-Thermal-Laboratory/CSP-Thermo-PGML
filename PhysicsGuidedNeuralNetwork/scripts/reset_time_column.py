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