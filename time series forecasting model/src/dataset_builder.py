import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob

class TempSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []

        for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
            df = pd.read_csv(file_path)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna().reset_index(drop=True)

            if df.empty:
                print(f"Skipped {os.path.basename(file_path)}: empty after cleaning")
                continue

            # Extract columns
            feature_cols = df.columns.drop("Time")  # everything except Time
            temp_cols = feature_cols.drop(["h", "flux", "abs", "surf"])

            temp_data = df[temp_cols].values
            params = df[["h", "flux", "abs", "surf"]].iloc[0].values.tolist()

            # Build sequences
            for i in range(len(temp_data) - sequence_length):
                X_seq = temp_data[i:i + sequence_length]
                y_target = temp_data[i + sequence_length]
                # Append static params to each step in the sequence
                X_seq_with_params = [list(row) + params for row in X_seq]
                self.samples.append((X_seq_with_params, y_target))

        print(f"Dataset ready: {len(self.samples)} total samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_seq, y = self.samples[idx]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)