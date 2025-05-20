import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
from sklearn.preprocessing import MinMaxScaler
import joblib

class TempSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10, scaler_path="models/scaler.save"):
        self.sequence_length = sequence_length
        self.samples = []
        self.scaler_path = scaler_path

        all_data = []
        for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
            df = pd.read_csv(file_path)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna().reset_index(drop=True)
            if df.empty:
                continue
            all_data.append(df)

        full_df = pd.concat(all_data, ignore_index=True)
        temp_cols = [col for col in full_df.columns if col.startswith("TC")]
        self.scaler = MinMaxScaler()
        self.scaler.fit(full_df[temp_cols])
        joblib.dump(self.scaler, self.scaler_path)

        for df in all_data:
            df[temp_cols] = self.scaler.transform(df[temp_cols])
            temp_data = df[temp_cols].values
            params = df[["h", "flux", "abs", "surf"]].iloc[0].values.tolist()

            for i in range(len(temp_data) - sequence_length):
                X_seq = temp_data[i:i + sequence_length]
                y_target = temp_data[i + sequence_length]
                X_seq_with_params = [list(row) + params for row in X_seq]
                self.samples.append((X_seq_with_params, y_target))

        print(f"Dataset ready: {len(self.samples)} total samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_seq, y = self.samples[idx]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
