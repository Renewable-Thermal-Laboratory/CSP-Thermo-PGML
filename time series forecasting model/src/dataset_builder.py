import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

class TempSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10, scaler_path="models/scaler_nt.save"):
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        self.sequence_length = sequence_length
        self.samples = []
        self.test_files = []  # Store test files separately

        thermal_data_all = []
        param_data_all = []
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))

        # Split files into train (80%) and test (20%)
        np.random.shuffle(all_files)
        split_idx = int(0.8 * len(all_files))
        train_files = all_files[:split_idx]
        self.test_files = all_files[split_idx:]  # Store for later test set

        # Read and collect training data only
        for file_path in train_files:
            df = pd.read_csv(file_path)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna().reset_index(drop=True)
            if df.empty:
                continue
            thermal_cols = [col for col in df.columns if col.startswith("TC")]
            thermal_data_all.append(df[thermal_cols])
            param_data_all.append(df[["h", "flux", "abs", "surf"]].iloc[0])

        # Thermal data scaling (with feature names)
        thermal_cols = ["TC1_tip", "TC2", "TC3", "TC4", "TC5", 
                "TC6", "TC7", "TC8", "TC9", "TC10"]
        full_thermal = pd.DataFrame(
            pd.concat(thermal_data_all, ignore_index=True),
            columns=thermal_cols
        )
        self.thermal_scaler = MinMaxScaler()
        self.thermal_scaler.fit(full_thermal)
        joblib.dump(self.thermal_scaler, scaler_path)

        # Parameter scaling (with feature names)
        param_cols = ["h", "flux", "abs", "surf"]
        full_params = pd.DataFrame(param_data_all, columns=param_cols)
        self.param_scaler = MinMaxScaler()
        self.param_scaler.fit(full_params)
        joblib.dump(self.param_scaler, scaler_path.replace(".save", "_params_nt.save"))

        # Build training samples
        for thermal_df, param_row in zip(thermal_data_all, param_data_all):
            scaled_temp = self.thermal_scaler.transform(thermal_df)
            scaled_param = self.param_scaler.transform([param_row])[0].tolist()

            for i in range(len(scaled_temp) - sequence_length):
                X_seq = scaled_temp[i:i + sequence_length]
                y_target = scaled_temp[i + sequence_length]
                X_seq_with_params = [list(row) + scaled_param for row in X_seq]
                self.samples.append((X_seq_with_params, y_target))

        print(f"Train dataset ready: {len(self.samples)} samples")
        print(f"Test files held out: {len(self.test_files)} files")

    def get_test_dataset(self):
        """Returns a list of (X_seq, y) tuples for test data"""
        test_samples = []
        for file_path in self.test_files:
            df = pd.read_csv(file_path)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            thermal_cols = [col for col in df.columns if col.startswith("TC")]
            
            # Use the same scalers from training
            scaled_temp = self.thermal_scaler.transform(df[thermal_cols])
            scaled_param = self.param_scaler.transform(df[["h", "flux", "abs", "surf"]].iloc[0].values.reshape(1, -1))[0]

            for i in range(len(scaled_temp) - self.sequence_length):
                X_seq = scaled_temp[i:i + self.sequence_length]
                y_target = scaled_temp[i + self.sequence_length]
                X_seq_with_params = [list(row) + scaled_param.tolist() for row in X_seq]
                test_samples.append((X_seq_with_params, y_target))
        
        print(f"Test dataset ready: {len(test_samples)} samples")
        return test_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_seq, y = self.samples[idx]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)