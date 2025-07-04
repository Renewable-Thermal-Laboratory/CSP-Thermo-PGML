import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np
from scipy.stats import zscore
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TempSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10, scaler_dir="models"):
        self.sequence_length = sequence_length
        self.scaler_dir = scaler_dir
        os.makedirs(self.scaler_dir, exist_ok=True)

        # Initialize scalers
        self.thermal_scaler = RobustScaler()
        self.param_scaler = RobustScaler()

        # Load and process data
        self._load_data(data_dir)
        self._build_samples()

        print(f"Dataset initialized with {len(self.samples)} sequences")
        print(f"Input shape: [batch, {sequence_length}, {self.samples[0][0].shape[1]}]")
        print(f"Output shape: [{self.samples[0][1].shape[0]}]")

    def _load_data(self, data_dir):
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not all_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        np.random.shuffle(all_files)
        split_1 = int(0.7 * len(all_files))
        split_2 = int(0.85 * len(all_files))

        self.train_files = all_files[:split_1]
        self.val_files = all_files[split_1:split_2]
        self.test_files = all_files[split_2:]

        train_dfs = []
        for file in self.train_files:
            df = self._process_file(file)
            train_dfs.append(df)

        thermal_cols = [col for col in train_dfs[0].columns if col.startswith("TC")]
        param_cols = ["h", "flux", "abs", "surf", "Time"]

        thermal_data = pd.concat([df[thermal_cols] for df in train_dfs])
        param_data = pd.concat([df[param_cols] for df in train_dfs])

        self.thermal_scaler.fit(thermal_data)
        self.param_scaler.fit(param_data)

        joblib.dump(self.thermal_scaler, os.path.join(self.scaler_dir, "thermal_scaler.save"))
        joblib.dump(self.param_scaler, os.path.join(self.scaler_dir, "param_scaler.save"))

    def _process_file(self, file_path):
        df = pd.read_csv(file_path)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        thermal_cols = [col for col in df.columns if col.startswith("TC")]
        if thermal_cols:
            z_scores = np.abs(zscore(df[thermal_cols]))
            df = df[(z_scores < 3).all(axis=1)]

        if "Time" in df.columns:
            df["Time"] = df["Time"] - df["Time"].iloc[0]

        return df

    def _build_samples(self):
        self.samples = []
        self.val_samples = []
        self.test_samples = []

        for file in self.train_files:
            df = self._process_file(file)
            self._create_sequences(df, self.samples)

        for file in self.val_files:
            df = self._process_file(file)
            self._create_sequences(df, self.val_samples)

        for file in self.test_files:
            df = self._process_file(file)
            self._create_sequences(df, self.test_samples)

    def _create_sequences(self, df, sample_list):
        thermal_cols = [col for col in df.columns if col.startswith("TC")]
        param_cols = ["h", "flux", "abs", "surf", "Time"]

        if not thermal_cols:
            return

        thermal_data = self.thermal_scaler.transform(df[thermal_cols])
        params = self.param_scaler.transform(df[param_cols].iloc[0:1].values)[0]

        for i in range(len(thermal_data) - self.sequence_length):
            sequence = thermal_data[i:i + self.sequence_length]
            target = thermal_data[i + self.sequence_length]

            seq_with_params = np.hstack([
                sequence,
                np.tile(params[:-1], (sequence.shape[0], 1)),
                np.linspace(0, params[-1], sequence.shape[0])[:, None]
            ])

            sample_list.append((seq_with_params, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.FloatTensor(seq), torch.FloatTensor(target)

    def get_test_data(self):
        X = torch.stack([torch.FloatTensor(x[0]) for x in self.test_samples])
        y = torch.stack([torch.FloatTensor(x[1]) for x in self.test_samples])
        return X, y

    def get_val_data(self):
        X = torch.stack([torch.FloatTensor(x[0]) for x in self.val_samples])
        y = torch.stack([torch.FloatTensor(x[1]) for x in self.val_samples])
        return X, y
