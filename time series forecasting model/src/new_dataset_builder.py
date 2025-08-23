import os
import pandas as pd
import torch
import torch.utils.data as data
import glob
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from scipy.stats import zscore
import random
import json
import re
from collections import defaultdict

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TempSequenceDataset(data.Dataset):
    """PyTorch Dataset for thermal system temperature sequences, compatible with power-based loss.

    Args:
        data_dir (str): Directory containing CSV files.
        sequence_length (int): Number of input timesteps (fixed at 20).
        prediction_horizon (int): Steps ahead to predict (1 = next step, -1 = final state of file).
        scaler_dir (str): Directory to save/load scalers.
        split (str): Dataset split - 'train', 'val', or 'test'.

    Returns:
        A PyTorch Dataset yielding tuples:
        - time_series: (sequence_length, 11) with normalized time and TC1-TC10.
        - static_params: (4,) with normalized h, flux, abs, surf.
        - target: (10,) with normalized target temperatures.
        - power_data: Dictionary with unscaled time and temperatures for power calculation.
    """
    def __init__(self, data_dir, sequence_length=20, prediction_horizon=1, 
                 scaler_dir="models_new_theoretical", split='train'):
        # Guard acceptable horizons (A4)
        assert (prediction_horizon >= 1) or (prediction_horizon == -1), \
            "prediction_horizon must be >=1 or -1 (use -1 to mean 'to the end of file')."
        
        # Enforce sequence_length = 20 per requirements    
        assert sequence_length == 20, "sequence_length must be 20 per requirements"
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_dir = scaler_dir
        self.split = split
        os.makedirs(self.scaler_dir, exist_ok=True)

        # Initialize scalers
        self.thermal_scaler = StandardScaler()
        self.param_scaler = StandardScaler()
        
        # Initialize data cache and statistics
        self._file_cache = {}
        self._data_stats = defaultdict(int)

        # Load and process data
        self._load_data(data_dir)
        
        if split == 'train':
            # Only fit scalers on training data
            self._fit_scalers()
            
        self.sample_indices = self._build_sample_indices()

        print(f"Dataset initialized with {len(self.sample_indices)} sequences for {split} split")
        print(f"Input shape: time_series [{self.sequence_length}, 11], static_params [4]")
        print(f"Output shape: [10]")
        print(f"Power data: Dictionary with unscaled time/temps for power calculation")
        print(f"Prediction horizon: {prediction_horizon} ({'final state' if prediction_horizon == -1 else f'{prediction_horizon} steps ahead'})")
        
        if split == 'train':
            self._print_data_statistics()

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """Get a single sample from the dataset with bounded retry on failure."""
        max_retries = 5
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try_idx = (idx + attempt) % len(self.sample_indices)
            file_path, start_idx = self.sample_indices[try_idx]
            sample = self._create_sequence(file_path, start_idx)
            
            if sample is not None:
                time_series, static_params, target, power_data = sample
                
                # Convert to PyTorch tensors
                time_series = torch.from_numpy(time_series).float()
                static_params = torch.from_numpy(static_params).float()
                target = torch.from_numpy(target).float()
                
                return time_series, static_params, target, power_data
        
        # If all retries failed, raise clear error
        raise RuntimeError(
            f"Failed to create valid sample after {max_retries} retries in {self.split} split "
            f"(dataset length: {len(self.sample_indices)}). Check data quality or sequence parameters."
        )

    def _get_thermal_columns(self, df):
        """Extract and validate thermal sensor columns (TC1-TC10)."""
        # Handle both TC1_tip and TC1 formats, extract sensor numbers
        potential_cols = []
        for col in df.columns:
            if 'TC' in col.upper():
                # Extract number from column name (handles TC1_tip, TC2, etc.)
                numbers = re.findall(r'\d+', col)
                if numbers:
                    sensor_num = int(numbers[0])
                    if 1 <= sensor_num <= 10:
                        potential_cols.append((sensor_num, col))
        
        # Sort by sensor number and extract column names
        potential_cols.sort(key=lambda x: x[0])
        thermal_cols = [col for _, col in potential_cols]
        
        if len(thermal_cols) != 10:
            available_cols = [col for col in df.columns if 'TC' in col.upper()]
            raise ValueError(
                f"Expected 10 thermal sensors (TC1-TC10), found {len(thermal_cols)}: {thermal_cols}. "
                f"Available TC columns: {available_cols}"
            )
        
        return thermal_cols

    def _validate_required_columns(self, df, file_path):
        """Validate that all required columns are present."""
        thermal_cols = self._get_thermal_columns(df)
        required_static = ['h', 'flux', 'abs', 'surf']
        required_time = ['Time']
        
        all_required = thermal_cols + required_static + required_time
        missing_cols = [col for col in all_required if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")
        
        return thermal_cols

    def _validate_data_quality(self, df, file_path, thermal_cols):
        """Validate data quality and log statistics."""
        original_length = len(df)
        
        # Check for sufficient data length (A3 - bounds update)
        if self.prediction_horizon == -1:
            min_length = self.sequence_length + 1
        else:
            min_length = self.sequence_length + self.prediction_horizon
            
        if original_length < min_length:
            self._data_stats['files_too_short'] += 1
            print(f"Warning: {file_path} too short ({original_length} rows), minimum required: {min_length}")
            return None
            
        # Check for outliers before removal (moved logic to _process_file for efficiency)
        # This section is now handled in _process_file to avoid duplicate z-score computation
                
        # Check for missing values
        missing_count = df[thermal_cols + ['Time', 'h', 'flux', 'abs', 'surf']].isnull().sum().sum()
        if missing_count > 0:
            self._data_stats['files_with_missing'] += 1
            self._data_stats['total_missing'] += missing_count
            
        return df

    def _load_data(self, data_dir):
        """Load and split data files."""
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not all_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        print(f"Found {len(all_files)} CSV files")
        
        # Shuffle files for random split
        np.random.shuffle(all_files)
        split_1 = int(0.7 * len(all_files))
        split_2 = int(0.85 * len(all_files))

        self.train_files = all_files[:split_1]
        self.val_files = all_files[split_1:split_2]
        self.test_files = all_files[split_2:]
        
        print(f"Split: {len(self.train_files)} train, {len(self.val_files)} val, {len(self.test_files)} test files")

        # Set current split files
        if self.split == 'train':
            self.current_files = self.train_files
        elif self.split == 'val':
            self.current_files = self.val_files
        else:  # test
            self.current_files = self.test_files

    def _fit_scalers(self):
        """Fit scalers on training data with proper validation."""
        train_thermal_data = []
        train_param_data = []
        
        valid_files = 0
        
        for file in self.train_files:
            try:
                df = self._process_file(file)  # No longer needs fit_scalers parameter
                if df is None or len(df) == 0:
                    continue
                    
                thermal_cols = self._get_thermal_columns(df)
                param_cols = ["h", "flux", "abs", "surf"]
                
                # Collect thermal data (all rows)
                train_thermal_data.append(df[thermal_cols])
                
                # Collect static parameters (first row only, as they're constant per file)
                train_param_data.append(df[param_cols].iloc[[0]])
                
                valid_files += 1
                
            except Exception as e:
                print(f"Warning: Skipping file {file} due to error: {e}")
                self._data_stats['invalid_files'] += 1
                continue
        
        if valid_files == 0:
            raise ValueError("No valid training files found for scaler fitting")
            
        print(f"Fitting scalers on {valid_files} valid training files")
        
        # Combine all training data
        all_thermal_data = pd.concat(train_thermal_data, ignore_index=True)
        all_param_data = pd.concat(train_param_data, ignore_index=True)
        
        # Fit scalers
        self.thermal_scaler.fit(all_thermal_data)
        self.param_scaler.fit(all_param_data)
        
        # Save scalers
        joblib.dump(self.thermal_scaler, os.path.join(self.scaler_dir, "thermal_scaler.save"))
        joblib.dump(self.param_scaler, os.path.join(self.scaler_dir, "param_scaler.save"))
        
        print("Scalers fitted and saved successfully")

    def _process_file(self, file_path):
        """Process a single CSV file with comprehensive validation."""
        # Check cache first
        if file_path in self._file_cache:
            return self._file_cache[file_path]
            
        try:
            # Load and basic cleaning
            df = pd.read_csv(file_path)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
            if len(df) == 0:
                print(f"Warning: {file_path} is empty after cleaning")
                return None
            
            # Validate columns
            thermal_cols = self._validate_required_columns(df, file_path)
            
            # Validate time is strictly increasing and compute diagnostic
            time_diffs = np.diff(df["Time"].values)
            if np.any(time_diffs <= 0):
                print(f"Warning: {file_path} has non-increasing time values")
                return None
            
            # Store median time step for diagnostics (in milliseconds)
            self._data_stats["median_dt_ms"] = float(np.median(time_diffs) * 1000)
            
            # Validate data quality
            df = self._validate_data_quality(df, file_path, thermal_cols)
            if df is None:
                return None
            
            # Compute z-scores once for outlier detection
            if len(thermal_cols) > 0:
                z_scores = np.abs(zscore(df[thermal_cols], nan_policy='omit'))
                outlier_mask_before = (z_scores >= 3).any(axis=1)
                outlier_count = outlier_mask_before.sum()
                
                if outlier_count > 0:
                    self._data_stats['total_outliers'] += outlier_count
                    self._data_stats['files_with_outliers'] += 1
                
                # Apply outlier filter (keep rows with z-score < 3 in all columns)
                outlier_mask = (z_scores < 3).all(axis=1)
                df = df[outlier_mask]
                
                if len(df) == 0:
                    print(f"Warning: {file_path} is empty after outlier removal")
                    return None
            
            # Cache the processed dataframe
            self._file_cache[file_path] = df
            self._data_stats['valid_files'] += 1
            
            return df
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            self._data_stats['invalid_files'] += 1
            return None

    def _build_sample_indices(self):
        """Build indices for all samples in current split."""
        sample_indices = []
        total_sequences = 0
        
        for file in self.current_files:
            df = self._process_file(file)
            if df is None:
                continue
                
            indices = self._create_sequence_indices(df)
            file_sequences = len(indices)
            sample_indices.extend([(file, i) for i in indices])
            total_sequences += file_sequences
            
        print(f"Generated {total_sequences} {self.split} sequences from {len(self.current_files)} files")
        return sample_indices

    def _create_sequence_indices(self, df):
        """Create valid sequence starting indices for a dataframe."""
        if df is None or len(df) == 0:
            return []
            
        thermal_cols = self._get_thermal_columns(df)
        if not thermal_cols:
            return []
            
        # Calculate minimum required length and max_start_idx based on horizon (A3)
        if self.prediction_horizon == -1:
            min_length = self.sequence_length + 1
            max_start_idx = len(df) - self.sequence_length - 1
        else:
            min_length = self.sequence_length + self.prediction_horizon
            max_start_idx = len(df) - self.sequence_length - self.prediction_horizon
            
        if len(df) < min_length:
            return []

        return list(range(max(0, max_start_idx + 1)))

    def _create_sequence(self, file_path, start_idx):
        """Create a single training sequence with proper indexing."""
        df = self._process_file(file_path)
        if df is None:
            return None
            
        thermal_cols = self._get_thermal_columns(df)
        param_cols = ["h", "flux", "abs", "surf"]
        time_col = "Time"

        # Validate sequence bounds
        if self.prediction_horizon == -1:
            min_required = self.sequence_length + 1
        else:
            min_required = self.sequence_length + self.prediction_horizon
            
        if len(df) < min_required or start_idx + min_required > len(df):
            return None

        try:
            # Get raw data
            thermal_data = df[thermal_cols].values
            time_data = df[time_col].values.reshape(-1, 1)
            static_params = df[param_cols].iloc[0].values

            # Apply scaling
            thermal_data_scaled = self.thermal_scaler.transform(thermal_data)
            static_params_scaled = self.param_scaler.transform([static_params])[0]

            # Normalize time (match model: mean=300, std=300)
            time_scaled = (time_data - 300.0) / 300.0

            # Create input sequence
            sequence_thermal = thermal_data_scaled[start_idx:start_idx + self.sequence_length]
            sequence_time = time_scaled[start_idx:start_idx + self.sequence_length]
            time_series = np.hstack([sequence_time, sequence_thermal])

            # Compute horizon-aware target index (A1)
            seq_end_idx = start_idx + self.sequence_length - 1
            if self.prediction_horizon == -1:
                target_idx = len(df) - 1
            else:
                target_idx = seq_end_idx + self.prediction_horizon

            # Create target
            target = thermal_data_scaled[target_idx]

            # Create horizon-consistent power data with unscaled values (A2, D1)
            time_row1 = float(df[time_col].iloc[start_idx])
            temps_row1 = df[thermal_cols].iloc[start_idx].values.astype(np.float32).tolist()
            time_target = float(df[time_col].iloc[target_idx])
            temps_target = df[thermal_cols].iloc[target_idx].values.astype(np.float32).tolist()
            
            # Calculate time difference (must be > 0)
            time_diff = time_target - time_row1
            assert time_diff > 0, f"Invalid time_diff ({time_diff}) in {file_path}: time must be strictly increasing from sequence start to target"
            
            # Build power data dictionary with unscaled static params
            power_data = {
                'time_row1': time_row1,
                'temps_row1': temps_row1,
                'time_target': time_target,
                'temps_target': temps_target,
                'time_diff': float(time_diff),
                'h': float(static_params[0]),          # Unscaled for physics calculations
                'q0': float(static_params[1]),         # Unscaled flux parameter 
                'start_idx': int(start_idx),
                'target_idx': int(target_idx),
                'file_path': file_path,
                'prediction_horizon': self.prediction_horizon  # Add horizon info to power_data
            }
            
            # For backward compatibility: if prediction_horizon == 1, 
            # set time_row21/temps_row21 to match target values
            if self.prediction_horizon == 1:
                power_data['time_row21'] = time_target
                power_data['temps_row21'] = temps_target

            return time_series, static_params_scaled, target, power_data
            
        except Exception as e:
            print(f"Error creating sequence from {file_path} at index {start_idx}: {e}")
            return None

    def get_physics_params(self):
        """Return physics parameters and scalers for loss calculation."""
        return {
            'thermal_scaler': self.thermal_scaler,
            'param_scaler': self.param_scaler,
            'rho': 1836.31,  # kg/m³
            'cp': 1512.0,    # J/(kg·K)
            'radius': 0.05175,  # m
            'prediction_horizon': self.prediction_horizon  # Include horizon in physics params
        }

    def get_dataset_statistics(self):
        """Return comprehensive dataset statistics."""
        # Calculate horizon-specific info
        horizon_description = "final state" if self.prediction_horizon == -1 else f"{self.prediction_horizon} steps ahead"
        
        return {
            'file_counts': {
                'total_files': len(self.train_files) + len(self.val_files) + len(self.test_files),
                'train_files': len(self.train_files),
                'val_files': len(self.val_files),
                'test_files': len(self.test_files),
                'valid_files': self._data_stats['valid_files'],
                'invalid_files': self._data_stats['invalid_files']
            },
            'sequence_info': {
                f'{self.split}_sequences': len(self.sample_indices),
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'horizon_description': horizon_description,
                'target_calculation': 'Last timestep of file' if self.prediction_horizon == -1 else f'Sequence end + {self.prediction_horizon} timesteps'
            },
            'data_quality': {
                'files_with_outliers': self._data_stats['files_with_outliers'],
                'total_outliers_removed': self._data_stats['total_outliers'],
                'files_with_missing': self._data_stats['files_with_missing'],
                'total_missing_values': self._data_stats['total_missing'],
                'files_too_short': self._data_stats['files_too_short'],
                'median_timestep_ms': self._data_stats.get('median_dt_ms', 'N/A')
            },
            'sensor_info': {
                'thermal_sensors': 10,
                'sensor_labels': 'TC1-TC10'
            },
            'scaler_stats': {
                'thermal_mean': self.thermal_scaler.mean_.tolist() if hasattr(self.thermal_scaler, 'mean_') else None,
                'thermal_std': self.thermal_scaler.scale_.tolist() if hasattr(self.thermal_scaler, 'scale_') else None,
                'param_mean': self.param_scaler.mean_.tolist() if hasattr(self.param_scaler, 'mean_') else None,
                'param_std': self.param_scaler.scale_.tolist() if hasattr(self.param_scaler, 'scale_') else None
            },
            'horizon_metadata': {
                'prediction_horizon': self.prediction_horizon,
                'horizon_type': 'fixed_steps' if self.prediction_horizon >= 1 else 'end_of_sequence',
                'min_sequence_length_required': self.sequence_length + (1 if self.prediction_horizon == -1 else self.prediction_horizon),
                'power_calculation_compatible': True
            }
        }

    def _print_data_statistics(self):
        """Print summary of dataset statistics."""
        stats = self.get_dataset_statistics()
        horizon_info = stats['horizon_metadata']
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Files: {stats['file_counts']['train_files']} train, {stats['file_counts']['val_files']} val, {stats['file_counts']['test_files']} test")
        print(f"Valid files: {stats['file_counts']['valid_files']}, Invalid: {stats['file_counts']['invalid_files']}")
        print(f"Training sequences: {stats['sequence_info'][f'{self.split}_sequences']}")
        print(f"Prediction horizon: {horizon_info['prediction_horizon']} ({stats['sequence_info']['horizon_description']})")
        print(f"Target calculation: {stats['sequence_info']['target_calculation']}")
        print(f"Min sequence length required: {horizon_info['min_sequence_length_required']}")
        print(f"Data quality issues:")
        print(f"  - Files with outliers: {stats['data_quality']['files_with_outliers']}")
        print(f"  - Files too short: {stats['data_quality']['files_too_short']}")
        print(f"  - Files with missing data: {stats['data_quality']['files_with_missing']}")
        print(f"  - Median timestep: {stats['data_quality']['median_timestep_ms']} ms")
        print("="*50 + "\n")

    def save_statistics(self, filepath=None):
        """Save dataset statistics to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.scaler_dir, f"dataset_statistics_{self.split}_horizon{self.prediction_horizon}.json")
            
        stats = self.get_dataset_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Dataset statistics saved to {filepath}")

    @classmethod
    def load_scalers(cls, scaler_dir):
        """Load pre-fitted scalers."""
        thermal_scaler_path = os.path.join(scaler_dir, "thermal_scaler.save")
        param_scaler_path = os.path.join(scaler_dir, "param_scaler.save")
        
        if not os.path.exists(thermal_scaler_path) or not os.path.exists(param_scaler_path):
            raise FileNotFoundError(f"Scalers not found in {scaler_dir}")
            
        thermal_scaler = joblib.load(thermal_scaler_path)
        param_scaler = joblib.load(param_scaler_path)
        
        return thermal_scaler, param_scaler

    def load_pretrained_scalers(self, scaler_dir=None):
        """Load pre-fitted scalers for validation/test datasets."""
        if scaler_dir is None:
            scaler_dir = self.scaler_dir
            
        thermal_scaler, param_scaler = self.load_scalers(scaler_dir)
        self.thermal_scaler = thermal_scaler
        self.param_scaler = param_scaler
        print(f"Loaded pre-fitted scalers from {scaler_dir}")

    def validate_horizon_consistency(self):
        """Validate that the dataset is properly configured for the current horizon."""
        issues = []
        
        # Check if any sequences exist
        if len(self.sample_indices) == 0:
            issues.append(f"No valid sequences found for horizon {self.prediction_horizon}")
            
        # Sample a few sequences to verify they're created correctly
        if len(self.sample_indices) > 0:
            try:
                sample = self[0]  # Try to get first sample
                time_series, static_params, target, power_data = sample
                
                # Validate power_data has required keys
                required_keys = ['time_row1', 'temps_row1', 'time_target', 'temps_target', 
                               'time_diff', 'h', 'q0', 'prediction_horizon']
                missing_keys = [key for key in required_keys if key not in power_data]
                if missing_keys:
                    issues.append(f"Power data missing keys: {missing_keys}")
                    
                # Validate time consistency
                if power_data['time_diff'] <= 0:
                    issues.append(f"Invalid time difference: {power_data['time_diff']}")
                    
                # Validate horizon matches
                if power_data['prediction_horizon'] != self.prediction_horizon:
                    issues.append(f"Horizon mismatch: dataset={self.prediction_horizon}, sample={power_data['prediction_horizon']}")
                    
            except Exception as e:
                issues.append(f"Failed to create sample: {e}")
        
        if issues:
            print(f"Horizon validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print(f"Horizon validation passed for prediction_horizon={self.prediction_horizon}")
            return True


# Utility functions for creating DataLoaders
def create_data_loaders(data_dir, batch_size=32, num_workers=4, sequence_length=20, 
                       prediction_horizon=30, scaler_dir="models_new_theoretical"):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir (str): Directory containing CSV files
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes for data loading
        sequence_length (int): Number of input timesteps
        prediction_horizon (int): Steps ahead to predict
        scaler_dir (str): Directory to save/load scalers
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset)
    """
    
    # Create training dataset (fits scalers)
    train_dataset = TempSequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        scaler_dir=scaler_dir,
        split='train'
    )
    
    # Create validation dataset (loads pre-fitted scalers)
    val_dataset = TempSequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        scaler_dir=scaler_dir,
        split='val'
    )
    val_dataset.load_pretrained_scalers(scaler_dir)
    
    # Create test dataset (loads pre-fitted scalers)
    test_dataset = TempSequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        scaler_dir=scaler_dir,
        split='test'
    )
    test_dataset.load_pretrained_scalers(scaler_dir)
    
    # Validate horizon consistency for all datasets
    print("Validating horizon consistency...")
    train_dataset.validate_horizon_consistency()
    val_dataset.validate_horizon_consistency()
    test_dataset.validate_horizon_consistency()
    
    # Create DataLoaders with custom collate function
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, train_dataset


def collate_fn(batch):
    """
    Custom collate function for handling power_data dictionaries.
    """
    time_series, static_params, targets, power_data = zip(*batch)
    
    # Stack tensors
    time_series = torch.stack(time_series)
    static_params = torch.stack(static_params)
    targets = torch.stack(targets)
    
    # Keep power_data as list of dictionaries
    return time_series, static_params, targets, list(power_data)


# Example usage
if __name__ == "__main__":
    # Example of how to use the PyTorch dataset
    data_dir = "path/to/your/csv/files"
    
    # Create data loaders with different horizons
    print("Testing horizon=1 (next step prediction):")
    train_loader_h1, val_loader_h1, test_loader_h1, train_dataset_h1 = create_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        prediction_horizon=1
    )
    
    print("\nTesting horizon=30 (30 steps ahead):")
    train_loader_h30, val_loader_h30, test_loader_h30, train_dataset_h30 = create_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        prediction_horizon=30
    )
    
    print("\nTesting horizon=-1 (final state):")
    train_loader_hf, val_loader_hf, test_loader_hf, train_dataset_hf = create_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        prediction_horizon=-1
    )
    
    # Example training loop iteration
    for batch_idx, (time_series, static_params, targets, power_data) in enumerate(train_loader_h30):
        print(f"\nBatch {batch_idx} (horizon=30):")
        print(f"  Time series shape: {time_series.shape}")  # [batch_size, seq_len, 11]
        print(f"  Static params shape: {static_params.shape}")  # [batch_size, 4]
        print(f"  Targets shape: {targets.shape}")  # [batch_size, 10]
        print(f"  Power data length: {len(power_data)}")  # batch_size
        
        # Show sample power_data structure
        sample_power = power_data[0]
        print(f"  Sample power data keys: {list(sample_power.keys())}")
        print(f"  Time difference: {sample_power['time_diff']:.3f}")
        print(f"  Prediction horizon: {sample_power['prediction_horizon']}")
        
        if batch_idx == 0:  # Just show first batch
            break
    
    # Get physics parameters for loss calculation
    physics_params = train_dataset_h30.get_physics_params()
    print(f"\nPhysics parameters: {list(physics_params.keys())}")
    print(f"Prediction horizon in physics params: {physics_params['prediction_horizon']}")
    
    # Save dataset statistics with horizon info
    train_dataset_h30.save_statistics()
    
    # Show statistics summary
    stats = train_dataset_h30.get_dataset_statistics()
    print(f"\nHorizon metadata:")
    for key, value in stats['horizon_metadata'].items():
        print(f"  {key}: {value}")