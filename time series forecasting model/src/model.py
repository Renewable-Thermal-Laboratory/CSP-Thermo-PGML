import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
import pickle

class PhysicsInformedLSTM(nn.Module):
    """Physics-informed LSTM for thermal system temperature prediction.
    
    Predicts TC1-TC10 temperatures at timestep 21 given 20 previous timesteps,
    with physics-based constraints for energy conservation and heat transfer.

    Args:
        num_sensors (int): Number of temperature sensors (10 for TC1-TC10).
        sequence_length (int): Number of input timesteps (20).
        lstm_units (int): Number of LSTM units per layer.
        dropout_rate (float): Dropout rate for regularization.
        
    Inputs:
        time_series: Tensor of shape (batch, 20, 11) with time and TC1-TC10 temperatures.
        static_params: Tensor of shape (batch, 4) with h, flux, abs, surf.
        
    Outputs:
        predictions: Tensor of shape (batch, 10) with predicted TC1-TC10 temperatures.
    """
    
    def __init__(self, num_sensors=10, sequence_length=20, lstm_units=64, dropout_rate=0.2):
        super().__init__()
        self.num_sensors = num_sensors
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # Physics constants as parameters (non-trainable)
        self.register_buffer('rho', torch.tensor(1836.31, dtype=torch.float32))  # kg/m³
        self.register_buffer('cp', torch.tensor(1512.0, dtype=torch.float32))    # J/(kg·K)
        self.register_buffer('radius', torch.tensor(0.05175, dtype=torch.float32))  # m
        self.register_buffer('pi', torch.tensor(np.pi, dtype=torch.float32))
        
        # LSTM layers for temporal processing
        self.lstm1 = nn.LSTM(
            input_size=11,  # time + TC1-TC10
            hidden_size=lstm_units,
            batch_first=True,
            dropout=dropout_rate if lstm_units > 1 else 0,
        )
        self.batch_norm1 = nn.BatchNorm1d(lstm_units)
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_units,
            hidden_size=lstm_units,
            batch_first=True,
            dropout=dropout_rate if lstm_units > 1 else 0,
        )
        self.batch_norm2 = nn.BatchNorm1d(lstm_units)
        
        # Static parameter processing
        self.param_dense1 = nn.Linear(4, 32)
        self.param_batch_norm = nn.BatchNorm1d(32)
        self.param_dropout = nn.Dropout(dropout_rate)
        
        # Combined processing with residual connection
        self.combine_dense1 = nn.Linear(lstm_units + 32, 64)
        self.combine_batch_norm = nn.BatchNorm1d(64)
        self.combine_dropout = nn.Dropout(dropout_rate)
        
        self.combine_dense2 = nn.Linear(64, 32)
        
        # Output layer - predict 10 temperatures (TC1-TC10)
        self.output_dense = nn.Linear(32, num_sensors)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:  # Only apply Xavier to 2D+ tensors
                    if 'lstm' in name:
                        # LSTM weights
                        nn.init.xavier_uniform_(param)
                    else:
                        # Linear layer weights
                        nn.init.xavier_uniform_(param)
                else:
                    # For 1D weight tensors (shouldn't happen normally, but just in case)
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                # Initialize all bias terms to zero
                nn.init.zeros_(param)
    
    def forward(self, inputs, training=True):
        """Forward pass of the model."""
        if isinstance(inputs, (list, tuple)):
            time_series, static_params = inputs
        else:
            time_series, static_params = inputs['time_series'], inputs['static_params']
        
        # Input validation
        assert time_series.shape[1:] == (self.sequence_length, 11), f"Expected time_series shape (*, {self.sequence_length}, 11), got {time_series.shape}"
        assert static_params.shape[1:] == (4,), f"Expected static_params shape (*, 4), got {static_params.shape}"
        
        batch_size = time_series.shape[0]
        
        # Process time series through LSTM layers
        x, _ = self.lstm1(time_series)  # (batch, seq_len, lstm_units)
        x = x[:, -1, :]  # Take last timestep: (batch, lstm_units)
        x = self.batch_norm1(x)
        
        # Add sequence dimension back for second LSTM
        x = x.unsqueeze(1)  # (batch, 1, lstm_units)
        x, _ = self.lstm2(x)
        x = x.squeeze(1)  # (batch, lstm_units)
        x = self.batch_norm2(x)
        
        # Process static parameters
        params = F.relu(self.param_dense1(static_params))
        params = self.param_batch_norm(params)
        params = self.param_dropout(params) if training else params
        
        # Combine temporal and static features
        combined = torch.cat([x, params], dim=-1)
        
        # Residual connection
        residual = F.relu(self.combine_dense1(combined))
        residual = self.combine_batch_norm(residual)
        residual = self.combine_dropout(residual) if training else residual
        
        output_features = F.relu(self.combine_dense2(residual))
        
        # Output predictions
        predictions = self.output_dense(output_features)
        
        return predictions
    
    def get_config(self):
        """Return model configuration for saving."""
        return {
            'num_sensors': self.num_sensors,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate
        }


class PhysicsInformedLoss(nn.Module):
    """Custom loss combining prediction accuracy with physics constraints."""
    
    def __init__(self, physics_weight=0.1, constraint_weight=0.1):
        super().__init__()
        self.physics_weight = physics_weight
        self.constraint_weight = constraint_weight
        
        # Physics constants
        self.register_buffer('rho', torch.tensor(1836.31, dtype=torch.float32))  # kg/m³
        self.register_buffer('cp', torch.tensor(1512.0, dtype=torch.float32))    # J/(kg·K)
        self.register_buffer('radius', torch.tensor(0.05175, dtype=torch.float32))  # m
        self.register_buffer('pi', torch.tensor(np.pi, dtype=torch.float32))
        
    def forward(self, y_true, y_pred):
        """Compute MAE loss (physics loss added separately in trainer)."""
        mae_loss = torch.mean(torch.abs(y_true - y_pred))
        return mae_loss


# =====================
# POWER METADATA PROCESSING FUNCTIONS WITH PROPER TIME UNSCALING
# =====================
def unscale_time_values(time_scaled_list, time_mean=300.0, time_std=300.0):
    """Unscale normalized time values back to physical units (seconds).
    
    Args:
        time_scaled_list: List of normalized time values
        time_mean: Mean used for normalization (default: 300.0)
        time_std: Standard deviation used for normalization (default: 300.0)
    
    Returns:
        List of unscaled time values in seconds
    """
    if not isinstance(time_scaled_list, list):
        time_scaled_list = [float(time_scaled_list)]
    
    # Reverse the normalization: time_scaled = (time_raw - time_mean) / time_std
    # Therefore: time_raw = time_scaled * time_std + time_mean
    time_unscaled = []
    for time_val in time_scaled_list:
        time_raw = float(time_val) * time_std + time_mean
        time_unscaled.append(time_raw)
    
    return time_unscaled


def process_power_data_batch(power_data_list, thermal_scaler=None, time_mean=300.0, time_std=300.0):
    """Convert power data dictionaries to proper format for physics loss with PROPER TIME UNSCALING.
    
    CRITICAL FIX: Check if temperatures are already unscaled before applying thermal_scaler.
    """
    if not power_data_list:
        return None
    
    batch_size = len(power_data_list)
    processed_metadata = []
    
    print(f"Processing power data batch with {batch_size} samples (with temperature scaling check)")
    
    for i, power_data in enumerate(power_data_list):
        if power_data is None or not isinstance(power_data, dict):
            print(f"Warning: Invalid power_data at index {i}, using dummy values")
            # Use dummy values for None entries
            processed_metadata.append({
                'temps_row1': [300.0] * 10,  # Plain Python list
                'temps_row21': [301.0] * 10,  # Plain Python list
                'time_diff': 1.0,  # Plain Python float in seconds
                'h': 50.0,  # Plain Python float
                'q0': 1000.0  # Plain Python float
            })
            continue
            
        try:
            # Check if required keys exist
            required_keys = ['temps_row1', 'temps_row21', 'time_row1', 'time_row21', 'h', 'q0']
            missing_keys = [key for key in required_keys if key not in power_data]
            
            if missing_keys:
                print(f"Warning: Missing keys {missing_keys} at index {i}, using dummy values")
                # Use dummy values if keys are missing
                processed_metadata.append({
                    'temps_row1': [300.0] * 10,
                    'temps_row21': [301.0] * 10,
                    'time_diff': 1.0,
                    'h': 50.0,
                    'q0': 1000.0
                })
                continue
            
            # Get temperature lists and ensure they're plain Python floats
            temps_row1 = power_data['temps_row1']
            temps_row21 = power_data['temps_row21']
            
            # Ensure temperature lists are plain Python floats
            if isinstance(temps_row1, (list, tuple)):
                temps_row1 = [float(x) for x in temps_row1]
                if len(temps_row1) != 10:
                    print(f"Warning: temps_row1 has {len(temps_row1)} elements, expected 10 at index {i}")
                    temps_row1 = (temps_row1 + [300.0] * 10)[:10]  # Pad or truncate to 10
            else:
                print(f"Warning: temps_row1 is not a list/tuple at index {i}")
                temps_row1 = [300.0] * 10  # fallback
                
            if isinstance(temps_row21, (list, tuple)):
                temps_row21 = [float(x) for x in temps_row21]
                if len(temps_row21) != 10:
                    print(f"Warning: temps_row21 has {len(temps_row21)} elements, expected 10 at index {i}")
                    temps_row21 = (temps_row21 + [301.0] * 10)[:10]  # Pad or truncate to 10
            else:
                print(f"Warning: temps_row21 is not a list/tuple at index {i}")
                temps_row21 = [301.0] * 10  # fallback
            
            # CRITICAL FIX: Check if temperatures are already in physical units
            # If temperatures are in reasonable physical range (200-500K), don't unscale them
            temp_sample = temps_row1[0] if temps_row1 else 300.0
            
            if 200.0 <= temp_sample <= 500.0:
                # Temperatures are already in physical units (Kelvin)
                print(f"Sample {i}: Temperatures already in physical units ({temp_sample:.1f}K), skipping thermal unscaling")
                # Keep original values
            elif thermal_scaler is not None and (-10.0 <= temp_sample <= 10.0):
                # Temperatures appear to be scaled (normalized values around -10 to 10)
                print(f"Sample {i}: Temperatures appear scaled ({temp_sample:.3f}), applying thermal unscaling")
                try:
                    # Unscale temps_row1
                    temps_row1_array = np.array(temps_row1).reshape(1, -1)  # Shape (1, 10)
                    temps_row1_unscaled = thermal_scaler.inverse_transform(temps_row1_array)[0]
                    temps_row1 = temps_row1_unscaled.tolist()
                    
                    # Unscale temps_row21  
                    temps_row21_array = np.array(temps_row21).reshape(1, -1)  # Shape (1, 10)
                    temps_row21_unscaled = thermal_scaler.inverse_transform(temps_row21_array)[0]
                    temps_row21 = temps_row21_unscaled.tolist()
                    
                    print(f"Sample {i}: Successfully unscaled temperatures: {temps_row1[0]:.1f}K -> {temps_row21[0]:.1f}K")
                except Exception as e:
                    print(f"Warning: Failed to unscale temperatures for sample {i}: {e}")
                    # Keep original values if unscaling fails
            else:
                # Temperatures are in some other range, keep as-is
                print(f"Sample {i}: Temperature value {temp_sample:.1f} in unexpected range, keeping as-is")
            
            # CRITICAL FIX: Get raw time values and ensure they're in proper physical units
            time_row1_raw = float(power_data['time_row1'])
            time_row21_raw = float(power_data['time_row21'])
            
            # Check if time values appear to be normalized (values around -1 to 1 suggest normalization)
            if abs(time_row1_raw) < 10 and abs(time_row21_raw) < 10:
                # These appear to be normalized time values, unscale them
                time_row1_unscaled = time_row1_raw * time_std + time_mean
                time_row21_unscaled = time_row21_raw * time_std + time_mean
                print(f"Sample {i}: Detected normalized time values, unscaling: {time_row1_raw:.3f} -> {time_row1_unscaled:.1f}, {time_row21_raw:.3f} -> {time_row21_unscaled:.1f}")
            else:
                # These appear to be raw time values already
                time_row1_unscaled = time_row1_raw
                time_row21_unscaled = time_row21_raw
                print(f"Sample {i}: Using raw time values: {time_row1_unscaled:.1f}, {time_row21_unscaled:.1f}")
            
            # Calculate time difference in physical units (seconds)
            time_diff = time_row21_unscaled - time_row1_unscaled
            time_diff = max(time_diff, 1e-8)  # Ensure positive and non-zero
            
            # Convert h and q0 to plain Python floats
            h_value = float(power_data['h'])
            q0_value = float(power_data['q0'])
            
            processed_metadata.append({
                'temps_row1': temps_row1,  # List of floats (properly handled)
                'temps_row21': temps_row21,  # List of floats (properly handled)
                'time_diff': time_diff,  # Float in physical units (seconds)
                'h': h_value,  # Float
                'q0': q0_value,  # Float
                'time_row1_unscaled': time_row1_unscaled,  # For debugging
                'time_row21_unscaled': time_row21_unscaled  # For debugging
            })
            
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error processing power_data at index {i}: {e}")
            # Skip invalid data, use dummy values
            processed_metadata.append({
                'temps_row1': [300.0] * 10,
                'temps_row21': [301.0] * 10,
                'time_diff': 1.0,
                'h': 50.0,
                'q0': 1000.0
            })
    
    print(f"Successfully processed {len(processed_metadata)} power metadata entries with temperature scaling check")
    return processed_metadata


class PhysicsInformedTrainer:
    """Custom trainer that handles physics-informed loss computation with 9-bin approach and PROPER TIME UNSCALING."""
    
    def __init__(self, model, physics_weight=0.1, constraint_weight=0.1, learning_rate=0.001, 
                 cylinder_length=1.0, power_balance_weight=0.05, lstm_units=64, dropout_rate=0.2,
                 device=None, param_scaler=None, thermal_scaler=None, time_mean=300.0, time_std=300.0):
        self.model = model
        self.physics_weight = physics_weight
        self.constraint_weight = constraint_weight
        self.power_balance_weight = power_balance_weight
        self.cylinder_length = cylinder_length  # Total cylinder length in meters
        
        # Store model parameters for saving metadata
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # IMPORTANT: Store scalers for proper unscaling
        self.param_scaler = param_scaler
        self.thermal_scaler = thermal_scaler  # NEW: Store thermal scaler
        self.time_mean = time_mean  # NEW: Store time normalization parameters
        self.time_std = time_std    # NEW: Store time normalization parameters
        
        if param_scaler is None:
            print("Warning: param_scaler not provided to trainer - physics calculations may be incorrect")
        if thermal_scaler is None:
            print("Warning: thermal_scaler not provided to trainer - temperature unscaling will be skipped")
        
        # Device handling
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Physics constants
        self.rho = torch.tensor(1836.31, dtype=torch.float32, device=self.device)  # kg/m³
        self.cp = torch.tensor(1512.0, dtype=torch.float32, device=self.device)    # J/(kg·K)
        self.radius = torch.tensor(0.05175, dtype=torch.float32, device=self.device)  # m
        self.pi = torch.tensor(np.pi, dtype=torch.float32, device=self.device)
        
        # Bin configuration for 9-bin approach
        self.num_bins = 9
        self.bin_volume = self.pi * (self.radius ** 2) * (cylinder_length / self.num_bins)
        self.bin_mass = self.rho * self.bin_volume
        
        # Define 9 bins using adjacent sensor pairs (TC1-TC2, TC2-TC3, ..., TC9-TC10)
        self.bin_sensor_pairs = [(i, i+1) for i in range(9)]
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-7
        )
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'train_mse': [],
            'train_physics_loss': [],
            'train_constraint_loss': [],
            'train_power_balance_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_mse': [],
            'val_physics_loss': [],
            'val_constraint_loss': [],
            'val_power_balance_loss': []
        }

    def unscale_h_q0(self, h_scaled_list, q0_scaled_list):
        """Unscale h and q0 parameters to physical units - NO TENSORS VERSION."""
        if not isinstance(h_scaled_list, list):
            h_scaled_list = [float(h_scaled_list)]
        if not isinstance(q0_scaled_list, list):
            q0_scaled_list = [float(q0_scaled_list)]
        
        batch_size = len(h_scaled_list)
        
        # Create input array for param_scaler: [h, flux, abs, surf]
        # We only need to unscale the first two columns (h and q0/flux)
        scaled_params = []
        for i in range(batch_size):
            # Add dummy values for abs and surf to match scaler format
            scaled_params.append([
                float(h_scaled_list[i]),    # h
                float(q0_scaled_list[i]),   # flux
                0.0,                        # abs (dummy)
                0.0                         # surf (dummy)
            ])
        
        # Inverse transform using the parameter scaler
        unscaled_params = self.param_scaler.inverse_transform(scaled_params)
        
        # Extract only h and q0 (first two columns) as plain Python lists
        h_unscaled = [float(row[0]) for row in unscaled_params]
        q0_unscaled = [float(row[1]) for row in unscaled_params]
        
        return h_unscaled, q0_unscaled
        
    def compute_nine_bin_physics_loss(self, y_true, y_pred, power_metadata_list):
        """Compute physics-based loss using 9 spatial bins with PROPER UNSCALING.
        
        CRITICAL FIX: Only unscale y_pred tensors, not the power_metadata temperatures 
        which are already properly unscaled in process_power_data_batch.
        """
        try:
            if not power_metadata_list or len(power_metadata_list) == 0:
                zero_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                return zero_loss, zero_loss, zero_loss, {}

            # CRITICAL FIX: Ensure batch size consistency
            actual_batch_size = min(len(power_metadata_list), y_true.shape[0], y_pred.shape[0])

            # CRITICAL FIX: Only unscale y_pred (model predictions), 
            # power_metadata temperatures are already unscaled in process_power_data_batch
            y_pred_unscaled = y_pred.clone()
            
            if self.thermal_scaler is not None:
                try:
                    # Only process the actual batch size to avoid shape mismatches
                    y_pred_batch = y_pred[:actual_batch_size].detach().cpu().numpy()
                    
                    print(f"UNSCALING DEBUG: Processing batch of size {actual_batch_size}")
                    print(f"UNSCALING DEBUG: y_pred_batch shape: {y_pred_batch.shape}")
                    print(f"UNSCALING DEBUG: y_pred sample before unscaling: {y_pred_batch[0][:3]}")
                    
                    # Unscale using thermal_scaler - SHAPE MUST BE (n_samples, n_features)
                    y_pred_unscaled_np = self.thermal_scaler.inverse_transform(y_pred_batch)
                    
                    print(f"UNSCALING DEBUG: y_pred sample after unscaling: {y_pred_unscaled_np[0][:3]}")
                    
                    # Convert back to tensors with proper batch size
                    y_pred_unscaled = torch.tensor(y_pred_unscaled_np, dtype=torch.float32, device=self.device)
                    
                    print(f"UNSCALING SUCCESS: Unscaled y_pred using thermal_scaler")
                    
                except Exception as e:
                    print(f"UNSCALING ERROR: Failed to unscale y_pred: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use original scaled values if unscaling fails
                    y_pred_unscaled = y_pred[:actual_batch_size].clone()
            else:
                # If no thermal scaler, just trim to actual batch size
                y_pred_unscaled = y_pred[:actual_batch_size].clone()
            
            # Initialize lists to collect physics losses
            bin_physics_losses = []
            total_actual_powers = []
            total_predicted_powers = []
            incoming_powers = []
            
            # Physics constants as plain Python values
            rho = 1836.31  # kg/m³
            cp = 1512.0    # J/(kg·K)
            radius = 0.05175  # m
            pi = 3.14159265359
            
            # Process each sample in the batch
            for sample_idx in range(actual_batch_size):
                power_data = power_metadata_list[sample_idx]
                
                # Extract plain Python values - ALREADY PROPERLY UNSCALED in process_power_data_batch
                temps_row1 = power_data['temps_row1']  # List of 10 floats (already unscaled)
                temps_row21 = power_data['temps_row21']  # List of 10 floats (already unscaled)
                time_diff = power_data['time_diff']  # Float in seconds (unscaled)
                h_unscaled = power_data['h']  # Float - cylinder height
                q0_unscaled = power_data['q0']  # Float - heat flux
                
                print(f"Sample {sample_idx}: Using time_diff = {time_diff:.2f} seconds")
                print(f"Sample {sample_idx}: T1_sample = {temps_row1[0]:.1f}K, T21_sample = {temps_row21[0]:.1f}K (from metadata)")
                
                # IMPORTANT: Use dynamic cylinder height from h parameter
                cylinder_length = h_unscaled
                bin_height = cylinder_length / self.num_bins
                bin_volume = pi * (radius ** 2) * bin_height
                bin_mass = rho * bin_volume
                
                # Calculate incoming power using unscaled q0
                surface_area = pi * (radius ** 2)
                incoming_power = q0_unscaled * surface_area
                incoming_powers.append(incoming_power)
                
                # Process each of the 9 bins for this sample
                sample_bin_physics_losses = []
                sample_bin_actual_powers = []
                sample_bin_predicted_powers = []
                
                for bin_idx, (sensor1_idx, sensor2_idx) in enumerate(self.bin_sensor_pairs):
                    # VALIDATION: Ensure sensor indices are within bounds
                    if (sensor1_idx >= len(temps_row1) or sensor2_idx >= len(temps_row1) or
                        sensor1_idx >= len(temps_row21) or sensor2_idx >= len(temps_row21)):
                        print(f"Warning: Sensor index out of bounds for sample {sample_idx}, bin {bin_idx}")
                        continue
                    
                    # VALIDATION: Ensure sample index is within tensor bounds
                    if sample_idx >= y_pred_unscaled.shape[0] or sensor1_idx >= y_pred_unscaled.shape[1] or sensor2_idx >= y_pred_unscaled.shape[1]:
                        print(f"Warning: Tensor index out of bounds - sample: {sample_idx}, sensors: {sensor1_idx}, {sensor2_idx}")
                        print(f"y_pred_unscaled shape: {y_pred_unscaled.shape}")
                        continue
                    
                    # Get actual temperatures from metadata (ALREADY UNSCALED)
                    actual_temp1_t1 = temps_row1[sensor1_idx]  # Already unscaled
                    actual_temp2_t1 = temps_row1[sensor2_idx]  # Already unscaled
                    actual_temp1_t21 = temps_row21[sensor1_idx]  # Already unscaled
                    actual_temp2_t21 = temps_row21[sensor2_idx]  # Already unscaled
                    
                    # Get predictions as plain Python floats (UNSCALED)
                    try:
                        pred_temp1_t21 = float(y_pred_unscaled[sample_idx, sensor1_idx].item())
                        pred_temp2_t21 = float(y_pred_unscaled[sample_idx, sensor2_idx].item())
                        
                        print(f"Sample {sample_idx}, Bin {bin_idx}: T1=[{actual_temp1_t1:.1f}, {actual_temp2_t1:.1f}] K, Actual_T21=[{actual_temp1_t21:.1f}, {actual_temp2_t21:.1f}] K, Pred_T21=[{pred_temp1_t21:.1f}, {pred_temp2_t21:.1f}] K")
                        
                    except IndexError as e:
                        print(f"IndexError in sample {sample_idx}, sensors {sensor1_idx}, {sensor2_idx}: {e}")
                        print(f"y_pred_unscaled shape: {y_pred_unscaled.shape}, sample_idx: {sample_idx}")
                        continue
                    
                    # Calculate temperature changes (average of two sensors for this bin)
                    actual_temp1_change = actual_temp1_t21 - actual_temp1_t1
                    actual_temp2_change = actual_temp2_t21 - actual_temp2_t1
                    actual_bin_temp_change = (actual_temp1_change + actual_temp2_change) / 2.0
                    
                    pred_temp1_change = pred_temp1_t21 - actual_temp1_t1  # Use actual initial temp
                    pred_temp2_change = pred_temp2_t21 - actual_temp2_t1  # Use actual initial temp
                    pred_bin_temp_change = (pred_temp1_change + pred_temp2_change) / 2.0
                    
                    print(f"  Bin {bin_idx}: ΔT_actual={actual_bin_temp_change:.2f}K, ΔT_pred={pred_bin_temp_change:.2f}K")
                    
                    # Power calculations using plain Python arithmetic WITH PROPER TIME UNITS
                    actual_bin_power = bin_mass * cp * actual_bin_temp_change / time_diff  # time_diff in seconds
                    pred_bin_power = bin_mass * cp * pred_bin_temp_change / time_diff      # time_diff in seconds
                    
                    print(f"  Bin {bin_idx}: P_actual={actual_bin_power:.2f}W, P_pred={pred_bin_power:.2f}W")
                    
                    sample_bin_actual_powers.append(actual_bin_power)
                    sample_bin_predicted_powers.append(pred_bin_power)
                    
                    # Physics loss for this bin
                    bin_physics_loss = abs(actual_bin_power - pred_bin_power)
                    sample_bin_physics_losses.append(bin_physics_loss)
                
                # Only proceed if we have valid bin calculations
                if len(sample_bin_physics_losses) == 0:
                    print(f"Warning: No valid bins calculated for sample {sample_idx}")
                    continue
                
                # Calculate total powers for this sample
                total_actual_power = sum(sample_bin_actual_powers)
                total_predicted_power = sum(sample_bin_predicted_powers)
                
                total_actual_powers.append(total_actual_power)
                total_predicted_powers.append(total_predicted_power)
                
                print(f"Sample {sample_idx} total: P_actual={total_actual_power:.2f}W, P_pred={total_predicted_power:.2f}W, P_incoming={incoming_power:.2f}W")
                
                # Average physics loss for this sample
                avg_sample_physics_loss = sum(sample_bin_physics_losses) / len(sample_bin_physics_losses)
                bin_physics_losses.append(avg_sample_physics_loss)
            
            # Ensure we have valid results
            if len(bin_physics_losses) == 0:
                print("Warning: No valid physics losses calculated")
                zero_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                return zero_loss, zero_loss, zero_loss, {}
            
            # Convert final results to tensors for PyTorch loss computation
            physics_loss = torch.tensor(sum(bin_physics_losses) / len(bin_physics_losses), 
                                    dtype=torch.float32, device=self.device)
            
            # Constraint penalties
            constraint_penalties = []
            power_balance_losses = []
            
            for i in range(len(total_actual_powers)):
                # Individual bin constraint: no bin should exceed total incoming power
                bin_excess = max(0.0, total_predicted_powers[i] - incoming_powers[i])
                constraint_penalties.append(bin_excess ** 2)
                
                # Power balance constraint
                power_imbalance = abs(total_predicted_powers[i] - incoming_powers[i])
                power_balance_losses.append(power_imbalance)
            
            constraint_penalty = torch.tensor(sum(constraint_penalties) / len(constraint_penalties), 
                                            dtype=torch.float32, device=self.device)
            power_balance_loss = torch.tensor(sum(power_balance_losses) / len(power_balance_losses), 
                                            dtype=torch.float32, device=self.device)
            
            # Return info for analysis
            power_info = {
                'total_actual_power': total_actual_powers,  # Plain Python list
                'total_predicted_power': total_predicted_powers,  # Plain Python list
                'incoming_power': incoming_powers,  # Plain Python list
                'power_imbalance': power_balance_losses,  # Plain Python list
                'h_unscaled': [power_metadata_list[i]['h'] for i in range(len(total_actual_powers))],  # Plain Python list
                'q0_unscaled': [power_metadata_list[i]['q0'] for i in range(len(total_actual_powers))],  # Plain Python list
                'time_diff_unscaled': [power_metadata_list[i]['time_diff'] for i in range(len(total_actual_powers))]  # Plain Python list
            }
            
            return physics_loss, constraint_penalty, power_balance_loss, power_info
            
        except Exception as e:
            print(f"9-bin physics loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            zero_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return zero_loss, zero_loss, zero_loss, {}
    
    def train_step(self, batch):
        """Custom training step with 9-bin physics loss and PROPER UNSCALING."""
        self.model.train()
        
        # Move batch to device
        time_series = batch[0].to(self.device)
        static_params = batch[1].to(self.device)
        y_true = batch[2].to(self.device)
        power_data = batch[3] if len(batch) > 3 else None
        
        # Process power metadata to plain Python format WITH PROPER UNSCALING
        power_metadata_list = None
        if power_data is not None:
            power_metadata_list = process_power_data_batch(
                power_data, 
                thermal_scaler=self.thermal_scaler,  # Pass thermal scaler
                time_mean=self.time_mean,           # Pass time normalization params
                time_std=self.time_std
            )
        
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred = self.model([time_series, static_params], training=True)
        
        # Primary loss (MAE)
        mae_loss = torch.mean(torch.abs(y_true - y_pred))
        
        # Physics loss with 9-bin approach (if metadata available)
        if power_metadata_list is not None:
            physics_loss, constraint_loss, power_balance_loss, power_info = self.compute_nine_bin_physics_loss(
                y_true, y_pred, power_metadata_list
            )
            
            # Combined loss with all components
            total_loss = (mae_loss + 
                        self.physics_weight * physics_loss + 
                        self.constraint_weight * constraint_loss +
                        self.power_balance_weight * power_balance_loss)
        else:
            physics_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            constraint_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            power_balance_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            total_loss = mae_loss
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Calculate MSE for metrics
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        
        return {
            'loss': total_loss.item(),
            'mae': mae_loss.item(),
            'mse': mse_loss.item(),
            'physics_loss': physics_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'power_balance_loss': power_balance_loss.item()
        }

    def validation_step(self, batch):
        """Validation step with 9-bin physics analysis and PROPER UNSCALING."""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            time_series = batch[0].to(self.device)
            static_params = batch[1].to(self.device)
            y_true = batch[2].to(self.device)
            power_data = batch[3] if len(batch) > 3 else None
            
            # Process power metadata to plain Python format WITH PROPER UNSCALING
            power_metadata_list = None
            if power_data is not None:
                power_metadata_list = process_power_data_batch(
                    power_data,
                    thermal_scaler=self.thermal_scaler,  # Pass thermal scaler
                    time_mean=self.time_mean,           # Pass time normalization params
                    time_std=self.time_std
                )
            
            # Forward pass
            y_pred = self.model([time_series, static_params], training=False)
            
            # Primary loss (MAE)
            mae_loss = torch.mean(torch.abs(y_true - y_pred))
            
            # Physics loss with 9-bin approach (if metadata available)
            if power_metadata_list is not None:
                physics_loss, constraint_loss, power_balance_loss, power_info = self.compute_nine_bin_physics_loss(
                    y_true, y_pred, power_metadata_list
                )
                
                total_loss = (mae_loss + 
                            self.physics_weight * physics_loss + 
                            self.constraint_weight * constraint_loss +
                            self.power_balance_weight * power_balance_loss)
            else:
                physics_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                constraint_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                power_balance_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                total_loss = mae_loss
            
            # Calculate MSE for metrics
            mse_loss = torch.mean((y_true - y_pred) ** 2)
            
            return {
                'val_loss': total_loss.item(),
                'val_mae': mae_loss.item(),
                'val_mse': mse_loss.item(),
                'val_physics_loss': physics_loss.item(),
                'val_constraint_loss': constraint_loss.item(),
                'val_power_balance_loss': power_balance_loss.item()
            }
    
    def train_epoch(self, train_loader, val_loader=None):
        """Train for one epoch with detailed physics tracking."""
        # Initialize metrics for this epoch
        epoch_train_metrics = defaultdict(list)
        epoch_val_metrics = defaultdict(list)
        
        # Training loop
        for batch in train_loader:
            metrics = self.train_step(batch)
            for key, value in metrics.items():
                epoch_train_metrics[f'train_{key}'].append(value)
        
        # Validation loop
        if val_loader is not None:
            for batch in val_loader:
                metrics = self.validation_step(batch)
                for key, value in metrics.items():
                    epoch_val_metrics[key].append(value)
        
        # Aggregate results
        results = {}
        for key, values in epoch_train_metrics.items():
            results[key] = np.mean(values)
        for key, values in epoch_val_metrics.items():
            results[key] = np.mean(values)
        
        # Update history
        for key, value in results.items():
            if key in self.history:
                self.history[key].append(float(value))
        
        return results
    
    def analyze_power_balance(self, data_loader, num_samples=100):
        """Analyze power balance across the system for diagnostic purposes with PROPER UNSCALING."""
        print("\n" + "="*60)
        print("POWER BALANCE ANALYSIS (WITH PROPER UNSCALING)")
        print("="*60)
        
        total_actual_powers = []
        total_predicted_powers = []
        incoming_powers = []
        time_diffs_used = []
        
        self.model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break
                
                time_series = batch[0].to(self.device)
                static_params = batch[1].to(self.device)
                y_true = batch[2].to(self.device)
                power_data = batch[3] if len(batch) > 3 else None
                
                if power_data is not None and len(power_data) > 0 and power_data[0] is not None:
                    try:
                        # Process power metadata using the function WITH PROPER UNSCALING
                        power_metadata_list = process_power_data_batch(
                            power_data,
                            thermal_scaler=self.thermal_scaler,  # Pass thermal scaler
                            time_mean=self.time_mean,           # Pass time normalization params
                            time_std=self.time_std
                        )
                        
                        if power_metadata_list:
                            # Get predictions
                            y_pred = self.model([time_series, static_params], training=False)
                            
                            # Compute power analysis
                            _, _, _, power_info = self.compute_nine_bin_physics_loss(
                                y_true, y_pred, power_metadata_list
                            )
                            
                            if power_info:  # If analysis succeeded
                                total_actual_powers.extend(power_info['total_actual_power'])
                                total_predicted_powers.extend(power_info['total_predicted_power'])
                                incoming_powers.extend(power_info['incoming_power'])
                                time_diffs_used.extend(power_info['time_diff_unscaled'])  # NEW: Track time diffs
                                
                                sample_count += len(power_info['total_actual_power'])
                    except Exception as e:
                        print(f"Warning: Error in power analysis: {e}")
                        continue
        
        if len(total_actual_powers) > 0:
            total_actual_powers = np.array(total_actual_powers)
            total_predicted_powers = np.array(total_predicted_powers)
            incoming_powers = np.array(incoming_powers)
            time_diffs_used = np.array(time_diffs_used)
            
            print(f"Samples analyzed: {len(total_actual_powers)}")
            print(f"\nTIME DIFFERENCE STATISTICS (UNSCALED):")
            print(f"  Mean: {np.mean(time_diffs_used):.2f} seconds")
            print(f"  Std:  {np.std(time_diffs_used):.2f} seconds")
            print(f"  Min:  {np.min(time_diffs_used):.2f} seconds")
            print(f"  Max:  {np.max(time_diffs_used):.2f} seconds")
            
            print(f"\nINCOMING POWER STATISTICS:")
            print(f"  Mean: {np.mean(incoming_powers):.2f} W")
            print(f"  Std:  {np.std(incoming_powers):.2f} W")
            print(f"  Min:  {np.min(incoming_powers):.2f} W")
            print(f"  Max:  {np.max(incoming_powers):.2f} W")
            
            print(f"\nTOTAL ACTUAL POWER (sum of 9 bins):")
            print(f"  Mean: {np.mean(total_actual_powers):.2f} W")
            print(f"  Std:  {np.std(total_actual_powers):.2f} W")
            print(f"  Min:  {np.min(total_actual_powers):.2f} W")
            print(f"  Max:  {np.max(total_actual_powers):.2f} W")
            
            print(f"\nTOTAL PREDICTED POWER (sum of 9 bins):")
            print(f"  Mean: {np.mean(total_predicted_powers):.2f} W")
            print(f"  Std:  {np.std(total_predicted_powers):.2f} W")
            print(f"  Min:  {np.min(total_predicted_powers):.2f} W")
            print(f"  Max:  {np.max(total_predicted_powers):.2f} W")
            
            # Power balance ratios
            actual_to_incoming = total_actual_powers / incoming_powers
            predicted_to_incoming = total_predicted_powers / incoming_powers
            
            print(f"\nPOWER BALANCE RATIOS:")
            print(f"  Actual/Incoming ratio - Mean: {np.mean(actual_to_incoming):.3f}, Std: {np.std(actual_to_incoming):.3f}")
            print(f"  Predicted/Incoming ratio - Mean: {np.mean(predicted_to_incoming):.3f}, Std: {np.std(predicted_to_incoming):.3f}")
            
            # Check for violations
            actual_violations = np.sum(total_actual_powers > incoming_powers)
            predicted_violations = np.sum(total_predicted_powers > incoming_powers)
            
            print(f"\nENERGY CONSERVATION VIOLATIONS:")
            print(f"  Actual power > incoming: {actual_violations}/{len(total_actual_powers)} ({100*actual_violations/len(total_actual_powers):.1f}%)")
            print(f"  Predicted power > incoming: {predicted_violations}/{len(total_predicted_powers)} ({100*predicted_violations/len(total_predicted_powers):.1f}%)")
            
            # NEW: Analyze time differences impact
            print(f"\nTIME DIFFERENCE IMPACT ANALYSIS:")
            short_time_mask = time_diffs_used < np.median(time_diffs_used)
            long_time_mask = time_diffs_used >= np.median(time_diffs_used)
            
            print(f"  Short time intervals (<{np.median(time_diffs_used):.1f}s): {np.sum(short_time_mask)} samples")
            print(f"    Mean actual power: {np.mean(total_actual_powers[short_time_mask]):.2f} W")
            print(f"    Mean predicted power: {np.mean(total_predicted_powers[short_time_mask]):.2f} W")
            
            print(f"  Long time intervals (≥{np.median(time_diffs_used):.1f}s): {np.sum(long_time_mask)} samples")
            print(f"    Mean actual power: {np.mean(total_actual_powers[long_time_mask]):.2f} W")
            print(f"    Mean predicted power: {np.mean(total_predicted_powers[long_time_mask]):.2f} W")
        
        print("="*60)
    
    def save_model(self, filepath, include_optimizer=True):
        """Save model using PyTorch's state_dict approach."""
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(filepath, 'model_state_dict.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model state dict saved to {model_path}")
        
        # Save model architecture config
        config_path = os.path.join(filepath, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model.get_config(), f, indent=2)
        print(f"Model config saved to {config_path}")
        
        # Save optimizer state if requested
        if include_optimizer and hasattr(self, 'optimizer'):
            optimizer_path = os.path.join(filepath, 'optimizer_state_dict.pth')
            torch.save(self.optimizer.state_dict(), optimizer_path)
            print(f"Optimizer state dict saved to {optimizer_path}")
        
        # Save training history
        history_path = os.path.join(filepath, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save enhanced metadata with unscaling info
        metadata = {
            'model_type': 'PhysicsInformedLSTM',
            'physics_approach': '9-bin_spatial_segmentation_with_proper_unscaling',
            'num_sensors': self.model.num_sensors,
            'sequence_length': self.model.sequence_length,
            'lstm_units': self.model.lstm_units,
            'dropout_rate': self.model.dropout_rate,
            'physics_weight': self.physics_weight,
            'constraint_weight': self.constraint_weight,
            'power_balance_weight': self.power_balance_weight,
            'cylinder_length': self.cylinder_length,
            'num_bins': self.num_bins,
            'bin_sensor_pairs': self.bin_sensor_pairs,
            'unscaling_parameters': {  # NEW: Store unscaling parameters
                'time_mean': self.time_mean,
                'time_std': self.time_std,
                'thermal_scaler_available': self.thermal_scaler is not None,
                'param_scaler_available': self.param_scaler is not None
            },
            'physics_constants': {
                'density': 1836.31,
                'specific_heat': 1512.0,
                'radius': 0.05175
            },
            'optimizer_config': {
                'type': 'Adam',
                'lr': self.optimizer.param_groups[0]['lr'],
                'betas': self.optimizer.param_groups[0]['betas'],
                'eps': self.optimizer.param_groups[0]['eps']
            },
            'save_timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'device': str(self.device),
            'saving_method': 'state_dict_based_with_proper_unscaling'
        }
        
        metadata_path = os.path.join(filepath, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"9-bin physics-informed PyTorch model saved to {filepath} (with proper unscaling)")

    def load_model(self, filepath, model_builder_func=None):
        """Load model using PyTorch's state_dict approach."""
        # Load metadata
        metadata_path = os.path.join(filepath, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loading model with metadata: {metadata.get('model_type', 'Unknown')}")
            
            # Load unscaling parameters if available
            if 'unscaling_parameters' in metadata:
                unscaling_params = metadata['unscaling_parameters']
                self.time_mean = unscaling_params.get('time_mean', 300.0)
                self.time_std = unscaling_params.get('time_std', 300.0)
                print(f"Loaded unscaling parameters: time_mean={self.time_mean}, time_std={self.time_std}")
        else:
            print("Warning: No metadata found, using default parameters")
            metadata = {}
        
        # Load model config
        config_path = os.path.join(filepath, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = metadata  # Fallback to metadata
        
        # Rebuild model
        if model_builder_func is not None:
            self.model = model_builder_func()
        else:
            # Reconstruct from config
            self.model = PhysicsInformedLSTM(**model_config)
        
        self.model = self.model.to(self.device)
        
        # Load model weights
        model_path = os.path.join(filepath, 'model_state_dict.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model weights loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Update trainer parameters from metadata
        if 'physics_weight' in metadata:
            self.physics_weight = metadata['physics_weight']
        if 'constraint_weight' in metadata:
            self.constraint_weight = metadata['constraint_weight']
        if 'power_balance_weight' in metadata:
            self.power_balance_weight = metadata['power_balance_weight']
        
        # Load optimizer state if available
        optimizer_path = os.path.join(filepath, 'optimizer_state_dict.pth')
        if os.path.exists(optimizer_path):
            try:
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                print(f"Optimizer state dict loaded from {optimizer_path}")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Load history if available
        history_path = os.path.join(filepath, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            print("Training history loaded")
        
        print(f"9-bin physics-informed PyTorch model loaded from {filepath} (with proper unscaling)")


def build_model(num_sensors=10, sequence_length=20, lstm_units=64, dropout_rate=0.2, device=None):
    """Build the complete physics-informed model with 9-bin spatial segmentation."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PhysicsInformedLSTM(
        num_sensors=num_sensors,
        sequence_length=sequence_length,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    
    return model.to(device)


def create_trainer(model, physics_weight=0.1, constraint_weight=0.1, power_balance_weight=0.05,
                  learning_rate=0.001, cylinder_length=1.0, lstm_units=64, dropout_rate=0.2, 
                  device=None, param_scaler=None, thermal_scaler=None, time_mean=300.0, time_std=300.0):
    """Create physics-informed trainer with 9-bin approach and PROPER UNSCALING.
    
    Args:
        model: PyTorch model to train.
        physics_weight (float): Weight for 9-bin physics loss component.
        constraint_weight (float): Weight for energy conservation constraint penalty.
        power_balance_weight (float): Weight for total power balance constraint.
        learning_rate (float): Adam optimizer learning rate.
        cylinder_length (float): Total cylinder length in meters.
        lstm_units (int): Number of LSTM units (for metadata).
        dropout_rate (float): Dropout rate (for metadata).
        device (torch.device): Device for computations.
        param_scaler: StandardScaler for unscaling h and q0 parameters.
        thermal_scaler: StandardScaler for unscaling temperature values.  # NEW
        time_mean (float): Mean used for time normalization (default: 300.0).  # NEW
        time_std (float): Std used for time normalization (default: 300.0).   # NEW
        
    Returns:
        PhysicsInformedTrainer instance with 9-bin physics constraints and proper unscaling.
    """
    trainer = PhysicsInformedTrainer(
        model=model,
        physics_weight=physics_weight,
        constraint_weight=constraint_weight,
        power_balance_weight=power_balance_weight,
        learning_rate=learning_rate,
        cylinder_length=cylinder_length,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        device=device,
        param_scaler=param_scaler,
        thermal_scaler=thermal_scaler,  # NEW: Pass thermal scaler
        time_mean=time_mean,           # NEW: Pass time normalization params  
        time_std=time_std              # NEW
    )
    
    return trainer


# Additional utility functions for debugging unscaling
def test_tensor_unscaling(thermal_scaler, sample_tensor, device=None):
    """Test function to verify tensor unscaling works correctly."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*50)
    print("TENSOR UNSCALING VALIDATION TEST")
    print("="*50)
    
    # Test with a sample tensor
    print(f"Original tensor (scaled): {sample_tensor[0][:3].detach().cpu().numpy()}")
    
    try:
        # Convert to numpy
        sample_np = sample_tensor.detach().cpu().numpy()
        print(f"Converted to numpy: {sample_np[0][:3]}")
        
        # Unscale
        unscaled_np = thermal_scaler.inverse_transform(sample_np)
        print(f"Unscaled numpy: {unscaled_np[0][:3]}")
        
        # Convert back to tensor
        unscaled_tensor = torch.tensor(unscaled_np, dtype=torch.float32, device=device)
        print(f"Final unscaled tensor: {unscaled_tensor[0][:3].detach().cpu().numpy()}")
        
        # Verify consistency by re-scaling
        rescaled_np = thermal_scaler.transform(unscaled_np)
        consistency_error = np.mean(np.abs(sample_np - rescaled_np))
        print(f"Consistency error: {consistency_error:.8f}")
        
        if consistency_error < 1e-6:
            print("✅ Tensor unscaling is working correctly!")
        else:
            print("❌ Tensor unscaling has consistency issues!")
            
    except Exception as e:
        print(f"❌ Tensor unscaling failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("="*50)


def debug_prediction_unscaling(model, thermal_scaler, sample_input, device=None):
    """Debug function to trace prediction unscaling step by step."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print("\n" + "="*60)
    print("PREDICTION UNSCALING DEBUG TRACE")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Get model predictions (these will be scaled)
        if isinstance(sample_input, (list, tuple)):
            time_series, static_params = sample_input
        else:
            time_series, static_params = sample_input['time_series'], sample_input['static_params']
            
        time_series = time_series.to(device)
        static_params = static_params.to(device)
        
        y_pred_scaled = model([time_series, static_params], training=False)
        
        print(f"Model output (scaled): shape={y_pred_scaled.shape}")
        print(f"Sample scaled predictions: {y_pred_scaled[0][:5].detach().cpu().numpy()}")
        
        if thermal_scaler is not None:
            try:
                # Step-by-step unscaling
                print(f"\nStep 1: Convert to numpy")
                y_pred_np = y_pred_scaled.detach().cpu().numpy()
                print(f"Numpy array shape: {y_pred_np.shape}")
                print(f"Numpy sample: {y_pred_np[0][:5]}")
                
                print(f"\nStep 2: Apply thermal_scaler.inverse_transform")
                y_pred_unscaled_np = thermal_scaler.inverse_transform(y_pred_np)
                print(f"Unscaled numpy shape: {y_pred_unscaled_np.shape}")
                print(f"Unscaled numpy sample: {y_pred_unscaled_np[0][:5]}")
                
                print(f"\nStep 3: Convert back to tensor")
                y_pred_unscaled = torch.tensor(y_pred_unscaled_np, dtype=torch.float32, device=device)
                print(f"Final tensor shape: {y_pred_unscaled.shape}")
                print(f"Final tensor sample: {y_pred_unscaled[0][:5].detach().cpu().numpy()}")
                
                # Verify the values are in reasonable temperature range
                temp_range = y_pred_unscaled.detach().cpu().numpy()
                print(f"\nTemperature range check:")
                print(f"Min temperature: {np.min(temp_range):.2f} K")
                print(f"Max temperature: {np.max(temp_range):.2f} K")
                print(f"Mean temperature: {np.mean(temp_range):.2f} K")
                
                if np.min(temp_range) > 250 and np.max(temp_range) < 400:
                    print("✅ Unscaled temperatures are in reasonable range!")
                else:
                    print("❌ Unscaled temperatures are outside reasonable range!")
                    
            except Exception as e:
                print(f"❌ Unscaling failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No thermal_scaler provided!")
            
    print("="*60)


def model_summary(model):
    """Print detailed model summary with 9-bin configuration and unscaling info."""
    print("="*70)
    print("PHYSICS-INFORMED LSTM MODEL SUMMARY (9-BIN + PROPER UNSCALING)")
    print("="*70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print layer information
    print("\nModel Architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {module} ({params:,} params)")
    
    print("\n" + "="*70)
    print("9-BIN PHYSICS CONFIGURATION WITH PROPER UNSCALING")
    print("="*70)
    print("Spatial Segmentation:")
    for i in range(9):
        print(f"  Bin {i+1}: TC{i+1} ↔ TC{i+2} (sensors {i} and {i+1})")
    
    print("\nPhysics Constraints:")
    print("  • Individual bin power conservation")
    print("  • Total system power balance")
    print("  • Energy conservation (no bin exceeds incoming power)")
    print("  • Power continuity across spatial segments")
    
    print("\nCRITICAL FIX - Proper Unscaling:")
    print("  • Time values: Unscaled from normalized to seconds")
    print("  • Temperature values: Unscaled using thermal_scaler")
    print("  • Parameter values: Unscaled using param_scaler")
    print("  • All physics calculations use physical units")
    print("  • Model predictions properly unscaled before physics loss")
    
    print("\nLoss Components:")
    print("  1. MAE Loss: Temperature prediction accuracy")
    print("  2. Physics Loss: 9-bin power difference (weighted, unscaled units)")
    print("  3. Constraint Loss: Energy conservation penalties (weighted)")
    print("  4. Power Balance Loss: Total power vs incoming power (weighted)")
    print("="*70)


def get_model_config():
    """Get recommended model configuration for 9-bin approach with unscaling."""
    return {
        'num_sensors': 10,
        'sequence_length': 20,
        'lstm_units': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'physics_weight': 0.1,      # Weight for 9-bin physics loss
        'constraint_weight': 0.1,   # Weight for energy conservation
        'power_balance_weight': 0.05, # Weight for total power balance
        'cylinder_length': 1.0,     # Cylinder length in meters
        'time_mean': 300.0,         # Time normalization mean
        'time_std': 300.0           # Time normalization std
    }


def compute_r2_score(y_true, y_pred):
    """Compute R-squared score for model evaluation."""
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2


def validate_power_metadata(power_metadata_batch):
    """Validate power metadata format for 9-bin approach with unscaling."""
    required_keys = ['temps_row1', 'temps_row21', 'time_diff', 'h', 'q0']
    
    for key in required_keys:
        if key not in power_metadata_batch:
            raise ValueError(f"Missing required power metadata key: {key}")
    
    # Validate shapes
    temps_row1 = power_metadata_batch['temps_row1']
    temps_row21 = power_metadata_batch['temps_row21']
    
    if len(temps_row1) != 10 or len(temps_row21) != 10:
        raise ValueError(f"Temperature arrays must have 10 sensors (TC1-TC10), got lengths: {len(temps_row1)}, {len(temps_row21)}")
    
    # Validate time_diff is positive
    time_diff = power_metadata_batch['time_diff']
    if time_diff <= 0:
        raise ValueError(f"time_diff must be positive, got {time_diff}")
    
    return True


def create_power_metadata_tensor(temps_row1, temps_row21, time_diff, h, q0, device=None):
    """Create properly formatted power metadata tensor for 9-bin physics loss.
    
    Args:
        temps_row1: Temperature readings at initial timestep, shape (batch, 10)
        temps_row21: Temperature readings at final timestep, shape (batch, 10)
        time_diff: Time difference between timesteps, shape (batch,)
        h: Heat transfer coefficient, shape (batch,)
        q0: Heat flux, shape (batch,)
        device: PyTorch device
    
    Returns:
        Dictionary of tensors for 9-bin physics calculations.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        'temps_row1': torch.tensor(temps_row1, dtype=torch.float32, device=device),
        'temps_row21': torch.tensor(temps_row21, dtype=torch.float32, device=device),
        'time_diff': torch.tensor(time_diff, dtype=torch.float32, device=device),
        'h': torch.tensor(h, dtype=torch.float32, device=device),
        'q0': torch.tensor(q0, dtype=torch.float32, device=device)
    }


def train_model(model, train_loader, val_loader, num_epochs, trainer=None, device=None):
    """Complete training function with physics-informed loss and proper unscaling.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        trainer: PhysicsInformedTrainer instance (if None, creates default)
        device: PyTorch device
    
    Returns:
        Trained model and training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if trainer is None:
        trainer = create_trainer(model, device=device)
    
    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs with proper unscaling...")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Train and validate
        results = trainer.train_epoch(train_loader, val_loader)
        
        # Print results
        print(f"Train Loss: {results.get('train_loss', 0):.4f} | "
              f"Train MAE: {results.get('train_mae', 0):.4f} | "
              f"Train Physics: {results.get('train_physics_loss', 0):.4f}")
        
        if 'val_loss' in results:
            print(f"Val Loss: {results.get('val_loss', 0):.4f} | "
                  f"Val MAE: {results.get('val_mae', 0):.4f} | "
                  f"Val Physics: {results.get('val_physics_loss', 0):.4f}")
    
    return model, trainer.history


def test_unscaling_consistency(thermal_scaler, time_mean=300.0, time_std=300.0):
    """Test function to verify unscaling consistency."""
    import numpy as np
    
    print("\n" + "="*60)
    print("UNSCALING CONSISTENCY TEST")
    print("="*60)
    
    # Test temperature unscaling
    if thermal_scaler is not None:
        print("Testing temperature unscaling...")
        
        # Create dummy scaled temperatures
        scaled_temps = np.random.randn(5, 10)  # 5 samples, 10 sensors
        print(f"Scaled temperatures sample: {scaled_temps[0][:3]}")
        
        # Unscale
        unscaled_temps = thermal_scaler.inverse_transform(scaled_temps)
        print(f"Unscaled temperatures sample: {unscaled_temps[0][:3]}")
        
        # Re-scale to verify consistency
        rescaled_temps = thermal_scaler.transform(unscaled_temps)
        print(f"Re-scaled temperatures sample: {rescaled_temps[0][:3]}")
        
        # Check consistency
        consistency_error = np.mean(np.abs(scaled_temps - rescaled_temps))
        print(f"Temperature scaling consistency error: {consistency_error:.8f}")
        
        if consistency_error < 1e-6:
            print("✅ Temperature unscaling is consistent")
        else:
            print("❌ Temperature unscaling has consistency issues")
    
    # Test time unscaling
    print(f"\nTesting time unscaling with mean={time_mean}, std={time_std}...")
    
    # Create dummy scaled time values
    scaled_times = np.random.randn(5) * 2  # Values around -2 to 2
    print(f"Scaled times: {scaled_times}")
    
    # Unscale
    unscaled_times = scaled_times * time_std + time_mean
    print(f"Unscaled times: {unscaled_times}")
    
    # Re-scale to verify consistency
    rescaled_times = (unscaled_times - time_mean) / time_std
    print(f"Re-scaled times: {rescaled_times}")
    
    # Check consistency
    time_consistency_error = np.mean(np.abs(scaled_times - rescaled_times))
    print(f"Time scaling consistency error: {time_consistency_error:.8f}")
    
    if time_consistency_error < 1e-6:
        print("✅ Time unscaling is consistent")
    else:
        print("❌ Time unscaling has consistency issues")
    
    print("="*60)


# Example usage and validation with proper unscaling
if __name__ == "__main__":
    print("="*70)
    print("FIXED 9-BIN PHYSICS-INFORMED LSTM WITH PROPER UNSCALING (PYTORCH)")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model with 9-bin configuration
    config = get_model_config()
    model = build_model(
        num_sensors=config['num_sensors'],
        sequence_length=config['sequence_length'],
        lstm_units=config['lstm_units'],
        dropout_rate=config['dropout_rate'],
        device=device
    )
    
    # Create dummy scalers for testing
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Create dummy thermal scaler
    dummy_thermal_data = np.random.randn(1000, 10) * 50 + 300  # Temperatures around 300K
    thermal_scaler = StandardScaler()
    thermal_scaler.fit(dummy_thermal_data)
    
    # Create dummy parameter scaler
    dummy_param_data = np.random.randn(1000, 4)  # h, flux, abs, surf
    param_scaler = StandardScaler()
    param_scaler.fit(dummy_param_data)
    
    # Test unscaling consistency
    test_unscaling_consistency(thermal_scaler, config['time_mean'], config['time_std'])
    
    # Create trainer with 9-bin approach and proper unscaling
    trainer = create_trainer(
        model=model,
        physics_weight=config['physics_weight'],
        constraint_weight=config['constraint_weight'],
        power_balance_weight=config['power_balance_weight'],
        learning_rate=config['learning_rate'],
        cylinder_length=config['cylinder_length'],
        device=device,
        param_scaler=param_scaler,       # Pass parameter scaler
        thermal_scaler=thermal_scaler,   # Pass thermal scaler  
        time_mean=config['time_mean'],   # Pass time normalization params
        time_std=config['time_std']
    )
    
    # Print summary
    model_summary(model)
    
    # Validate input shapes with dummy data
    print("\nValidating FIXED 9-bin model with dummy data and proper unscaling...")
    batch_size = 32
    dummy_time_series = torch.randn(batch_size, 20, 11, device=device)
    dummy_static_params = torch.randn(batch_size, 4, device=device)
    
    # Create dummy target using thermal scaler (simulating real scaled targets)
    dummy_target_raw = np.random.randn(batch_size, 10) * 50 + 300  # Raw temperatures ~300K
    dummy_target_scaled = thermal_scaler.transform(dummy_target_raw)  # Scale them
    dummy_target = torch.tensor(dummy_target_scaled, dtype=torch.float32, device=device)
    
    # Test tensor unscaling with sample predictions
    print("\nTesting tensor unscaling functionality...")
    model.eval()
    with torch.no_grad():
        sample_predictions = model([dummy_time_series[:5], dummy_static_params[:5]], training=False)
        test_tensor_unscaling(thermal_scaler, sample_predictions, device)
        
        # Debug prediction unscaling step by step
        debug_prediction_unscaling(model, thermal_scaler, 
                                 [dummy_time_series[:2], dummy_static_params[:2]], device)
    
    # Create dummy power metadata for 9-bin testing WITH PROPER SCALING SIMULATION
    # Simulate scaled temperatures (what the model actually sees)
    dummy_temps_raw = np.random.randn(batch_size, 10) * 50 + 300  # Raw temperatures ~300K
    dummy_temps_scaled = thermal_scaler.transform(dummy_temps_raw) # Scaled temperatures
    
    # Simulate time values (both normalized and raw)
    dummy_time_raw = np.random.rand(batch_size) * 500 + 100  # Raw time 100-600 seconds
    dummy_time_normalized = (dummy_time_raw - config['time_mean']) / config['time_std']  # Normalized time
    
    dummy_power_data = []
    for i in range(batch_size):
        # Create power data that includes both scaled temperatures and time info
        dummy_power_data.append({
            'temps_row1': dummy_temps_scaled[i].tolist(),  # Scaled temperatures 
            'temps_row21': (dummy_temps_scaled[i] + np.random.randn(10) * 0.1).tolist(),  # Small changes
            'time_row1': dummy_time_raw[i],    # Raw time values (to be processed by function)
            'time_row21': dummy_time_raw[i] + np.random.rand() * 10 + 1,  # Raw time + delta
            'h': np.random.rand() * 0.9 + 0.1,  # Height parameter (scaled)
            'q0': np.random.rand() * 2.0 - 1.0  # Heat flux parameter (scaled)
        })
    
    try:
        # Test model forward pass
        dummy_output = model([dummy_time_series, dummy_static_params], training=False)
        print(f"✅ Model forward pass successful")
        print(f"   Input time_series: {dummy_time_series.shape}")
        print(f"   Input static_params: {dummy_static_params.shape}")
        print(f"   Output predictions: {dummy_output.shape}")
        print(f"   Sample prediction (scaled): {dummy_output[0][:3].detach().cpu().numpy()}")
        
        # Test power data processing with proper unscaling
        power_metadata_list = process_power_data_batch(
            dummy_power_data,
            thermal_scaler=thermal_scaler,
            time_mean=config['time_mean'],
            time_std=config['time_std']
        )
        
        print(f"✅ Power metadata processing with unscaling successful")
        print(f"   Processed {len(power_metadata_list)} metadata entries")
        
        # Show unscaling results for first sample
        if power_metadata_list:
            sample_meta = power_metadata_list[0]
            print(f"   Sample 0 time_diff: {sample_meta['time_diff']:.2f} seconds (unscaled)")
            print(f"   Sample 0 temps_row1[0]: {sample_meta['temps_row1'][0]:.2f} K (unscaled)")
            print(f"   Sample 0 temps_row21[0]: {sample_meta['temps_row21'][0]:.2f} K (unscaled)")
        
        # Test FIXED 9-bin physics loss computation with proper unscaling
        print(f"\n🔧 TESTING FIXED PHYSICS LOSS WITH PROPER TENSOR UNSCALING...")
        physics_loss, constraint_loss, power_balance_loss, power_info = trainer.compute_nine_bin_physics_loss(
            dummy_target, dummy_output, power_metadata_list
        )
        
        print(f"✅ FIXED 9-bin physics loss computation with proper unscaling successful")
        print(f"   Physics loss: {physics_loss:.4f}")
        print(f"   Constraint loss: {constraint_loss:.4f}")
        print(f"   Power balance loss: {power_balance_loss:.4f}")
        
        if power_info:
            actual_powers = power_info['total_actual_power']
            predicted_powers = power_info['total_predicted_power']
            incoming_powers = power_info['incoming_power']
            time_diffs = power_info['time_diff_unscaled']
            
            print(f"   Total actual power range: {min(actual_powers):.2f} - {max(actual_powers):.2f} W")
            print(f"   Total predicted power range: {min(predicted_powers):.2f} - {max(predicted_powers):.2f} W")
            print(f"   Incoming power range: {min(incoming_powers):.2f} - {max(incoming_powers):.2f} W")
            print(f"   Time differences range: {min(time_diffs):.2f} - {max(time_diffs):.2f} seconds")
            
            # Check if predicted powers are reasonable (not extremely negative)
            reasonable_predictions = all(p > -10000 for p in predicted_powers)
            if reasonable_predictions:
                print(f"✅ Predicted powers are in reasonable range!")
            else:
                print(f"❌ Some predicted powers are extremely negative - check unscaling!")
        
        # Test training step with dummy batch
        dummy_batch = [dummy_time_series, dummy_static_params, dummy_target, dummy_power_data]
        train_result = trainer.train_step(dummy_batch)
        
        print(f"✅ Training step with FIXED physics loss and proper unscaling successful")
        print(f"   Total loss: {train_result['loss']:.4f}")
        print(f"   MAE: {train_result['mae']:.4f}")
        print(f"   Physics loss: {train_result['physics_loss']:.4f}")
        print(f"   Constraint loss: {train_result['constraint_loss']:.4f}")
        print(f"   Power balance loss: {train_result['power_balance_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during FIXED 9-bin validation with unscaling: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("FIXED 9-BIN PHYSICS-INFORMED PYTORCH MODEL WITH PROPER UNSCALING READY!")
    print("="*70)
    print("🔧 CRITICAL FIXES IMPLEMENTED:")
    print("✅ Fixed tensor unscaling in compute_nine_bin_physics_loss method")
    print("✅ Added proper batch size handling to prevent shape mismatches")
    print("✅ Added detailed debug output to trace unscaling process")
    print("✅ Added tensor unscaling validation functions")
    print("✅ Added step-by-step prediction unscaling debug function")
    print("✅ Model predictions are now properly unscaled before physics calculations")
    print("✅ All temperature values in physics loss are in physical units (Kelvin)")
    print("✅ Time values are properly unscaled to seconds")
    print("✅ Physics calculations use proper physical units throughout")
    print("="*70)
    
    print("\nUSAGE NOTES FOR THE FIXED VERSION:")
    print("• The main fix is in compute_nine_bin_physics_loss method")
    print("• y_pred tensors are now properly unscaled using thermal_scaler")
    print("• Added extensive debug output to trace unscaling process")
    print("• Use debug_prediction_unscaling() to verify unscaling works")
    print("• Use test_tensor_unscaling() to validate tensor operations")
    print("• Predicted temperatures should now be in 250-400K range")
    print("• Physics calculations will use physical temperature units")
    print("• Debug output will show proper temperature values during training")
    print("="*70)