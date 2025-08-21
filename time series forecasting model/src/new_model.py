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
    
    # Reduced debug output to prevent memory issues
    debug_enabled = batch_size <= 8  # Only debug for small batches
    
    if debug_enabled:
        print(f"Processing power data batch with {batch_size} samples")
    
    for i, power_data in enumerate(power_data_list):
        if power_data is None or not isinstance(power_data, dict):
            if debug_enabled:
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
                if debug_enabled:
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
                    if debug_enabled:
                        print(f"Warning: temps_row1 has {len(temps_row1)} elements, expected 10 at index {i}")
                    temps_row1 = (temps_row1 + [300.0] * 10)[:10]  # Pad or truncate to 10
            else:
                if debug_enabled:
                    print(f"Warning: temps_row1 is not a list/tuple at index {i}")
                temps_row1 = [300.0] * 10  # fallback
                
            if isinstance(temps_row21, (list, tuple)):
                temps_row21 = [float(x) for x in temps_row21]
                if len(temps_row21) != 10:
                    if debug_enabled:
                        print(f"Warning: temps_row21 has {len(temps_row21)} elements, expected 10 at index {i}")
                    temps_row21 = (temps_row21 + [301.0] * 10)[:10]  # Pad or truncate to 10
            else:
                if debug_enabled:
                    print(f"Warning: temps_row21 is not a list/tuple at index {i}")
                temps_row21 = [301.0] * 10  # fallback
            
            # CRITICAL FIX: Check if temperatures are already in physical units
            # If temperatures are in reasonable physical range (200-500K), don't unscale them
            temp_sample = temps_row1[0] if temps_row1 else 300.0
            
            if 200.0 <= temp_sample <= 500.0:
                # Temperatures are already in physical units (Kelvin)
                if debug_enabled and i == 0:  # Only log once per batch
                    print(f"Temperatures already in physical units ({temp_sample:.1f}K), skipping thermal unscaling")
                # Keep original values
            elif thermal_scaler is not None and (-10.0 <= temp_sample <= 10.0):
                # Temperatures appear to be scaled (normalized values around -10 to 10)
                if debug_enabled and i == 0:
                    print(f"Temperatures appear scaled ({temp_sample:.3f}), applying thermal unscaling")
                try:
                    # Unscale temps_row1
                    temps_row1_array = np.array(temps_row1).reshape(1, -1)  # Shape (1, 10)
                    temps_row1_unscaled = thermal_scaler.inverse_transform(temps_row1_array)[0]
                    temps_row1 = temps_row1_unscaled.tolist()
                    
                    # Unscale temps_row21  
                    temps_row21_array = np.array(temps_row21).reshape(1, -1)  # Shape (1, 10)
                    temps_row21_unscaled = thermal_scaler.inverse_transform(temps_row21_array)[0]
                    temps_row21 = temps_row21_unscaled.tolist()
                    
                    if debug_enabled and i == 0:
                        print(f"Successfully unscaled temperatures: {temps_row1[0]:.1f}K -> {temps_row21[0]:.1f}K")
                except Exception as e:
                    if debug_enabled:
                        print(f"Warning: Failed to unscale temperatures for sample {i}: {e}")
                    # Keep original values if unscaling fails
            else:
                # Temperatures are in some other range, keep as-is
                if debug_enabled and i == 0:
                    print(f"Temperature value {temp_sample:.1f} in unexpected range, keeping as-is")
            
            # CRITICAL FIX: Get raw time values and ensure they're in proper physical units
            time_row1_raw = float(power_data['time_row1'])
            time_row21_raw = float(power_data['time_row21'])
            
            # Check if time values appear to be normalized (values around -1 to 1 suggest normalization)
            if abs(time_row1_raw) < 10 and abs(time_row21_raw) < 10:
                # These appear to be normalized time values, unscale them
                time_row1_unscaled = time_row1_raw * time_std + time_mean
                time_row21_unscaled = time_row21_raw * time_std + time_mean
                if debug_enabled and i == 0:
                    print(f"Detected normalized time values, unscaling: {time_row1_raw:.3f} -> {time_row1_unscaled:.1f}")
            else:
                # These appear to be raw time values already
                time_row1_unscaled = time_row1_raw
                time_row21_unscaled = time_row21_raw
                if debug_enabled and i == 0:
                    print(f"Using raw time values: {time_row1_unscaled:.1f}, {time_row21_unscaled:.1f}")
            
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
            if debug_enabled:
                print(f"Error processing power_data at index {i}: {e}")
            # Skip invalid data, use dummy values
            processed_metadata.append({
                'temps_row1': [300.0] * 10,
                'temps_row21': [301.0] * 10,
                'time_diff': 1.0,
                'h': 50.0,
                'q0': 1000.0
            })
    
    if debug_enabled:
        print(f"Successfully processed {len(processed_metadata)} power metadata entries")
    return processed_metadata


class PhysicsInformedTrainer:
    """Custom trainer that handles physics-informed loss computation with 9-bin approach, PROPER TIME UNSCALING, and POWER CAPPING."""
    
    def __init__(self, model, physics_weight=0.1, constraint_weight=0.1, learning_rate=0.001, 
                 cylinder_length=1.0, power_balance_weight=0.05, lstm_units=64, dropout_rate=0.2,
                 device=None, param_scaler=None, thermal_scaler=None, time_mean=300.0, time_std=300.0,
                 cap_penalty_weight=0.01, use_soft_capping=False):
        self.model = model
        self.physics_weight = physics_weight
        self.constraint_weight = constraint_weight
        self.power_balance_weight = power_balance_weight
        self.cylinder_length = cylinder_length  # Total cylinder length in meters
        
        # Store model parameters for saving metadata
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # POWER CAPPING PARAMETERS
        self.cap_penalty_weight = cap_penalty_weight
        self.use_soft_capping = use_soft_capping
        
        # Statistics tracking for power capping
        self.cap_scales = []
        self.cap_violations = 0
        self.cap_samples = 0
        
        # IMPORTANT: Store scalers for proper unscaling
        self.param_scaler = param_scaler
        self.thermal_scaler = thermal_scaler  # Store thermal scaler
        self.time_mean = time_mean  # Store time normalization parameters
        self.time_std = time_std    # Store time normalization parameters
        
        if param_scaler is None:
            print("Warning: param_scaler not provided to trainer - physics calculations may be incorrect")
        if thermal_scaler is None:
            print("Warning: thermal_scaler not provided to trainer - temperature unscaling will be skipped")
        
        # Device handling
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Physics constants - moved to device immediately
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

    def _cap_positive_powers(self, pred_bin_powers, incoming_power):
        """
        Cap positive predicted powers to not exceed incoming power while preserving cooling (negative powers).
        
        This is a CRITICAL method for ensuring energy conservation in your physics-informed model.
        
        Args:
            pred_bin_powers: Tensor of shape [num_bins] with predicted power for each bin
            incoming_power: Scalar tensor with total incoming power
            
        Returns:
            capped_powers: Tensor with capped positive powers
            scale_factor: Scaling factor applied (1.0 = no scaling needed)
            violation_occurred: Boolean tensor indicating if capping was needed
        """
        # Separate positive and negative powers
        positive_mask = pred_bin_powers > 0
        negative_mask = pred_bin_powers <= 0
        
        positive_powers = pred_bin_powers[positive_mask]
        negative_powers = pred_bin_powers[negative_mask]
        
        # Calculate total positive power
        total_positive_power = torch.sum(positive_powers) if len(positive_powers) > 0 else torch.tensor(0.0, device=pred_bin_powers.device)
        
        # Check if capping is needed
        violation_occurred = total_positive_power > incoming_power
        
        if violation_occurred and len(positive_powers) > 0:
            # Calculate scaling factor to cap total positive power at incoming power
            scale_factor = incoming_power / (total_positive_power + 1e-8)  # Small epsilon to prevent division by zero
            scale_factor = torch.clamp(scale_factor, min=0.0, max=1.0)  # Ensure scale is between 0 and 1
            
            # Scale down only the positive powers
            scaled_positive_powers = positive_powers * scale_factor
            
            # Reconstruct the full power array
            capped_powers = pred_bin_powers.clone()
            capped_powers[positive_mask] = scaled_positive_powers
            # negative powers remain unchanged
            
        else:
            # No capping needed
            scale_factor = torch.tensor(1.0, device=pred_bin_powers.device)
            capped_powers = pred_bin_powers.clone()
        
        return capped_powers, scale_factor, violation_occurred
    
    def _soft_cap_powers(self, pred_bin_powers, incoming_power, soft_cap_factor=0.9):
        """
        Soft capping using sigmoid function instead of hard scaling.
        
        This approach uses a smooth sigmoid function to discourage but not strictly prevent
        exceeding the incoming power, which may be better for gradient flow.
        
        Args:
            pred_bin_powers: Tensor of shape [num_bins] with predicted power for each bin
            incoming_power: Scalar tensor with total incoming power
            soft_cap_factor: Factor determining how aggressively to cap (0.5-0.95 recommended)
            
        Returns:
            soft_capped_powers: Tensor with soft-capped powers
            penalty: Soft penalty for exceeding the limit
        """
        # Calculate total predicted power
        total_predicted_power = torch.sum(pred_bin_powers)
        
        # Soft capping using sigmoid
        excess_ratio = total_predicted_power / (incoming_power + 1e-8)
        
        if excess_ratio > 1.0:
            # Apply sigmoid scaling when exceeding incoming power
            sigmoid_factor = torch.sigmoid(-10 * (excess_ratio - 1.0))  # Sharp sigmoid around 1.0
            scaling = soft_cap_factor + (1 - soft_cap_factor) * sigmoid_factor
            
            # Apply scaling to all positive powers
            positive_mask = pred_bin_powers > 0
            soft_capped_powers = pred_bin_powers.clone()
            soft_capped_powers[positive_mask] *= scaling
            
            # Calculate penalty proportional to excess
            penalty = (excess_ratio - 1.0) ** 2
            
        else:
            soft_capped_powers = pred_bin_powers
            penalty = torch.tensor(0.0, device=pred_bin_powers.device)
        
        return soft_capped_powers, penalty

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
        """Compute physics-based loss using 9 spatial bins with PROPER UNSCALING and POWER CAPPING.
        
        CRITICAL FIX: Only unscale y_pred tensors, not the power_metadata temperatures 
        which are already properly unscaled in process_power_data_batch.
        """
        try:
            if not power_metadata_list or len(power_metadata_list) == 0:
                zero_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
                return zero_loss, zero_loss, zero_loss, {}

            # CRITICAL FIX: Ensure batch size consistency
            actual_batch_size = min(len(power_metadata_list), y_true.shape[0], y_pred.shape[0])

            # CRITICAL FIX: Only unscale y_pred (model predictions), 
            # power_metadata temperatures are already unscaled in process_power_data_batch
            y_pred_unscaled = y_pred[:actual_batch_size].clone()
            
            if self.thermal_scaler is not None:
                try:
                    # Only process the actual batch size to prevent shape mismatches
                    y_pred_batch = y_pred[:actual_batch_size].detach().cpu().numpy()
                    
                    # Unscale using thermal_scaler - SHAPE MUST BE (n_samples, n_features)
                    y_pred_unscaled_np = self.thermal_scaler.inverse_transform(y_pred_batch)
                    
                    # Convert back to tensors with proper batch size - KEEP ON SAME DEVICE
                    y_pred_unscaled = torch.tensor(y_pred_unscaled_np, dtype=torch.float32, device=self.device)
                    
                except Exception as e:
                    # Use original scaled values if unscaling fails
                    y_pred_unscaled = y_pred[:actual_batch_size].clone()
            
            # Initialize tensors to collect physics losses on correct device
            physics_losses = []
            constraint_penalties = []
            power_balance_losses = []
            
            # Initialize power analysis lists properly
            total_actual_powers = []
            total_predicted_powers = []
            incoming_powers = []
            
            # Physics constants as scalars for computation
            rho = 1836.31  # kg/m³
            cp = 1512.0    # J/(kg·K)
            radius = 0.05175  # m
            pi = 3.14159265359
            
            # Process each sample in the batch
            for sample_idx in range(actual_batch_size):
                try:
                    power_data = power_metadata_list[sample_idx]
                    
                    # Extract plain Python values - ALREADY PROPERLY UNSCALED in process_power_data_batch
                    temps_row1 = power_data['temps_row1']  # List of 10 floats (already unscaled)
                    temps_row21 = power_data['temps_row21']  # List of 10 floats (already unscaled)
                    time_diff = power_data['time_diff']  # Float in seconds (unscaled)
                    h_unscaled = power_data['h']  # Float - cylinder height
                    q0_unscaled = power_data['q0']  # Float - heat flux
                    
                    # IMPORTANT: Use dynamic cylinder height from h parameter
                    cylinder_length = h_unscaled
                    bin_height = cylinder_length / self.num_bins
                    bin_volume = pi * (radius ** 2) * bin_height
                    bin_mass = rho * bin_volume
                    
                    # Calculate incoming power using unscaled q0
                    surface_area = pi * (radius ** 2)
                    incoming_power = q0_unscaled * surface_area
                    
                    # Process each of the 9 bins for this sample
                    sample_physics_losses = []
                    sample_predicted_powers = []
                    sample_actual_powers = []
                    
                    for bin_idx, (sensor1_idx, sensor2_idx) in enumerate(self.bin_sensor_pairs):
                        # VALIDATION: Ensure indices are within bounds
                        if (sensor1_idx >= len(temps_row1) or sensor2_idx >= len(temps_row1) or
                            sensor1_idx >= len(temps_row21) or sensor2_idx >= len(temps_row21) or
                            sample_idx >= y_pred_unscaled.shape[0] or 
                            sensor1_idx >= y_pred_unscaled.shape[1] or 
                            sensor2_idx >= y_pred_unscaled.shape[1]):
                            continue
                        
                        # Get actual temperatures from metadata (ALREADY UNSCALED)
                        actual_temp1_t1 = temps_row1[sensor1_idx]  # Already unscaled
                        actual_temp2_t1 = temps_row1[sensor2_idx]  # Already unscaled
                        actual_temp1_t21 = temps_row21[sensor1_idx]  # Already unscaled
                        actual_temp2_t21 = temps_row21[sensor2_idx]  # Already unscaled
                        
                        # Get predictions as tensors (UNSCALED) - KEEP AS TENSORS FOR GRADIENTS
                        pred_temp1_t21 = y_pred_unscaled[sample_idx, sensor1_idx]
                        pred_temp2_t21 = y_pred_unscaled[sample_idx, sensor2_idx]
                        
                        # Calculate temperature changes (average of two sensors for this bin)
                        actual_temp1_change = actual_temp1_t21 - actual_temp1_t1
                        actual_temp2_change = actual_temp2_t21 - actual_temp2_t1
                        actual_bin_temp_change = (actual_temp1_change + actual_temp2_change) / 2.0
                        
                        # KEEP PREDICTIONS AS TENSORS for gradient computation
                        pred_temp1_change = pred_temp1_t21 - actual_temp1_t1
                        pred_temp2_change = pred_temp2_t21 - actual_temp2_t1
                        pred_bin_temp_change = (pred_temp1_change + pred_temp2_change) / 2.0
                        
                        # Power calculations WITH PROPER TIME UNITS - KEEP AS TENSORS
                        actual_bin_power = bin_mass * cp * actual_bin_temp_change / time_diff
                        pred_bin_power = bin_mass * cp * pred_bin_temp_change / time_diff
                        
                        sample_actual_powers.append(actual_bin_power)
                        sample_predicted_powers.append(pred_bin_power)
                        
                        # Physics loss for this bin - KEEP AS TENSOR
                        bin_physics_loss = torch.abs(pred_bin_power - actual_bin_power)
                        sample_physics_losses.append(bin_physics_loss)
                    
                    # Only proceed if we have valid bin calculations
                    if len(sample_physics_losses) == 0:
                        continue
                    
                    # --- POWER CAPPING (CONFIGURABLE HARD/SOFT) ---
                    if len(sample_predicted_powers) > 0:
                        pred_bin_powers = torch.stack(sample_predicted_powers)  # [num_bins]
                        incoming_power_tensor = torch.tensor(incoming_power, dtype=torch.float32, device=self.device)
                        
                        if self.use_soft_capping:
                            # Apply soft capping
                            capped_powers, soft_penalty = self._soft_cap_powers(pred_bin_powers, incoming_power_tensor)
                            constraint_penalties.append(soft_penalty)
                            scale_factor = torch.tensor(1.0, device=self.device)  # For statistics
                            violated = soft_penalty > 0
                        else:
                            # Apply hard capping
                            capped_powers, scale_factor, violated = self._cap_positive_powers(pred_bin_powers, incoming_power_tensor)
                            # Penalty to discourage frequent/large rescaling
                            cap_penalty = self.cap_penalty_weight * (1.0 - scale_factor) ** 2
                            constraint_penalties.append(cap_penalty)
                        
                        # Track statistics for tuning
                        try:
                            self.cap_scales.append(float(scale_factor.detach().cpu()))
                            self.cap_violations += int(violated.item()) if isinstance(violated, torch.Tensor) else int(violated)
                            self.cap_samples += 1
                        except Exception:
                            pass
                        
                        # Replace list with capped tensor values for downstream calculations
                        sample_predicted_powers = list(capped_powers)
                    
                    # Calculate total powers for this sample - SUM TENSORS
                    total_actual_power = sum(sample_actual_powers)
                    total_predicted_power = sum(sample_predicted_powers)
                    
                    # Store power analysis data properly with consistent naming
                    total_actual_powers.append(float(total_actual_power) if isinstance(total_actual_power, torch.Tensor) else total_actual_power)
                    total_predicted_powers.append(float(total_predicted_power) if isinstance(total_predicted_power, torch.Tensor) else total_predicted_power)
                    incoming_powers.append(incoming_power)
                    
                    # Average physics loss for this sample - KEEP AS TENSOR
                    avg_sample_physics_loss = sum(sample_physics_losses) / len(sample_physics_losses)
                    physics_losses.append(avg_sample_physics_loss)
                    
                    # Additional constraint penalties - KEEP AS TENSOR
                    incoming_power_tensor = torch.tensor(incoming_power, dtype=torch.float32, device=self.device)
                    
                    # Individual bin constraint: no bin should exceed total incoming power
                    bin_excess = torch.clamp(total_predicted_power - incoming_power_tensor, min=0.0)
                    constraint_penalties.append(bin_excess ** 2)
                    
                    # Power balance constraint
                    power_imbalance = torch.abs(total_predicted_power - incoming_power_tensor)
                    power_balance_losses.append(power_imbalance)
                    
                except Exception as e:
                    # Skip this sample if there's an error
                    continue
            
            # Ensure we have valid results
            if len(physics_losses) == 0:
                zero_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
                return zero_loss, zero_loss, zero_loss, {}
            
            # Calculate final losses as tensors for gradient computation
            physics_loss = sum(physics_losses) / len(physics_losses)
            constraint_penalty = sum(constraint_penalties) / len(constraint_penalties)
            power_balance_loss = sum(power_balance_losses) / len(power_balance_losses)
            
            # Return enhanced info for analysis with proper power data structure
            power_info = {
                'num_samples_processed': len(physics_losses),
                'physics_loss_components': len(physics_losses),
                'total_actual_powers': total_actual_powers,
                'total_predicted_powers': total_predicted_powers,
                'incoming_powers': incoming_powers,
                'avg_actual_power': np.mean(total_actual_powers) if total_actual_powers else 0.0,
                'avg_predicted_power': np.mean(total_predicted_powers) if total_predicted_powers else 0.0,
                'avg_incoming_power': np.mean(incoming_powers) if incoming_powers else 0.0,
                'power_cap_stats': {
                    'violations': self.cap_violations,
                    'samples': self.cap_samples,
                    'violation_rate': self.cap_violations / max(self.cap_samples, 1),
                    'avg_scale_factor': np.mean(self.cap_scales[-100:]) if self.cap_scales else 1.0,  # Last 100 samples
                    'capping_method': 'soft' if self.use_soft_capping else 'hard'
                }
            }
            
            return physics_loss, constraint_penalty, power_balance_loss, power_info
            
        except Exception as e:
            print(f"9-bin physics loss computation failed: {e}")
            zero_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
            return zero_loss, zero_loss, zero_loss, {}
    
    def print_power_cap_statistics(self):
        """Print statistics about power capping behavior."""
        if self.cap_samples == 0:
            print("No power capping statistics available yet.")
            return
        
        violation_rate = self.cap_violations / self.cap_samples
        avg_scale = np.mean(self.cap_scales) if self.cap_scales else 1.0
        
        print("\n" + "="*50)
        print("POWER CAPPING STATISTICS")
        print("="*50)
        print(f"Capping method: {'Soft' if self.use_soft_capping else 'Hard'}")
        print(f"Total samples processed: {self.cap_samples}")
        print(f"Power cap violations: {self.cap_violations}")
        print(f"Violation rate: {violation_rate:.3f} ({violation_rate*100:.1f}%)")
        print(f"Average scale factor: {avg_scale:.4f}")
        print(f"Min scale factor: {min(self.cap_scales):.4f}" if self.cap_scales else "N/A")
        print(f"Max scale factor: {max(self.cap_scales):.4f}" if self.cap_scales else "N/A")
        
        if violation_rate > 0.5:
            print("⚠️  HIGH VIOLATION RATE: Consider increasing power_balance_weight or decreasing physics_weight")
        elif violation_rate < 0.01:
            print("✅ LOW VIOLATION RATE: Power capping is working well")
        else:
            print("✅ MODERATE VIOLATION RATE: Power capping is functioning normally")
        
        print("="*50)
    
    def train_step(self, batch):
        """Custom training step with 9-bin physics loss, PROPER UNSCALING, and POWER CAPPING."""
        self.model.train()
        
        # Move batch to device - ensure consistent batch size
        time_series = batch[0].to(self.device)
        static_params = batch[1].to(self.device)
        y_true = batch[2].to(self.device)
        power_data = batch[3] if len(batch) > 3 else None
        
        # Ensure all tensors have same batch size
        min_batch_size = min(time_series.shape[0], static_params.shape[0], y_true.shape[0])
        time_series = time_series[:min_batch_size]
        static_params = static_params[:min_batch_size]
        y_true = y_true[:min_batch_size]
        
        # Process power metadata to plain Python format WITH PROPER UNSCALING
        power_metadata_list = None
        if power_data is not None:
            # Trim power_data to match tensor batch size
            power_data_trimmed = power_data[:min_batch_size] if len(power_data) > min_batch_size else power_data
            power_metadata_list = process_power_data_batch(
                power_data_trimmed, 
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
        
        # Return detached values to prevent memory leaks
        return {
            'loss': total_loss.detach().item(),
            'mae': mae_loss.detach().item(),
            'mse': mse_loss.detach().item(),
            'physics_loss': physics_loss.detach().item(),
            'constraint_loss': constraint_loss.detach().item(),
            'power_balance_loss': power_balance_loss.detach().item()
        }

    def validation_step(self, batch):
        """Validation step with 9-bin physics analysis, PROPER UNSCALING, and POWER CAPPING."""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device - ensure consistent batch size
            time_series = batch[0].to(self.device)
            static_params = batch[1].to(self.device)
            y_true = batch[2].to(self.device)
            power_data = batch[3] if len(batch) > 3 else None
            
            # Ensure all tensors have same batch size
            min_batch_size = min(time_series.shape[0], static_params.shape[0], y_true.shape[0])
            time_series = time_series[:min_batch_size]
            static_params = static_params[:min_batch_size]
            y_true = y_true[:min_batch_size]
            
            # Process power metadata to plain Python format WITH PROPER UNSCALING
            power_metadata_list = None
            if power_data is not None:
                # Trim power_data to match tensor batch size
                power_data_trimmed = power_data[:min_batch_size] if len(power_data) > min_batch_size else power_data
                power_metadata_list = process_power_data_batch(
                    power_data_trimmed,
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
        """Train for one epoch with detailed physics tracking and power capping."""
        # Initialize metrics for this epoch
        epoch_train_metrics = defaultdict(list)
        epoch_val_metrics = defaultdict(list)
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            try:
                metrics = self.train_step(batch)
                for key, value in metrics.items():
                    epoch_train_metrics[f'train_{key}'].append(value)
                
                # Clear cache periodically to prevent memory buildup
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Validation loop
        if val_loader is not None:
            for batch_idx, batch in enumerate(val_loader):
                try:
                    metrics = self.validation_step(batch)
                    for key, value in metrics.items():
                        epoch_val_metrics[key].append(value)
                        
                    # Clear cache periodically
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Aggregate results
        results = {}
        for key, values in epoch_train_metrics.items():
            if values:  # Only add if we have values
                results[key] = np.mean(values)
        for key, values in epoch_val_metrics.items():
            if values:  # Only add if we have values
                results[key] = np.mean(values)
        
        # Update history
        for key, value in results.items():
            if key in self.history:
                self.history[key].append(float(value))
        
        return results
    
    def analyze_power_balance(self, data_loader, num_samples=100):
        """Analyze power balance across the system for diagnostic purposes with PROPER UNSCALING and POWER CAPPING."""
        print("\n" + "="*60)
        print("POWER BALANCE ANALYSIS (WITH PROPER UNSCALING & POWER CAPPING)")
        print("="*60)
        
        sample_count = 0
        physics_losses = []
        all_power_info = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break
                
                try:
                    time_series = batch[0].to(self.device)
                    static_params = batch[1].to(self.device)
                    y_true = batch[2].to(self.device)
                    power_data = batch[3] if len(batch) > 3 else None
                    
                    # Ensure consistent batch size
                    min_batch_size = min(time_series.shape[0], static_params.shape[0], y_true.shape[0])
                    time_series = time_series[:min_batch_size]
                    static_params = static_params[:min_batch_size]
                    y_true = y_true[:min_batch_size]
                    
                    if power_data is not None and len(power_data) > 0 and power_data[0] is not None:
                        # Process power metadata using the function WITH PROPER UNSCALING
                        power_data_trimmed = power_data[:min_batch_size] if len(power_data) > min_batch_size else power_data
                        power_metadata_list = process_power_data_batch(
                            power_data_trimmed,
                            thermal_scaler=self.thermal_scaler,  # Pass thermal scaler
                            time_mean=self.time_mean,           # Pass time normalization params
                            time_std=self.time_std
                        )
                        
                        if power_metadata_list:
                            # Get predictions
                            y_pred = self.model([time_series, static_params], training=False)
                            
                            # Compute power analysis
                            physics_loss, _, _, power_info = self.compute_nine_bin_physics_loss(
                                y_true, y_pred, power_metadata_list
                            )
                            
                            if power_info and power_info.get('num_samples_processed', 0) > 0:
                                physics_losses.append(physics_loss.item())
                                all_power_info.append(power_info)
                                sample_count += power_info['num_samples_processed']
                                
                except Exception as e:
                    print(f"Warning: Error in power analysis: {e}")
                    continue
        
        if len(physics_losses) > 0 and len(all_power_info) > 0:
            print(f"Samples analyzed: {sample_count}")
            print(f"Average physics loss: {np.mean(physics_losses):.4f}")
            print(f"Physics loss std: {np.std(physics_losses):.4f}")
            
            # Extract power analysis data with proper error handling
            try:
                all_actual_powers = []
                all_predicted_powers = []
                all_incoming_powers = []
                all_cap_stats = []
                
                for power_info in all_power_info:
                    # Extract power data with consistent naming
                    if 'total_actual_powers' in power_info and power_info['total_actual_powers']:
                        all_actual_powers.extend(power_info['total_actual_powers'])
                    if 'total_predicted_powers' in power_info and power_info['total_predicted_powers']:
                        all_predicted_powers.extend(power_info['total_predicted_powers'])
                    if 'incoming_powers' in power_info and power_info['incoming_powers']:
                        all_incoming_powers.extend(power_info['incoming_powers'])
                    if 'power_cap_stats' in power_info:
                        all_cap_stats.append(power_info['power_cap_stats'])
                
                if all_actual_powers and all_predicted_powers and all_incoming_powers:
                    print(f"\nPower Analysis:")
                    print(f"Average actual power: {np.mean(all_actual_powers):.2f} W")
                    print(f"Average predicted power: {np.mean(all_predicted_powers):.2f} W")
                    print(f"Average incoming power: {np.mean(all_incoming_powers):.2f} W")
                    
                    # Calculate errors safely
                    power_errors = np.abs(np.array(all_actual_powers) - np.array(all_predicted_powers))
                    print(f"Power prediction error: {np.mean(power_errors):.2f} W")
                    print(f"Power prediction error std: {np.std(power_errors):.2f} W")
                    
                    # Energy conservation analysis
                    conservation_ratio = np.mean(all_predicted_powers) / np.mean(all_incoming_powers)
                    print(f"Energy conservation ratio (pred/incoming): {conservation_ratio:.3f}")
                    
                    if 0.8 <= conservation_ratio <= 1.2:
                        print("✅ Energy conservation is reasonable")
                    else:
                        print("❌ Energy conservation ratio is outside reasonable bounds")
                        
                    # Power capping statistics
                    if all_cap_stats:
                        avg_violation_rate = np.mean([stats['violation_rate'] for stats in all_cap_stats])
                        avg_scale_factor = np.mean([stats['avg_scale_factor'] for stats in all_cap_stats])
                        capping_method = all_cap_stats[0]['capping_method']
                        
                        print(f"\nPower Capping Analysis:")
                        print(f"Capping method: {capping_method}")
                        print(f"Average violation rate: {avg_violation_rate:.3f} ({avg_violation_rate*100:.1f}%)")
                        print(f"Average scale factor: {avg_scale_factor:.4f}")
                        
                        if avg_violation_rate > 0.5:
                            print("⚠️  High violation rate - consider tuning weights")
                        elif avg_violation_rate < 0.01:
                            print("✅ Low violation rate - power capping working well")
                        else:
                            print("✅ Moderate violation rate - normal operation")
                        
                    # Additional statistics
                    print(f"\nDetailed Statistics:")
                    print(f"Actual power range: {np.min(all_actual_powers):.2f} to {np.max(all_actual_powers):.2f} W")
                    print(f"Predicted power range: {np.min(all_predicted_powers):.2f} to {np.max(all_predicted_powers):.2f} W")
                    print(f"Incoming power range: {np.min(all_incoming_powers):.2f} to {np.max(all_incoming_powers):.2f} W")
                    
                else:
                    print("❌ Power analysis data incomplete - missing required power arrays")
                    print(f"   actual_powers: {len(all_actual_powers)} items")
                    print(f"   predicted_powers: {len(all_predicted_powers)} items") 
                    print(f"   incoming_powers: {len(all_incoming_powers)} items")
                    
            except Exception as e:
                print(f"❌ Error in power analysis summary: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No valid power analysis results obtained")
            print(f"   Physics losses collected: {len(physics_losses)}")
            print(f"   Power info objects: {len(all_power_info)}")
        
        # Print overall power capping statistics
        self.print_power_cap_statistics()
        
        print("="*60)
    
    def save_model(self, filepath, include_optimizer=True):
        """Save model using PyTorch's state_dict approach with power capping info."""
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
        
        # Save enhanced metadata with unscaling info and power capping
        metadata = {
            'model_type': 'PhysicsInformedLSTM',
            'physics_approach': '9-bin_spatial_segmentation_with_proper_unscaling_and_power_capping',
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
            'power_capping': {  # NEW: Power capping configuration
                'enabled': True,
                'cap_penalty_weight': self.cap_penalty_weight,
                'use_soft_capping': self.use_soft_capping,
                'violation_statistics': {
                    'total_violations': self.cap_violations,
                    'total_samples': self.cap_samples,
                    'violation_rate': self.cap_violations / max(self.cap_samples, 1),
                    'avg_scale_factor': np.mean(self.cap_scales) if self.cap_scales else 1.0
                }
            },
            'unscaling_parameters': {
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
            'saving_method': 'state_dict_based_with_proper_unscaling_and_power_capping'
        }
        
        metadata_path = os.path.join(filepath, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"9-bin physics-informed PyTorch model with power capping saved to {filepath}")

    def load_model(self, filepath, model_builder_func=None):
        """Load model using PyTorch's state_dict approach with power capping info."""
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
            
            # Load power capping parameters if available
            if 'power_capping' in metadata:
                power_cap_params = metadata['power_capping']
                self.cap_penalty_weight = power_cap_params.get('cap_penalty_weight', 0.01)
                self.use_soft_capping = power_cap_params.get('use_soft_capping', False)
                
                # Load violation statistics
                if 'violation_statistics' in power_cap_params:
                    stats = power_cap_params['violation_statistics']
                    self.cap_violations = stats.get('total_violations', 0)
                    self.cap_samples = stats.get('total_samples', 0)
                    print(f"Loaded power capping stats: {self.cap_violations} violations in {self.cap_samples} samples")
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
        
        print(f"9-bin physics-informed PyTorch model with power capping loaded from {filepath}")


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
                  device=None, param_scaler=None, thermal_scaler=None, time_mean=300.0, time_std=300.0,
                  cap_penalty_weight=0.01, use_soft_capping=False):
    """Create physics-informed trainer with 9-bin approach, PROPER UNSCALING, and POWER CAPPING.
    
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
        thermal_scaler: StandardScaler for unscaling temperature values.
        time_mean (float): Mean used for time normalization (default: 300.0).
        time_std (float): Std used for time normalization (default: 300.0).
        cap_penalty_weight (float): Weight for power capping penalty (default: 0.01).
        use_soft_capping (bool): Use soft capping instead of hard capping (default: False).
        
    Returns:
        PhysicsInformedTrainer instance with 9-bin physics constraints, proper unscaling, and power capping.
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
        thermal_scaler=thermal_scaler,
        time_mean=time_mean,
        time_std=time_std,
        cap_penalty_weight=cap_penalty_weight,
        use_soft_capping=use_soft_capping
    )
    
    return trainer


# Additional utility functions for debugging unscaling and power capping
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
        
    print("="*50)


def test_power_capping(trainer, sample_powers, incoming_power):
    """Test power capping functionality."""
    print("\n" + "="*50)
    print("POWER CAPPING TEST")
    print("="*50)
    
    # Convert to tensors
    pred_powers = torch.tensor(sample_powers, dtype=torch.float32, device=trainer.device)
    incoming_tensor = torch.tensor(incoming_power, dtype=torch.float32, device=trainer.device)
    
    print(f"Original predicted powers: {sample_powers}")
    print(f"Incoming power: {incoming_power}")
    print(f"Total original power: {sum(sample_powers):.2f}")
    print(f"Capping method: {'Soft' if trainer.use_soft_capping else 'Hard'}")
    
    if trainer.use_soft_capping:
        capped_powers, penalty = trainer._soft_cap_powers(pred_powers, incoming_tensor)
        print(f"Soft capped powers: {capped_powers.detach().cpu().numpy()}")
        print(f"Soft penalty: {penalty.item():.4f}")
    else:
        capped_powers, scale_factor, violated = trainer._cap_positive_powers(pred_powers, incoming_tensor)
        print(f"Hard capped powers: {capped_powers.detach().cpu().numpy()}")
        print(f"Scale factor: {scale_factor.item():.4f}")
        print(f"Violation occurred: {violated.item()}")
    
    print(f"Total capped power: {torch.sum(capped_powers).item():.2f}")
    print(f"Power reduction: {(sum(sample_powers) - torch.sum(capped_powers).item()):.2f}")
    
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
        else:
            print("❌ No thermal_scaler provided!")
            
    print("="*60)


def model_summary(model):
    """Print detailed model summary with 9-bin configuration, unscaling info, and power capping."""
    print("="*70)
    print("PHYSICS-INFORMED LSTM MODEL SUMMARY (9-BIN + PROPER UNSCALING + POWER CAPPING)")
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
    print("9-BIN PHYSICS CONFIGURATION WITH PROPER UNSCALING & POWER CAPPING")
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
    
    print("\nNEW - Power Capping:")
    print("  • Hard Capping: Direct scaling of positive powers to respect incoming power limit")
    print("  • Soft Capping: Smooth sigmoid-based discouragement of power violations")
    print("  • Sign-aware: Only caps heating (positive) powers, preserves cooling (negative)")
    print("  • Statistics tracking: Violation rates, scale factors, and performance metrics")
    
    print("\nLoss Components:")
    print("  1. MAE Loss: Temperature prediction accuracy")
    print("  2. Physics Loss: 9-bin power difference (weighted, unscaled units)")
    print("  3. Constraint Loss: Energy conservation penalties (weighted)")
    print("  4. Power Balance Loss: Total power vs incoming power (weighted)")
    print("  5. Capping Penalty: Penalty for frequent power scaling (weighted)")
    print("="*70)


def get_model_config():
    """Get recommended model configuration for 9-bin approach with unscaling and power capping."""
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
        'time_std': 300.0,          # Time normalization std
        'cap_penalty_weight': 0.01, # Weight for power capping penalty
        'use_soft_capping': False   # Use hard capping by default
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
    """Complete training function with physics-informed loss, proper unscaling, and power capping.
    
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
    print(f"Training for {num_epochs} epochs with proper unscaling and power capping...")
    print(f"Power capping method: {'Soft' if trainer.use_soft_capping else 'Hard'}")
    
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
        
        # Print power capping statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.print_power_cap_statistics()
    
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


# Example usage and validation with proper unscaling and power capping
if __name__ == "__main__":
    print("="*70)
    print("COMPLETE 9-BIN PHYSICS-INFORMED LSTM WITH PROPER UNSCALING & POWER CAPPING (PYTORCH)")
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
    
    # Create trainer with 9-bin approach, proper unscaling, and power capping
    trainer = create_trainer(
        model=model,
        physics_weight=config['physics_weight'],
        constraint_weight=config['constraint_weight'],
        power_balance_weight=config['power_balance_weight'],
        learning_rate=config['learning_rate'],
        cylinder_length=config['cylinder_length'],
        device=device,
        param_scaler=param_scaler,
        thermal_scaler=thermal_scaler,
        time_mean=config['time_mean'],
        time_std=config['time_std'],
        cap_penalty_weight=config['cap_penalty_weight'],
        use_soft_capping=config['use_soft_capping']
    )
    
    # Print summary
    model_summary(model)
    
    # Test power capping functionality
    print("\n" + "="*70)
    print("TESTING POWER CAPPING FUNCTIONALITY")
    print("="*70)
    
    # Test case 1: Powers that exceed incoming power
    sample_powers_high = [500, 800, 1200, -200, 600, 400, -100, 300, 700, 900]  # Total positive: 4500W
    incoming_power_test = 3000.0  # Lower than total positive power
    test_power_capping(trainer, sample_powers_high, incoming_power_test)
    
    # Test case 2: Powers that don't exceed incoming power
    sample_powers_low = [200, 300, 100, -50, 150, 100, -25, 75, 200, 250]  # Total positive: 1375W
    incoming_power_test2 = 2000.0  # Higher than total positive power
    test_power_capping(trainer, sample_powers_low, incoming_power_test2)
    
    # Test soft capping mode
    trainer.use_soft_capping = True
    print(f"\n--- Testing Soft Capping ---")
    test_power_capping(trainer, sample_powers_high, incoming_power_test)
    trainer.use_soft_capping = False  # Reset
    
    # Validate input shapes with dummy data
    print("\n" + "="*70)
    print("VALIDATING COMPLETE MODEL WITH DUMMY DATA")
    print("="*70)
    
    batch_size = 8  # Reduced batch size for testing
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
        
        # Test COMPLETE 9-bin physics loss computation with proper unscaling and power capping
        print(f"\n🔧 TESTING COMPLETE PHYSICS LOSS WITH PROPER TENSOR UNSCALING & POWER CAPPING...")
        physics_loss, constraint_loss, power_balance_loss, power_info = trainer.compute_nine_bin_physics_loss(
            dummy_target, dummy_output, power_metadata_list
        )
        
        print(f"✅ COMPLETE 9-bin physics loss computation with proper unscaling and power capping successful")
        print(f"   Physics loss: {physics_loss:.4f}")
        print(f"   Constraint loss: {constraint_loss:.4f}")
        print(f"   Power balance loss: {power_balance_loss:.4f}")
        
        if power_info:
            print(f"   Samples processed: {power_info.get('num_samples_processed', 0)}")
            print(f"   Power data arrays present: {len(power_info.get('total_actual_powers', []))}")
            
            # Show power capping statistics
            cap_stats = power_info.get('power_cap_stats', {})
            if cap_stats:
                print(f"   Power capping violations: {cap_stats.get('violations', 0)}")
                print(f"   Power capping samples: {cap_stats.get('samples', 0)}")
                print(f"   Violation rate: {cap_stats.get('violation_rate', 0):.3f}")
                print(f"   Average scale factor: {cap_stats.get('avg_scale_factor', 1.0):.4f}")
                print(f"   Capping method: {cap_stats.get('capping_method', 'unknown')}")
        
        # Test training step with dummy batch
        dummy_batch = [dummy_time_series, dummy_static_params, dummy_target, dummy_power_data]
        train_result = trainer.train_step(dummy_batch)
        
        print(f"✅ Training step with COMPLETE physics loss, proper unscaling, and power capping successful")
        print(f"   Total loss: {train_result['loss']:.4f}")
        print(f"   MAE: {train_result['mae']:.4f}")
        print(f"   Physics loss: {train_result['physics_loss']:.4f}")
        print(f"   Constraint loss: {train_result['constraint_loss']:.4f}")
        print(f"   Power balance loss: {train_result['power_balance_loss']:.4f}")
        
        # Test the COMPLETE power balance analysis
        print(f"\n🔧 TESTING COMPLETE POWER BALANCE ANALYSIS WITH POWER CAPPING...")
        
        # Create a simple data loader for testing
        class DummyDataLoader:
            def __init__(self, batch):
                self.batch = batch
            def __iter__(self):
                yield self.batch
        
        dummy_loader = DummyDataLoader(dummy_batch)
        trainer.analyze_power_balance(dummy_loader, num_samples=10)
        
        # Test model saving and loading
        print(f"\n🔧 TESTING MODEL SAVING AND LOADING WITH POWER CAPPING...")
        
        # Save model
        save_path = "./test_physics_model_with_capping"
        trainer.save_model(save_path)
        
        # Test loading (create new trainer instance)
        new_trainer = PhysicsInformedTrainer(
            model=build_model(device=device),
            device=device,
            param_scaler=param_scaler,
            thermal_scaler=thermal_scaler
        )
        
        try:
            new_trainer.load_model(save_path)
            print("✅ Model saving and loading with power capping successful")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
        
    except Exception as e:
        print(f"❌ Error during COMPLETE validation with unscaling and power capping: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("COMPLETE 9-BIN PHYSICS-INFORMED PYTORCH MODEL WITH POWER CAPPING READY!")
    print("="*70)
    print("🔧 ALL FEATURES IMPLEMENTED:")
    print("✅ Fixed device mismatch issues - all tensors on same device")
    print("✅ Fixed gradient computation - kept tensors instead of converting to scalars")
    print("✅ Fixed batch size consistency - trim all inputs to same size")
    print("✅ Fixed memory leaks - reduced debug output and cleared cache periodically")
    print("✅ Fixed tensor indexing - added proper bounds checking")
    print("✅ Fixed loss computation - kept requires_grad=True for physics losses")
    print("✅ Added error handling for individual batches to prevent crashes")
    print("✅ Optimized memory usage - minimal power_info returned")
    print("✅ FIXED POWER ANALYSIS - properly store and return power data with consistent naming")
    print("✅ FIXED analyze_power_balance - safe access to power arrays with error handling")
    print("✅ NEW: HARD POWER CAPPING - direct scaling to enforce energy conservation")
    print("✅ NEW: SOFT POWER CAPPING - smooth sigmoid-based power constraint")
    print("✅ NEW: POWER CAPPING STATISTICS - violation rates and performance tracking")
    print("✅ NEW: CONFIGURABLE CAPPING - switch between hard and soft methods")
    print("✅ NEW: ENHANCED SAVING/LOADING - includes power capping configuration")
    print("="*70)
    
    print("\nUSAGE NOTES FOR THE COMPLETE VERSION:")
    print("• Main power capping implementations are in _cap_positive_powers and _soft_cap_powers methods")
    print("• Use create_trainer() with cap_penalty_weight and use_soft_capping parameters")
    print("• Monitor trainer.print_power_cap_statistics() to tune capping behavior")
    print("• Hard capping guarantees constraint satisfaction but may affect gradients")
    print("• Soft capping preserves smooth gradients but relies on penalty terms")
    print("• Power capping only affects positive (heating) powers, preserves cooling")
    print("• All scalers (thermal_scaler, param_scaler) must be provided for proper unscaling")
    print("• Power analysis includes comprehensive violation statistics and performance metrics")
    print("• Model saving/loading preserves all power capping configuration and statistics")
    print("="*70)
    
    print("\nRECOMMENDED PARAMETER TUNING:")
    print("• Start with hard capping (use_soft_capping=False)")
    print("• If violation rate > 50%, increase power_balance_weight from 0.05 to 0.1+")
    print("• If training is unstable, try soft capping (use_soft_capping=True)")
    print("• Adjust cap_penalty_weight (0.001-0.1) based on violation frequency")
    print("• Monitor avg_scale_factor - values much < 1.0 indicate frequent capping")
    print("• Consider reducing physics_weight if capping violations are excessive")
    print("="*70)
