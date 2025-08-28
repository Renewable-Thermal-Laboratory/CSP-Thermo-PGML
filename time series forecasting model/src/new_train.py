import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import warnings
import re
import argparse
import pandas as pd

# Set PyTorch multiprocessing sharing strategy
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

# Suppress all warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Fixed imports - ensure all classes are imported
from new_model import (
    build_model, 
    create_trainer, 
    compute_r2_score, 
    PhysicsInformedTrainer,
    PhysicsInformedLSTM
)
from new_dataset_builder import create_data_loaders


# =====================
# HELPER FUNCTION FOR SAFE TENSOR/FLOAT CONVERSION
# =====================
def _as_float(x):
    """Convert tensor or float to float safely."""
    return x.item() if hasattr(x, "item") else float(x)


# =====================
# CLEANUP HELPER FOR MEMORY MANAGEMENT
# =====================
def cleanup_dataloaders(*objs):
    for o in objs:
        if o is None:
            continue
        try:
            del o
        except Exception:
            pass
    import gc; gc.collect()
    # Free accelerator caches if present
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# =====================
# FIXED HORIZON-AGNOSTIC POWER METADATA PROCESSING FUNCTIONS
# =====================
def extract_power_metadata_from_batch(time_series_batch, static_params_batch, targets_batch, thermal_scaler, param_scaler, horizon_steps=1):
    """
    Extract power metadata from the actual batch data using canonical keys (horizon-agnostic).
    
    FIXES:
    1. Use targets_batch instead of last input row for target temperatures
    2. Compute time_target correctly using dt * horizon_steps
    3. Include absorptivity and surface fraction in metadata
    
    Args:
        time_series_batch: (batch_size, seq_len, 11) - time + 10 temperature sensors
        static_params_batch: (batch_size, 4) - [htc, flux, abs, surf] (scaled)
        targets_batch: (batch_size, 10) - target temperatures (scaled)
        thermal_scaler: StandardScaler for temperatures
        param_scaler: StandardScaler for static parameters
        horizon_steps: int - prediction horizon in steps
    
    Returns:
        List of power metadata dictionaries for physics calculations
    """
    batch_size = time_series_batch.shape[0]
    power_metadata_list = []
    
    # Convert tensors to numpy for processing
    time_series_np = time_series_batch.detach().cpu().numpy()
    static_params_np = static_params_batch.detach().cpu().numpy()
    targets_np = targets_batch.detach().cpu().numpy()
    
    for batch_idx in range(batch_size):
        try:
            # Extract temperature data from time series
            # time_series shape: (seq_len, 11) where first column is time, remaining 10 are temperatures
            temp_sequence = time_series_np[batch_idx, :, 1:]  # Skip time column, get temperatures (seq_len, 10)
            
            # Get temperatures at first timestep and use provided targets
            temps_initial_scaled = temp_sequence[0, :]   # First timestep (10 temperatures)
            temps_target_scaled = targets_np[batch_idx]  # Use provided targets, not last input row
            
            # Unscale temperatures to get actual physical values
            temps_initial_unscaled = thermal_scaler.inverse_transform([temps_initial_scaled])[0]
            temps_target_unscaled = thermal_scaler.inverse_transform([temps_target_scaled])[0]
            
            # Compute time information correctly
            t0 = float(time_series_np[batch_idx, 0, 0])  # Time at first timestep
            if time_series_np.shape[1] > 1:
                dt = float(time_series_np[batch_idx, 1, 0] - time_series_np[batch_idx, 0, 0])
            else:
                dt = 1.0  # fallback time step
            
            tT = t0 + horizon_steps * dt  # Correct target time
            time_initial = t0
            time_target = tT
            time_diff = max(time_target - time_initial, 1e-8)
            
            # Extract and unscale static parameters [htc, flux, abs, surf]
            static_params_scaled = static_params_np[batch_idx, :]
            static_params_unscaled = param_scaler.inverse_transform([static_params_scaled])[0]
            
            htc_unscaled = float(static_params_unscaled[0])    # Heat transfer coefficient (W/m²·K)
            flux_unscaled = float(static_params_unscaled[1])   # Heat flux (W/m²)
            abs_coeff = float(static_params_unscaled[2])       # Include absorptivity
            surf_frac = float(static_params_unscaled[3])       # Include surface fraction
            
            # Create power metadata dictionary with canonical keys
            power_metadata = {
                # Canonical keys (horizon-agnostic)
                'temps_row1': temps_initial_unscaled.tolist(),     # List of 10 floats - initial state
                'temps_target': temps_target_unscaled.tolist(),    # List of 10 floats - target state
                'time_row1': time_initial,                         # Float - initial time
                'time_target': time_target,                        # Float - target time
                'time_diff': time_diff,                            # Float - time difference
                'horizon_steps': horizon_steps,                    # Int - prediction horizon
                'time_normalized': False,                          # Bool - whether time is normalized
                
                # Physics parameters (clarified: h is HTC, not cylinder height)
                'h': htc_unscaled,                                 # Float - Heat Transfer Coefficient (W/m²·K)
                'q0': flux_unscaled,                              # Float - heat flux (W/m²)
                'abs_coeff': abs_coeff,                           # Absorptivity coefficient
                'surf_frac': surf_frac,                           # Illuminated surface fraction
                
                # Legacy keys for backward compatibility (will be deprecated)
                'temps_row21': temps_target_unscaled.tolist(),    # Legacy: same as temps_target
                'time_row21': time_target,                        # Legacy: same as time_target
            }
            
            power_metadata_list.append(power_metadata)
            
        except Exception as e:
            print(f"Error extracting power metadata for batch index {batch_idx}: {e}")
            # Create dummy metadata as fallback
            power_metadata_list.append({
                'temps_row1': [300.0] * 10,
                'temps_target': [301.0] * 10,
                'time_row1': 0.0,
                'time_target': float(horizon_steps),
                'time_diff': float(horizon_steps),
                'horizon_steps': horizon_steps,
                'time_normalized': False,
                'h': 50.0,  # HTC, not cylinder height
                'q0': 1000.0,
                'abs_coeff': 0.8,
                'surf_frac': 1.0,
                # Legacy keys
                'temps_row21': [301.0] * 10,
                'time_row21': float(horizon_steps),
            })
    
    return power_metadata_list


def process_power_data_batch_fixed(power_data_list):
    """
    Fixed version that handles the extracted power metadata correctly (horizon-agnostic).
    """
    if not power_data_list:
        return None
    
    batch_size = len(power_data_list)
    processed_metadata = []
    
    print(f"Processing extracted power data batch with {batch_size} samples")
    
    for i, power_data in enumerate(power_data_list):
        if power_data is None or not isinstance(power_data, dict):
            print(f"Warning: Invalid power_data at index {i}, using dummy values")
            processed_metadata.append({
                'temps_row1': [300.0] * 10,
                'temps_target': [301.0] * 10,
                'time_diff': 1.0,
                'horizon_steps': 1,
                'h': 50.0,  # HTC
                'q0': 1000.0,
                'abs_coeff': 0.8,
                'surf_frac': 1.0
            })
            continue
            
        try:
            # Extract values using canonical keys (with legacy fallbacks)
            temps_row1 = power_data.get('temps_row1', [300.0] * 10)
            temps_target = power_data.get('temps_target', power_data.get('temps_row21', [301.0] * 10))
            time_diff = power_data.get('time_diff', 1.0)
            horizon_steps = power_data.get('horizon_steps', 1)
            htc_value = power_data.get('h', 50.0)  # Heat transfer coefficient
            q0_value = power_data.get('q0', 1000.0)
            abs_coeff = power_data.get('abs_coeff', 0.8)  # Include absorptivity
            surf_frac = power_data.get('surf_frac', 1.0)  # Include surface fraction
            
            # Validate data
            if (isinstance(temps_row1, list) and len(temps_row1) == 10 and
                isinstance(temps_target, list) and len(temps_target) == 10 and
                isinstance(time_diff, (int, float)) and time_diff > 0 and
                isinstance(htc_value, (int, float)) and isinstance(q0_value, (int, float))):
                
                processed_metadata.append({
                    'temps_row1': [float(x) for x in temps_row1],
                    'temps_target': [float(x) for x in temps_target],
                    'time_diff': float(time_diff),
                    'horizon_steps': int(horizon_steps),
                    'h': float(htc_value),  # HTC
                    'q0': float(q0_value),
                    'abs_coeff': float(abs_coeff),
                    'surf_frac': float(surf_frac)
                })
            else:
                print(f"Warning: Invalid data format at index {i}, using dummy values")
                processed_metadata.append({
                    'temps_row1': [300.0] * 10,
                    'temps_target': [301.0] * 10,
                    'time_diff': 1.0,
                    'horizon_steps': 1,
                    'h': 50.0,
                    'q0': 1000.0,
                    'abs_coeff': 0.8,
                    'surf_frac': 1.0
                })
                
        except Exception as e:
            print(f"Error processing power_data at index {i}: {e}")
            processed_metadata.append({
                'temps_row1': [300.0] * 10,
                'temps_target': [301.0] * 10,
                'time_diff': 1.0,
                'horizon_steps': 1,
                'h': 50.0,
                'q0': 1000.0,
                'abs_coeff': 0.8,
                'surf_frac': 1.0
            })
    
    print(f"Successfully processed {len(processed_metadata)} power metadata entries")
    return processed_metadata


class FixedUnscaledEvaluationTrainer:
    """
    Fixed wrapper that extracts power metadata from actual batch data (horizon-agnostic).
    """
    
    def __init__(self, base_trainer, thermal_scaler, param_scaler, horizon_steps=1, device=None):
        self.base_trainer = base_trainer
        self.thermal_scaler = thermal_scaler
        self.param_scaler = param_scaler
        self.horizon_steps = horizon_steps
        
        # Device handling
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert scaler parameters to PyTorch tensors
        self.thermal_mean = torch.tensor(thermal_scaler.mean_, dtype=torch.float32, device=self.device)
        self.thermal_scale = torch.tensor(thermal_scaler.scale_, dtype=torch.float32, device=self.device)
        
        # History for unscaled metrics
        self.unscaled_history = {
            'train_mae_unscaled': [],
            'train_rmse_unscaled': [],
            'val_mae_unscaled': [],
            'val_rmse_unscaled': []
        }
        
        # Access the underlying model and methods
        self.model = self.base_trainer.model
    
    def unscale_temperatures(self, scaled_temps):
        """Convert scaled temperatures back to original units using PyTorch operations with device-safe tensors."""
        # FIXED: Move scaler tensors to the same device as input to prevent device mismatch
        mean = self.thermal_mean.to(scaled_temps.device)
        scale = self.thermal_scale.to(scaled_temps.device)
        unscaled_temps = scaled_temps * scale + mean
        return unscaled_temps
    
    def train_step_unscaled(self, batch):
        """Training step with fixed power metadata extraction (horizon-agnostic)."""
        # Extract original batch components
        time_series, static_params, targets, original_power_data = batch
        
        # Move to device
        time_series = time_series.to(self.device)
        static_params = static_params.to(self.device)
        targets = targets.to(self.device)
        
        # Extract power metadata from actual batch data (horizon-agnostic)
        extracted_power_metadata = extract_power_metadata_from_batch(
            time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps
        )
        
        # Use the extracted power metadata
        trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
        train_results = self.base_trainer.train_step(trainer_batch)
        
        # Get unscaled temperatures for additional metrics
        with torch.no_grad():
            y_pred_scaled = self.base_trainer.model([time_series, static_params])
            
            # Convert to unscaled
            y_true_unscaled = self.unscale_temperatures(targets)
            y_pred_unscaled = self.unscale_temperatures(y_pred_scaled)
            
            # Compute unscaled metrics
            mae_unscaled = torch.mean(torch.abs(y_true_unscaled - y_pred_unscaled))
            rmse_unscaled = torch.sqrt(torch.mean(torch.square(y_true_unscaled - y_pred_unscaled)))
            
            # Add unscaled metrics to results
            train_results.update({
                'mae_unscaled': mae_unscaled.item(),
                'rmse_unscaled': rmse_unscaled.item()
            })
        
        return train_results
    
    def validation_step_unscaled(self, batch):
        """Validation step with fixed power metadata extraction (horizon-agnostic)."""
        # Extract original batch components
        time_series, static_params, targets, original_power_data = batch
        
        # Move to device
        time_series = time_series.to(self.device)
        static_params = static_params.to(self.device)
        targets = targets.to(self.device)
        
        # Extract power metadata from actual batch data (horizon-agnostic)
        extracted_power_metadata = extract_power_metadata_from_batch(
            time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps
        )
        
        # Use the extracted power metadata
        trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
        val_results = self.base_trainer.validation_step(trainer_batch)
        
        # Get unscaled temperatures for additional metrics
        with torch.no_grad():
            y_pred_scaled = self.base_trainer.model([time_series, static_params])
            
            # Convert to unscaled
            y_true_unscaled = self.unscale_temperatures(targets)
            y_pred_unscaled = self.unscale_temperatures(y_pred_scaled)
            
            # Compute unscaled metrics
            mae_unscaled = torch.mean(torch.abs(y_true_unscaled - y_pred_unscaled))
            rmse_unscaled = torch.sqrt(torch.mean(torch.square(y_true_unscaled - y_pred_unscaled)))
            
            # Add unscaled metrics to results
            val_results.update({
                'val_mae_unscaled': mae_unscaled.item(),
                'val_rmse_unscaled': rmse_unscaled.item()
            })
        
        return val_results

    def train_epoch_unscaled(self, train_loader, val_loader=None):
        """Train for one epoch with fixed power metadata extraction (horizon-agnostic)."""
        from collections import defaultdict
        
        # Initialize metrics for this epoch
        epoch_train_metrics = defaultdict(list)
        epoch_val_metrics = defaultdict(list)
        
        # Training loop
        for batch in train_loader:
            metrics = self.train_step_unscaled(batch)
            for key, value in metrics.items():
                epoch_train_metrics[f'train_{key}'].append(value)
        
        # Validation loop
        if val_loader is not None:
            for batch in val_loader:
                metrics = self.validation_step_unscaled(batch)
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
            if key in self.base_trainer.history:
                self.base_trainer.history[key].append(float(value))
        
        return results

    def evaluate_unscaled(self, data_loader, split_name="test"):
        """Comprehensive evaluation with fixed power metadata extraction (horizon-agnostic)."""
        self.model.eval()
        
        all_predictions_scaled = []
        all_targets_scaled = []
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in data_loader:
                time_series, static_params, targets, original_power_data = batch
                
                # Move to device
                time_series = time_series.to(self.device)
                static_params = static_params.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions
                predictions_scaled = self.model([time_series, static_params])
                
                # Keep tensors on GPU during collection
                all_predictions_scaled.append(predictions_scaled.detach().cpu())
                all_targets_scaled.append(targets.detach().cpu())
                
                # Extract power metadata from actual batch data (horizon-agnostic)
                extracted_power_metadata = extract_power_metadata_from_batch(
                    time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps
                )
                
                # Compute batch metrics using validation step with extracted metadata
                trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
                batch_metrics = self.base_trainer.validation_step(trainer_batch)
                
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
        
        # Concatenate all predictions and targets (lists already hold CPU tensors)
        all_predictions_scaled = torch.cat(all_predictions_scaled, dim=0)
        all_targets_scaled = torch.cat(all_targets_scaled, dim=0)
        
        # Convert to unscaled
        all_predictions_unscaled = self.unscale_temperatures(all_predictions_scaled)
        all_targets_unscaled = self.unscale_temperatures(all_targets_scaled)
        
        # Overall unscaled metrics - FIXED with _as_float
        mae_unscaled = _as_float(torch.mean(torch.abs(all_targets_unscaled - all_predictions_unscaled)))
        rmse_unscaled = _as_float(torch.sqrt(torch.mean(torch.square(all_targets_unscaled - all_predictions_unscaled))))
        r2_overall_unscaled = _as_float(compute_r2_score(all_targets_unscaled, all_predictions_unscaled))
        
        # Per-sensor unscaled metrics - FIXED with _as_float
        per_sensor_metrics = []
        for sensor_idx in range(10):
            y_true_sensor = all_targets_unscaled[:, sensor_idx]
            y_pred_sensor = all_predictions_unscaled[:, sensor_idx]
            
            mae_sensor = _as_float(torch.mean(torch.abs(y_true_sensor - y_pred_sensor)))
            rmse_sensor = _as_float(torch.sqrt(torch.mean(torch.square(y_true_sensor - y_pred_sensor))))
            r2_sensor = _as_float(compute_r2_score(y_true_sensor, y_pred_sensor))
            
            per_sensor_metrics.append({
                'mae': mae_sensor,
                'rmse': rmse_sensor,
                'r2': r2_sensor
            })
        
        # Aggregate batch metrics
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            aggregated_metrics[key] = np.mean(values)
        
        # Create proper constraint loss from soft and excess penalties
        test_physics_loss = aggregated_metrics.get('val_physics_loss', 0.0)
        test_constraint_loss = (aggregated_metrics.get('val_soft_penalty', 0.0) + 
                               aggregated_metrics.get('val_excess_penalty', 0.0))
        test_power_balance_loss = aggregated_metrics.get('val_power_balance_loss', 0.0)
        
        # Final results with all required keys - FIXED with _as_float
        results = {
            f'{split_name}_mae_unscaled': mae_unscaled,
            f'{split_name}_rmse_unscaled': rmse_unscaled,
            f'{split_name}_r2_overall_unscaled': r2_overall_unscaled,
            f'{split_name}_per_sensor_metrics': per_sensor_metrics,
            f'{split_name}_physics_loss': test_physics_loss,
            f'{split_name}_constraint_loss': test_constraint_loss,
            f'{split_name}_power_balance_loss': test_power_balance_loss,
            # Move to CPU only when converting to numpy
            'predictions_unscaled': {
                'y_true': all_targets_unscaled.detach().cpu().numpy(),
                'y_pred': all_predictions_unscaled.detach().cpu().numpy()
            }
        }
        
        # Add aggregated batch metrics
        results.update(aggregated_metrics)
        
        horizon_label = f"{self.horizon_steps} step{'s' if self.horizon_steps != 1 else ''}"
        print(f"\nTEST SET EVALUATION (UNSCALED - HORIZON = {horizon_label}):")
        print(f"   MAE:  {mae_unscaled:.2f} K")
        print(f"   RMSE: {rmse_unscaled:.2f} K") 
        print(f"   R²:   {r2_overall_unscaled:.6f}")
        print(f"   Physics Loss: {test_physics_loss:.6f}")
        print(f"   Constraint Loss: {test_constraint_loss:.6f}")
        print(f"   Power Balance Loss: {test_power_balance_loss:.6f}")
        
        return results

    def analyze_power_balance(self, data_loader, num_samples=100):
        """
        Power balance analysis with fixed metadata extraction (horizon-agnostic).
        Now returns results dictionary for clean integration.
        """
        horizon_label = f"{self.horizon_steps} step{'s' if self.horizon_steps != 1 else ''}"
        print("\n" + "="*60)
        print(f"POWER BALANCE ANALYSIS (HORIZON = {horizon_label})")
        print("="*60)
        
        total_actual_powers = []
        total_predicted_powers = []
        incoming_powers = []
        
        self.model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break
                
                time_series, static_params, targets, original_power_data = batch
                
                # Move to device
                time_series = time_series.to(self.device)
                static_params = static_params.to(self.device)
                targets = targets.to(self.device)
                
                try:
                    # Extract power metadata from actual batch data (horizon-agnostic)
                    extracted_power_metadata = extract_power_metadata_from_batch(
                        time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps
                    )
                    
                    if extracted_power_metadata:
                        # Get predictions
                        y_pred = self.model([time_series, static_params])
                        
                        # Compute power analysis with correct call signature
                        physics_loss, soft_penalty, excess_penalty, power_balance_loss, power_info = \
                            self.base_trainer.compute_nine_bin_physics_loss(
                                y_pred, extracted_power_metadata
                            )
                        
                        if power_info:  # If analysis succeeded
                            # Use correct plural key names that match compute_nine_bin_physics_loss
                            if 'total_actual_powers' in power_info and power_info['total_actual_powers']:
                                total_actual_powers.extend(power_info['total_actual_powers'])
                            
                            if 'total_predicted_powers' in power_info and power_info['total_predicted_powers']:
                                total_predicted_powers.extend(power_info['total_predicted_powers'])
                            
                            if 'incoming_powers' in power_info and power_info['incoming_powers']:
                                incoming_powers.extend(power_info['incoming_powers'])
                            
                            # Count samples processed
                            sample_count += power_info.get('num_samples_processed', 0)
                            
                except Exception as e:
                    print(f"Warning: Error in power analysis: {e}")
                    continue
        
        # Build return dictionary with real data
        if len(total_actual_powers) > 0:
            total_actual_powers = np.array(total_actual_powers)
            total_predicted_powers = np.array(total_predicted_powers)
            incoming_powers = np.array(incoming_powers)
            
            print(f"Samples analyzed: {len(total_actual_powers)}")
            print(f"Prediction horizon: {horizon_label}")
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
            
            # CRITICAL FIX: Make power balance ratios numerically safe
            eps = 1e-8
            incoming_safe = np.maximum(incoming_powers, eps)
            actual_to_incoming = total_actual_powers / incoming_safe
            predicted_to_incoming = total_predicted_powers / incoming_safe
            
            print(f"\nPOWER BALANCE RATIOS:")
            print(f"  Actual/Incoming ratio - Mean: {np.mean(actual_to_incoming):.3f}, Std: {np.std(actual_to_incoming):.3f}")
            print(f"  Predicted/Incoming ratio - Mean: {np.mean(predicted_to_incoming):.3f}, Std: {np.std(predicted_to_incoming):.3f}")
            
            # Check for violations (use original incoming_powers for violation checks)
            actual_violations = np.sum(total_actual_powers > incoming_powers)
            predicted_violations = np.sum(total_predicted_powers > incoming_powers)
            
            print(f"\nENERGY CONSERVATION VIOLATIONS:")
            print(f"  Actual power > incoming: {actual_violations}/{len(total_actual_powers)} ({100*actual_violations/len(total_actual_powers):.1f}%)")
            print(f"  Predicted power > incoming: {predicted_violations}/{len(total_predicted_powers)} ({100*predicted_violations/len(total_predicted_powers):.1f}%)")
            
            print(f"\nSUCCESS: Real power data extracted and analyzed (horizon = {horizon_label})!")
            
            # Return structured results
            return {
                'horizon_steps': self.horizon_steps,
                'horizon_label': horizon_label,
                'mean_actual_power': float(np.mean(total_actual_powers)),
                'mean_predicted_power': float(np.mean(total_predicted_powers)),
                'mean_incoming_power': float(np.mean(incoming_powers)),
                'mean_actual_to_incoming_ratio': float(np.mean(actual_to_incoming)),
                'mean_predicted_to_incoming_ratio': float(np.mean(predicted_to_incoming)),
                'conservation_violations': {
                    'count': int(predicted_violations),
                    'percentage': float(100.0 * predicted_violations / len(total_predicted_powers)),
                    'mean_violation_amount': float(np.mean((total_predicted_powers - incoming_powers)[total_predicted_powers > incoming_powers])) if predicted_violations > 0 else 0.0
                }
            }
        else:
            print("No valid power analysis results obtained")
            
            # Return empty results structure
            return {
                'horizon_steps': self.horizon_steps,
                'horizon_label': horizon_label,
                'mean_actual_power': 0.0,
                'mean_predicted_power': 0.0,
                'mean_incoming_power': 0.0,
                'mean_actual_to_incoming_ratio': 0.0,
                'mean_predicted_to_incoming_ratio': 0.0,
                'conservation_violations': {
                    'count': 0,
                    'percentage': 0.0,
                    'mean_violation_amount': 0.0
                }
            }
            
        print("="*60)

    def save_model(self, filepath, include_optimizer=True):
        """Delegate to base trainer's save method."""
        return self.base_trainer.save_model(filepath, include_optimizer)

    def load_model(self, filepath, model_builder_func=None):
        """Delegate to base trainer's load method."""
        return self.base_trainer.load_model(filepath, model_builder_func)


# =====================
# HORIZON-AGNOSTIC FILENAME EXTRACTION FUNCTIONS (Updated)
# =====================

def get_test_filenames_and_sample_mapping(test_loader):
    """
    Extract actual filenames from the test dataset properly (horizon-agnostic).
    """
    print("Extracting ACTUAL test file information...")
    
    sample_to_filename = {}
    
    try:
        # Get the dataset from test_loader
        if hasattr(test_loader, 'dataset'):
            dataset = test_loader.dataset
            print(f"Found test dataset: {type(dataset).__name__}")
            
            # Access the actual current_files from TempSequenceDataset
            if hasattr(dataset, 'current_files'):
                current_files = dataset.current_files
                print(f"Found {len(current_files)} test files in dataset.current_files")
                
                # Build sample mapping based on sample_indices
                if hasattr(dataset, 'sample_indices'):
                    sample_indices = dataset.sample_indices
                    print(f"Found {len(sample_indices)} sample indices")
                    
                    # Each element in sample_indices is (file_path, start_idx)
                    for idx, (file_path, start_idx) in enumerate(sample_indices):
                        filename = os.path.basename(file_path)
                        sample_to_filename[idx] = filename
                    
                    print(f"Successfully mapped {len(sample_to_filename)} samples to ACTUAL filenames")
                    
                    # Show first few mappings as verification
                    print(f"First 5 sample-to-filename mappings:")
                    for i in range(min(5, len(sample_to_filename))):
                        print(f"   Sample {i}: {sample_to_filename[i]}")
                    
                    return sample_to_filename
                
                else:
                    print("Dataset doesn't have sample_indices attribute")
            
            # Alternative: try to access files directly
            if hasattr(dataset, 'test_files'):
                test_files = dataset.test_files
                print(f"Found {len(test_files)} files in dataset.test_files")
                
                for idx, file_path in enumerate(test_files):
                    filename = os.path.basename(file_path)
                    sample_to_filename[idx] = filename
                
                print(f"Successfully mapped {len(sample_to_filename)} files to filenames")
                return sample_to_filename
            
            # Try accessing split files
            current_split = getattr(dataset, 'split', 'unknown')
            print(f"Dataset split: {current_split}")
            
            # Check for different file list attributes
            possible_file_attrs = ['current_files', 'test_files', 'val_files', 'train_files', 'files']
            for attr in possible_file_attrs:
                if hasattr(dataset, attr):
                    files = getattr(dataset, attr)
                    if files and len(files) > 0:
                        print(f"Found {len(files)} files in dataset.{attr}")
                        
                        # If this is for test dataset, map files to samples
                        for idx, file_path in enumerate(files):
                            filename = os.path.basename(file_path)
                            sample_to_filename[idx] = filename
                        
                        print(f"Successfully mapped {len(sample_to_filename)} files from {attr}")
                        return sample_to_filename
        
        print("Could not extract actual filenames from dataset")
        print("Available dataset attributes:", [attr for attr in dir(dataset) if not attr.startswith('_')])
        
    except Exception as e:
        print(f"Error extracting filenames: {e}")
    
    # Only use fallback if absolutely necessary
    print("WARNING: Using fallback generic filenames - this is not ideal!")
    print("Please check your dataset implementation to expose actual filenames")
    
    # Create a minimal set of realistic fallback names based on your file patterns
    fallback_files = [
        "h0.4_flux40000_abs15_surf70_600s.csv",
        "h0.4_flux50000_abs10_surf50_600s.csv", 
        "h0.5_flux40000_abs20_surf90_600s.csv",
        "h0.5_flux100000_abs5_surf50_600s.csv",
        "h1.0_flux40000_abs15_surf70_600s.csv"
    ]
    
    # Map samples to fallback filenames (cycling through the list)
    for sample_idx in range(1000):  # Reasonable upper limit
        filename = fallback_files[sample_idx % len(fallback_files)]
        sample_to_filename[sample_idx] = filename
    
    print(f"Created {len(sample_to_filename)} mappings using fallback filenames")
    return sample_to_filename


def parse_height_from_filename(filename):
    """
    Parse cylinder height from filename (horizon-agnostic).
    Expected format: "h{height}_flux{flux}_abs{abs}_surf{surf}_{time}s.csv"
    
    FIXED: Removed special h6 case - use consistent parsing or Config.cylinder_length
    """
    # Look for patterns like "h0.4", "h0.5", "h1.0", etc.
    height_pattern = r'h(\d+\.?\d*)'
    match = re.search(height_pattern, filename.lower())
    
    if match:
        height = float(match.group(1))
        print(f"Parsed height {height}m from filename: {filename}")
        return height
    else:
        # If no height found, try alternative patterns
        alt_patterns = [
            r'(\d+\.?\d*)m',  # patterns like "1.0m", "0.5m"
            r'height_?(\d+\.?\d*)',  # patterns like "height_1.0"
            r'h_(\d+\.?\d*)'  # patterns like "h_0.4"
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                height = float(match.group(1))
                print(f"Parsed height {height}m from filename using alternative pattern: {filename}")
                return height
        
        # Default fallback - use Config.cylinder_length instead of hardcoded value
        print(f"Could not parse height from filename '{filename}', using Config.cylinder_length")
        return 1.0  # This should be Config.cylinder_length in real usage


# =====================
# FIXED PLOTTING FUNCTIONS (WITH HORIZON AWARENESS)
# =====================

def plot_unscaled_training_curves(train_history, output_dir, best_epoch, horizon_steps=1):
    """Plot training curves showing both scaled and unscaled metrics (horizon-agnostic)."""
    plt.style.use('default')
    horizon_label = f"{horizon_steps} step{'s' if horizon_steps != 1 else ''}"
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'Training Progress - Horizon: {horizon_label} (PyTorch)', fontsize=16)
    
    epochs = range(1, len(train_history) + 1)
    
    # Loss curves (scaled)
    axes[0, 0].plot(epochs, [h['train_loss'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, [h['val_loss'] for h in train_history], 'r-', label='Validation', linewidth=2)
    axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_epoch})')
    axes[0, 0].set_title(f'Total Loss (Scaled) - H={horizon_steps}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curves (scaled)
    axes[0, 1].plot(epochs, [h['train_mae'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, [h['val_mae'] for h in train_history], 'r-', label='Validation', linewidth=2)
    axes[0, 1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    axes[0, 1].set_title(f'MAE (Scaled) - H={horizon_steps}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (Scaled)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE curves (unscaled) - MOST IMPORTANT
    axes[1, 0].plot(epochs, [h['train_mae_unscaled'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, [h['val_mae_unscaled'] for h in train_history], 'r-', label='Validation', linewidth=2)
    axes[1, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    axes[1, 0].set_title(f'MAE (Unscaled) - Horizon: {horizon_label}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (K or °C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMSE curves (unscaled) - MOST IMPORTANT
    axes[1, 1].plot(epochs, [h['train_rmse_unscaled'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, [h['val_rmse_unscaled'] for h in train_history], 'r-', label='Validation', linewidth=2)
    axes[1, 1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    axes[1, 1].set_title(f'RMSE (Unscaled) - Horizon: {horizon_label}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE (K or °C)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Physics loss curves
    axes[2, 0].plot(epochs, [h['train_physics_loss'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[2, 0].plot(epochs, [h['val_physics_loss'] for h in train_history], 'r-', label='Validation', linewidth=2)
    axes[2, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    axes[2, 0].set_title(f'Physics Loss - H={horizon_steps}')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Physics Loss')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Combined constraint losses using actual available keys
    train_combined_constraint = [
        h.get('train_soft_penalty', 0.0) + h.get('train_excess_penalty', 0.0) + h.get('train_power_balance_loss', 0.0)
        for h in train_history
    ]
    val_combined_constraint = [
        h.get('val_soft_penalty', 0.0) + h.get('val_excess_penalty', 0.0) + h.get('val_power_balance_loss', 0.0)
        for h in train_history
    ]
    
    axes[2, 1].plot(epochs, train_combined_constraint, 'b-', label='Train', linewidth=2)
    axes[2, 1].plot(epochs, val_combined_constraint, 'r-', label='Validation', linewidth=2)
    axes[2, 1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    axes[2, 1].set_title(f'Combined Constraint Losses - H={horizon_steps}')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Constraint Loss')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_H{horizon_steps}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved (horizon = {horizon_label})")


# =====================
# NEW: CONFIG UPDATE HELPER AND SINGLE EXPERIMENT RUNNER
# =====================

def update_config_for(L: int, H: int) -> None:
    """
    Helper to safely update Config for each run with new L and H values.
    """
    Config.sequence_length = L
    Config.prediction_horizon_steps = H
    
    # Recompute class attributes
    Config.output_dir = f"output/{Config.experiment_name}_L{L}_H{H}"
    Config.run_tag = f"{Config.experiment_name}_L{L}_H{H}"
    
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    print(f"Config updated: L={L}, H={H}")
    print(f"   Output dir: {Config.output_dir}")
    print(f"   Run tag: {Config.run_tag}")


def run_single_experiment(L: int, H: int, seed: int = 42) -> dict:
    """
    Run a single experiment with specified sequence length L and horizon H.
    Returns a flat dictionary with key metrics for sweep analysis.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING SINGLE EXPERIMENT: L={L}, H={H}")
    print(f"{'='*80}")
    
    # Update config for this run
    update_config_for(L, H)
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"\nLoading datasets for L={L}, H={H}...")
    # Initialize to avoid UnboundLocalError in except cleanup
    train_loader = val_loader = test_loader = train_dataset = None
    try:
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
            data_dir=Config.data_dir,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
            sequence_length=Config.sequence_length,
            prediction_horizon=Config.prediction_horizon_steps,
            scaler_dir=Config.scaler_dir
        )
        
        # Get scalers from dataset
        physics_params = train_dataset.get_physics_params()
        thermal_scaler = physics_params['thermal_scaler']
        param_scaler = physics_params['param_scaler']
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "mae": float('inf'), "rmse": float('inf'), "r2": -float('inf'),
            "physics_loss": float('inf'), "pb_loss": float('inf'), "viol_pct": 100.0,
            "pred_to_in_ratio": float('inf'), "status": "data_load_failed"
        }
    
    # Build model and trainer
    print(f"Building model for L={L}, H={H}...")
    try:
        model = build_model(
            num_sensors=10,
            sequence_length=Config.sequence_length,
            lstm_units=Config.lstm_units,
            dropout_rate=Config.dropout_rate,
            device=device
        )
        
        base_trainer = create_trainer(
            model=model,
            physics_weight=Config.physics_weight,
            soft_penalty_weight=Config.soft_penalty_weight,
            excess_penalty_weight=Config.excess_penalty_weight,
            power_balance_weight=Config.power_balance_weight,
            learning_rate=Config.learning_rate,
            lstm_units=Config.lstm_units,
            dropout_rate=Config.dropout_rate,
            device=device,
            #cylinder_length=Config.cylinder_length,
            #param_scaler=param_scaler,
            thermal_scaler=thermal_scaler
        )
        
        trainer = FixedUnscaledEvaluationTrainer(
            base_trainer, thermal_scaler, param_scaler, 
            horizon_steps=Config.prediction_horizon_steps, device=device
        )
        
    except Exception as e:
        print(f"Error building model: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "mae": float('inf'), "rmse": float('inf'), "r2": -float('inf'),
            "physics_loss": float('inf'), "pb_loss": float('inf'), "viol_pct": 100.0,
            "pred_to_in_ratio": float('inf'), "status": "model_build_failed"
        }
    
    # Training loop
    print(f"\nTraining model for L={L}, H={H}...")
    best_val_loss = np.inf
    best_val_mae_unscaled = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    train_history = []
    
    # Fix: Use a unique filename with timestamp to avoid directory conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(Config.output_dir, f'best_model_L{L}_H{H}_{timestamp}.pth')
    
    # Ensure the directory exists and is clean
    if not os.path.exists(Config.output_dir):
        os.makedirs(Config.output_dir, exist_ok=True)
    
    # Remove any existing directory with the same name as our intended file
    old_model_path = os.path.join(Config.output_dir, 'model_state_dict.pth')
    if os.path.isdir(old_model_path):
        import shutil
        print(f"Removing conflicting directory: {old_model_path}")
        shutil.rmtree(old_model_path)
    
    try:
        for epoch in range(Config.max_epochs):
            # Train and validate
            results = trainer.train_epoch_unscaled(train_loader, val_loader)
            train_history.append(results)
            
            # Early stopping based on unscaled validation MAE
            val_mae_unscaled = results['val_mae_unscaled']
            
            if val_mae_unscaled < best_val_mae_unscaled:
                best_val_loss = results['val_loss']
                best_val_mae_unscaled = val_mae_unscaled
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # FIXED: Save best model as actual file (not directory)
                try:
                    torch.save(trainer.model.state_dict(), best_model_path)
                    print(f"Saved best weights to: {best_model_path}")
                except Exception as save_error:
                    print(f"Warning: Could not save model: {save_error}")
                
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= Config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
    except Exception as e:
        print(f"Error during training: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "mae": float('inf'), "rmse": float('inf'), "r2": -float('inf'),
            "physics_loss": float('inf'), "pb_loss": float('inf'), "viol_pct": 100.0,
            "pred_to_in_ratio": float('inf'), "status": "training_failed"
        }
    
    # Load best model and evaluate
    print(f"\nEvaluating best model for L={L}, H={H}...")
    try:
        # FIXED: Fallback if directory was created instead of file
        if os.path.isdir(best_model_path):
            cand = os.path.join(best_model_path, "model_state_dict.pth")
            if os.path.isfile(cand):
                best_model_path = cand
        
        # Load best model if it exists and is a file (not directory)
        if os.path.exists(best_model_path) and os.path.isfile(best_model_path):
            print(f"Loading model from: {best_model_path}")
            trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print(f"Best model file not found or not a file, using current model state")
        
        # Test evaluation
        test_results = trainer.evaluate_unscaled(test_loader, "test")
        
        # Power balance analysis
        power_summary = trainer.analyze_power_balance(test_loader, num_samples=500)
        
        # Extract key metrics
        mae = test_results['test_mae_unscaled']
        rmse = test_results['test_rmse_unscaled']
        r2 = test_results['test_r2_overall_unscaled']
        physics_loss = test_results['test_physics_loss']
        pb_loss = test_results['test_power_balance_loss']
        viol_pct = power_summary['conservation_violations']['percentage']
        pred_to_in_ratio = power_summary['mean_predicted_to_incoming_ratio']
        
        print(f"Experiment L={L}, H={H} completed successfully")
        print(f"   MAE: {mae:.2f} °C, Violation: {viol_pct:.1f}%, Ratio: {pred_to_in_ratio:.3f}")
        
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L,
            "H": H,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "physics_loss": float(physics_loss),
            "pb_loss": float(pb_loss),
            "viol_pct": float(viol_pct),
            "pred_to_in_ratio": float(pred_to_in_ratio),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "mae": float('inf'), "rmse": float('inf'), "r2": -float('inf'),
            "physics_loss": float('inf'), "pb_loss": float('inf'), "viol_pct": 100.0,
            "pred_to_in_ratio": float('inf'), "status": "evaluation_failed"
        }


# =====================
# NEW: HORIZON SWEEP IMPLEMENTATION
# =====================

def horizon_sweep_fixed_seq(seq_len: int = 20,
                          horizons: list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
                          mae_thresh: float = 3.0,
                          viol_thresh_pct: float = 1.0,
                          ratio_thresh: float = 1.01,
                          pb_mult_baseline: float = 3.0) -> None:
    """
    Run horizon sweep with fixed sequence length, evaluating accuracy and physics constraints.
    """
    print(f"\n{'='*100}")
    print(f"HORIZON SWEEP: Fixed Sequence Length L={seq_len}")
    print(f"{'='*100}")
    print(f"Horizons to test: {horizons}")
    print(f"Thresholds: MAE≤{mae_thresh}°C, Violations≤{viol_thresh_pct}%, Ratio≤{ratio_thresh}, PB≤{pb_mult_baseline}x baseline")
    
    # Create sweep output directory
    sweep_dir = "output/sweeps_experiments"
    os.makedirs(sweep_dir, exist_ok=True)
    
    results = []
    pb_baseline = None
    
    # Run experiments for each horizon
    for i, H in enumerate(horizons):
        print(f"\nRunning experiment {i+1}/{len(horizons)}: L={seq_len}, H={H}")
        
        # Run single experiment
        result = run_single_experiment(seq_len, H, seed=42)
        
        # Store baseline power balance loss from H=1
        if H == 1 and result['status'] == 'success':
            pb_baseline = result['pb_loss']
            print(f"Power balance baseline (H=1): {pb_baseline:.6f}")
        
        # Determine pass/fail status
        if result['status'] == 'success':
            mae_pass = result['mae'] <= mae_thresh
            viol_pass = result['viol_pct'] <= viol_thresh_pct
            ratio_pass = result['pred_to_in_ratio'] <= ratio_thresh
            
            # Power balance check (only if we have baseline)
            if pb_baseline is not None:
                pb_pass = result['pb_loss'] <= pb_mult_baseline * pb_baseline
            else:
                pb_pass = True  # Can't check without baseline
            
            overall_pass = mae_pass and viol_pass and ratio_pass and pb_pass
            
            print(f"   Results: MAE={result['mae']:.2f}°C {'✅' if mae_pass else '❌'}, "
                  f"Viol={result['viol_pct']:.1f}% {'✅' if viol_pass else '❌'}, "
                  f"Ratio={result['pred_to_in_ratio']:.3f} {'✅' if ratio_pass else '❌'}, "
                  f"PB={result['pb_loss']:.6f} {'✅' if pb_pass else '❌'}")
            print(f"   Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
            
        else:
            overall_pass = False
            print(f"   Experiment failed: {result['status']}")
        
        # Add pass/fail info to result
        result['pass'] = overall_pass
        results.append(result)
    
    # Determine maximum viable horizon
    max_viable_H = None
    for result in results:
        if result['pass']:
            max_viable_H = result['H']
        else:
            break  # Stop at first failure since we test in ascending order
    
    print(f"\nHORIZON SWEEP COMPLETED")
    print(f"   Maximum viable horizon: H={max_viable_H if max_viable_H is not None else 'None'}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(sweep_dir, f"seq{seq_len}_horizon_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save summary JSON
    summary = {
        "sequence_length": seq_len,
        "horizons_tested": horizons,
        "max_viable_H": max_viable_H,
        "thresholds": {
            "mae_thresh": mae_thresh,
            "viol_thresh_pct": viol_thresh_pct,
            "ratio_thresh": ratio_thresh,
            "pb_mult_baseline": pb_mult_baseline
        },
        "pb_baseline": pb_baseline,
        "per_horizon_results": [
            {
                "H": r['H'],
                "pass": r['pass'],
                "mae": r['mae'] if r['mae'] != float('inf') else None,
                "viol_pct": r['viol_pct'] if r['viol_pct'] != 100.0 else None,
                "pred_to_in_ratio": r['pred_to_in_ratio'] if r['pred_to_in_ratio'] != float('inf') else None
            }
            for r in results
        ]
    }
    
    json_path = os.path.join(sweep_dir, f"seq{seq_len}_horizon_sweep_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")
    
    # Generate plots
    print(f"\nGenerating sweep plots...")
    generate_sweep_plots(results, sweep_dir, seq_len, mae_thresh, viol_thresh_pct, ratio_thresh, pb_baseline, pb_mult_baseline)
    
    return results


def generate_sweep_plots(results, sweep_dir, seq_len, mae_thresh, viol_thresh_pct, ratio_thresh, pb_baseline=None, pb_mult_baseline=3.0):
    """
    Generate and save the 4 required sweep plots.
    """
    # Filter successful results for plotting
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    horizons = [r['H'] for r in successful_results]
    maes = [r['mae'] for r in successful_results]
    viols = [r['viol_pct'] for r in successful_results]
    ratios = [r['pred_to_in_ratio'] for r in successful_results]
    pb_losses = [r['pb_loss'] for r in successful_results]
    passes = [r['pass'] for r in successful_results]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Horizon Sweep Results - Sequence Length L={seq_len}', fontsize=16)
    
    # 1. MAE vs H
    colors = ['green' if p else 'red' for p in passes]
    axes[0, 0].scatter(horizons, maes, c=colors, s=80, alpha=0.7, edgecolors='black')
    axes[0, 0].axhline(y=mae_thresh, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({mae_thresh}°C)')
    axes[0, 0].set_xlabel('Horizon Steps (H)')
    axes[0, 0].set_ylabel('MAE (°C)')
    axes[0, 0].set_title('MAE vs Horizon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Violation % vs H
    colors = ['green' if p else 'red' for p in passes]
    axes[0, 1].scatter(horizons, viols, c=colors, s=80, alpha=0.7, edgecolors='black')
    axes[0, 1].axhline(y=viol_thresh_pct, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({viol_thresh_pct}%)')
    axes[0, 1].set_xlabel('Horizon Steps (H)')
    axes[0, 1].set_ylabel('Violation Percentage (%)')
    axes[0, 1].set_title('Energy Conservation Violations vs Horizon')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Pred/Incoming ratio vs H
    colors = ['green' if p else 'red' for p in passes]
    axes[1, 0].scatter(horizons, ratios, c=colors, s=80, alpha=0.7, edgecolors='black')
    axes[1, 0].axhline(y=ratio_thresh, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({ratio_thresh})')
    axes[1, 0].axhline(y=1.0, color='blue', linestyle=':', alpha=0.6, label='Perfect Balance')
    axes[1, 0].set_xlabel('Horizon Steps (H)')
    axes[1, 0].set_ylabel('Predicted/Incoming Ratio')
    axes[1, 0].set_title('Power Balance Ratio vs Horizon')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Power Balance Loss vs H
    colors = ['green' if p else 'red' for p in passes]
    axes[1, 1].scatter(horizons, pb_losses, c=colors, s=80, alpha=0.7, edgecolors='black')
    if pb_baseline is not None:
        axes[1, 1].axhline(y=pb_mult_baseline * pb_baseline, color='red', linestyle='--', alpha=0.8, 
                          label=f'Threshold ({pb_mult_baseline}x baseline)')
    axes[1, 1].set_xlabel('Horizon Steps (H)')
    axes[1, 1].set_ylabel('Power Balance Loss')
    axes[1, 1].set_title('Power Balance Loss vs Horizon')
    axes[1, 1].set_yscale('log')  # Log scale for better visualization
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save individual plots
    mae_plot_path = os.path.join(sweep_dir, f"seq{seq_len}_mae_vs_h.png")
    viol_plot_path = os.path.join(sweep_dir, f"seq{seq_len}_viol_vs_h.png")
    ratio_plot_path = os.path.join(sweep_dir, f"seq{seq_len}_ratio_vs_h.png")
    pb_plot_path = os.path.join(sweep_dir, f"seq{seq_len}_pbloss_vs_h.png")
    
    # Save combined plot
    combined_path = os.path.join(sweep_dir, f"seq{seq_len}_horizon_sweep_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    
    # Save individual plots by extracting each subplot
    # MAE plot
    fig_mae, ax_mae = plt.subplots(figsize=(8, 6))
    ax_mae.scatter(horizons, maes, c=colors, s=80, alpha=0.7, edgecolors='black')
    ax_mae.axhline(y=mae_thresh, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({mae_thresh}°C)')
    ax_mae.set_xlabel('Horizon Steps (H)')
    ax_mae.set_ylabel('MAE (°C)')
    ax_mae.set_title(f'MAE vs Horizon (L={seq_len})')
    ax_mae.legend()
    ax_mae.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mae_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Violation plot
    fig_viol, ax_viol = plt.subplots(figsize=(8, 6))
    ax_viol.scatter(horizons, viols, c=colors, s=80, alpha=0.7, edgecolors='black')
    ax_viol.axhline(y=viol_thresh_pct, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({viol_thresh_pct}%)')
    ax_viol.set_xlabel('Horizon Steps (H)')
    ax_viol.set_ylabel('Violation Percentage (%)')
    ax_viol.set_title(f'Energy Conservation Violations vs Horizon (L={seq_len})')
    ax_viol.legend()
    ax_viol.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viol_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Ratio plot
    fig_ratio, ax_ratio = plt.subplots(figsize=(8, 6))
    ax_ratio.scatter(horizons, ratios, c=colors, s=80, alpha=0.7, edgecolors='black')
    ax_ratio.axhline(y=ratio_thresh, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({ratio_thresh})')
    ax_ratio.axhline(y=1.0, color='blue', linestyle=':', alpha=0.6, label='Perfect Balance')
    ax_ratio.set_xlabel('Horizon Steps (H)')
    ax_ratio.set_ylabel('Predicted/Incoming Ratio')
    ax_ratio.set_title(f'Power Balance Ratio vs Horizon (L={seq_len})')
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ratio_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Power balance loss plot
    fig_pb, ax_pb = plt.subplots(figsize=(8, 6))
    ax_pb.scatter(horizons, pb_losses, c=colors, s=80, alpha=0.7, edgecolors='black')
    if pb_baseline is not None:
        ax_pb.axhline(y=pb_mult_baseline * pb_baseline, color='red', linestyle='--', alpha=0.8, 
                      label=f'Threshold ({pb_mult_baseline}x baseline)')
    ax_pb.set_xlabel('Horizon Steps (H)')
    ax_pb.set_ylabel('Power Balance Loss')
    ax_pb.set_title(f'Power Balance Loss vs Horizon (L={seq_len})')
    ax_pb.set_yscale('log')
    ax_pb.legend()
    ax_pb.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pb_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved:")
    print(f"   Combined: {combined_path}")
    print(f"   MAE: {mae_plot_path}")
    print(f"   Violations: {viol_plot_path}")
    print(f"   Ratios: {ratio_plot_path}")
    print(f"   Power Balance: {pb_plot_path}")


# =====================
# REMAINING FUNCTIONS FROM ORIGINAL SCRIPT
# =====================

def generate_overall_statistics_summary(test_results, error_summary, power_summary, conservation_status, output_dir, horizon_steps=1):
    """
    Overall Statistics Summary (horizon-agnostic).
    """
    horizon_label = f"{horizon_steps} step{'s' if horizon_steps != 1 else ''}"
    
    overall_stats = {
        'horizon_steps': horizon_steps,
        'horizon_label': horizon_label,
        'temperature_accuracy': {
            'mae_overall': test_results['test_mae_unscaled'],
            'rmse_overall': test_results['test_rmse_unscaled'],
            'r2_overall': test_results['test_r2_overall_unscaled'],
            'mean_mae_across_sensors': error_summary['overall_statistics']['mean_mae_across_sensors'],
            'mean_rmse_across_sensors': error_summary['overall_statistics']['mean_rmse_across_sensors'],
            'mean_r2_across_sensors': error_summary['overall_statistics']['mean_r2_across_sensors'],
            'max_error_across_all': error_summary['overall_statistics']['max_error_across_all']
        },
        'physics_losses': {
            'physics_loss': test_results['test_physics_loss'],
            'constraint_loss': test_results.get('test_constraint_loss', 0.0),
            'power_balance_loss': test_results['test_power_balance_loss']
        },
        'power_balance': power_summary,
        'energy_conservation': conservation_status
    }
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Overall Statistics Summary (Horizon: {horizon_label})', fontsize=16)
    
    # Temperature accuracy metrics
    accuracy_metrics = ['MAE', 'RMSE', 'R²']
    accuracy_values = [
        overall_stats['temperature_accuracy']['mae_overall'],
        overall_stats['temperature_accuracy']['rmse_overall'],
        overall_stats['temperature_accuracy']['r2_overall']
    ]
    
    bars1 = axes[0, 0].bar(accuracy_metrics, accuracy_values, color=['skyblue', 'lightcoral', 'lightgreen'], 
                          alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'Temperature Accuracy Metrics (H={horizon_steps})')
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars1, accuracy_values):
        height = bar.get_height()
        axes[0, 0].annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    # Physics losses
    physics_metrics = ['Physics', 'Constraint', 'Power Balance']
    physics_values = [
        overall_stats['physics_losses']['physics_loss'],
        overall_stats['physics_losses']['constraint_loss'],
        overall_stats['physics_losses']['power_balance_loss']
    ]
    
    bars2 = axes[0, 1].bar(physics_metrics, physics_values, color=['gold', 'orange', 'tomato'], 
                          alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f'Physics Loss Components (H={horizon_steps})')
    axes[0, 1].set_ylabel('Loss Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars2, physics_values):
        height = bar.get_height()
        axes[0, 1].annotate(f'{value:.6f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    
    # Power balance ratios
    power_ratios = [
        overall_stats['power_balance']['mean_actual_to_incoming_ratio'],
        overall_stats['power_balance']['mean_predicted_to_incoming_ratio']
    ]
    ratio_labels = ['Actual/Incoming', 'Predicted/Incoming']
    
    bars3 = axes[1, 0].bar(ratio_labels, power_ratios, color=['blue', 'red'], alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.8, label='Perfect Balance')
    axes[1, 0].set_title(f'Power Balance Ratios (H={horizon_steps})')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars3, power_ratios):
        height = bar.get_height()
        axes[1, 0].annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    # Conservation status
    conservation_data = [
        100 - overall_stats['energy_conservation']['violation_percentage'],
        overall_stats['energy_conservation']['violation_percentage']
    ]
    conservation_labels = ['Conserved', 'Violated']
    
    bars4 = axes[1, 1].bar(conservation_labels, conservation_data, color=['green', 'red'], 
                          alpha=0.7, edgecolor='black')
    axes[1, 1].set_title(f'Energy Conservation Status (H={horizon_steps})')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add percentage values on bars
    for bar, value in zip(bars4, conservation_data):
        height = bar.get_height()
        axes[1, 1].annotate(f'{value:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'overall_statistics_summary_H{horizon_steps}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive statistics
    with open(os.path.join(output_dir, f'overall_statistics_summary_H{horizon_steps}.json'), 'w') as f:
        json.dump(overall_stats, f, indent=2, default=str)
    
    print(f"Overall statistics summary completed (horizon = {horizon_label})")
    
    # Print summary to console
    print(f"\nOVERALL STATISTICS SUMMARY (HORIZON = {horizon_label}):")
    print(f"   Temperature Accuracy:")
    print(f"     MAE: {overall_stats['temperature_accuracy']['mae_overall']:.2f} °C")
    print(f"     RMSE: {overall_stats['temperature_accuracy']['rmse_overall']:.2f} °C")
    print(f"     R²: {overall_stats['temperature_accuracy']['r2_overall']:.6f}")
    print(f"   Physics Losses:")
    print(f"     Physics Loss: {overall_stats['physics_losses']['physics_loss']:.6f}")
    print(f"     Constraint Loss: {overall_stats['physics_losses']['constraint_loss']:.6f}")
    print(f"     Power Balance Loss: {overall_stats['physics_losses']['power_balance_loss']:.6f}")
    print(f"   Power Balance:")
    print(f"     Actual/Incoming Ratio: {overall_stats['power_balance']['mean_actual_to_incoming_ratio']:.3f}")
    print(f"     Predicted/Incoming Ratio: {overall_stats['power_balance']['mean_predicted_to_incoming_ratio']:.3f}")
    print(f"   Energy Conservation:")
    print(f"     Conservation Violated: {overall_stats['energy_conservation']['conservation_violated']}")
    print(f"     Violation Percentage: {overall_stats['energy_conservation']['violation_percentage']:.1f}%")
    
    return overall_stats


def analyze_energy_conservation_status(test_results, power_summary, horizon_steps=1):
    """
    Energy Conservation Status (horizon-agnostic).
    """
    horizon_label = f"{horizon_steps} step{'s' if horizon_steps != 1 else ''}"
    
    conservation_status = {
        'horizon_steps': horizon_steps,
        'horizon_label': horizon_label,
        'conservation_violated': power_summary['conservation_violations']['count'] > 0,
        'violation_count': power_summary['conservation_violations']['count'],
        'violation_percentage': power_summary['conservation_violations']['percentage'],
        'mean_violation_amount': power_summary['conservation_violations']['mean_violation_amount']
    }
    
    print(f"\nENERGY CONSERVATION STATUS (HORIZON = {horizon_label}):")
    print(f"   Conservation Violated: {conservation_status['conservation_violated']}")
    print(f"   Violation Count: {conservation_status['violation_count']}")
    print(f"   Violation Percentage: {conservation_status['violation_percentage']:.1f}%")
    if conservation_status['conservation_violated']:
        print(f"   Mean Violation Amount: {conservation_status['mean_violation_amount']:.2f} W")
    
    return conservation_status


def print_final_summary_fixed(best_epoch, best_val_mae_unscaled, best_val_loss, test_results, output_dir, horizon_steps=1):
    """Print comprehensive final summary (horizon-agnostic)."""
    horizon_label = f"{horizon_steps} step{'s' if horizon_steps != 1 else ''}"
    
    print("\n" + "="*80)
    print(f"FIXED HORIZON-AGNOSTIC PYTORCH TRAINING COMPLETE - HORIZON: {horizon_label}")
    print("="*80)
    
    print(f"All outputs saved to: {output_dir}")
    
    print(f"\nBEST TRAINING PERFORMANCE (Epoch {best_epoch}):")
    print(f"   Validation MAE (unscaled): {best_val_mae_unscaled:.2f} K (or °C)")
    print(f"   Validation Loss (scaled):  {best_val_loss:.6f}")
    print(f"   Prediction horizon: {horizon_label}")
    
    print(f"\nFINAL TEST SET PERFORMANCE (HORIZON: {horizon_label}):")
    print(f"   MAE:  {test_results['test_mae_unscaled']:.2f} K (or °C)")
    print(f"   RMSE: {test_results['test_rmse_unscaled']:.2f} K (or °C)")
    print(f"   R²:   {test_results['test_r2_overall_unscaled']:.6f}")
    
    print(f"\nPHYSICS COMPONENTS (HORIZON: {horizon_label}):")
    print(f"   Physics Loss:       {test_results['test_physics_loss']:.6f}")
    print(f"   Constraint Loss:    {test_results.get('test_constraint_loss', 0.0):.6f}")
    print(f"   Power Balance Loss: {test_results['test_power_balance_loss']:.6f}")
    
    print(f"\nTEMPERATURE DATA ANALYSIS:")
    y_true_temps = test_results['predictions_unscaled']['y_true']
    y_pred_temps = test_results['predictions_unscaled']['y_pred']
    print(f"   True temperature range:      {y_true_temps.min():.1f} to {y_true_temps.max():.1f}")
    print(f"   Predicted temperature range: {y_pred_temps.min():.1f} to {y_pred_temps.max():.1f}")
    print(f"   Mean true temperature:       {y_true_temps.mean():.1f}")
    print(f"   Mean predicted temperature:  {y_pred_temps.mean():.1f}")
    
    avg_temp = y_true_temps.mean()
    if avg_temp < 100:
        print(f"   Data appears to be in Celsius (avg: {avg_temp:.1f}°C)")
    elif 250 < avg_temp < 400:
        print(f"   Data appears to be in Kelvin (avg: {avg_temp:.1f}K)")
    else:
        print(f"   Unusual temperature range - please verify units")
    
    print("\n" + "="*80)
    print(f"SUCCESS: FIXED HORIZON-AGNOSTIC VERSION - HORIZON: {horizon_label}")
    print("="*80)


# =====================
# Updated Configuration Settings (FIXED - HORIZON-AGNOSTIC)
# =====================
class Config:
    # Data and model settings
    data_dir = "data/processed_H6"
    scaler_dir = "models_new_theoretical_30sec"
    batch_size = 32
    learning_rate = 0.001
    max_epochs = 100
    patience = 10
    lstm_units = 64
    dropout_rate = 0.2
    
    # Physics loss weights
    physics_weight = 0.001
    soft_penalty_weight = 0.001
    excess_penalty_weight = 0.001
    power_balance_weight = 0.0005
    
    # Temporal settings (SINGLE SOURCE OF TRUTH)
    sequence_length = 20
    prediction_horizon_steps = 1  # CHANGE ONLY THIS VALUE TO SET HORIZON
    
    # Physical parameters
    cylinder_length = 1.0
    num_workers = 0  # Set to 0 for memory management
    
    # Experiment settings
    experiment_name = "new_theoretical_30sec"

# Compute derived attributes as class attributes (not properties)
Config.output_dir = f"output/{Config.experiment_name}_H{Config.prediction_horizon_steps}"
Config.run_tag = f"{Config.experiment_name}_H{Config.prediction_horizon_steps}"

# Sanity check for horizon
assert Config.prediction_horizon_steps >= 1, "Prediction horizon must be >= 1"

# Create output directory
os.makedirs(Config.output_dir, exist_ok=True)


# =====================
# ORIGINAL MAIN FUNCTION (FOR SINGLE RUNS) - FIXED WEIGHT INITIALIZATION
# =====================

def main():
    """Original main function for single experiment runs."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Horizon configuration display
    horizon_label = f"{Config.prediction_horizon_steps} step{'s' if Config.prediction_horizon_steps != 1 else ''}"
    print(f"\nFULLY FIXED HORIZON-AGNOSTIC CONFIGURATION:")
    print(f"   Prediction horizon: {horizon_label}")
    print(f"   Sequence length: {Config.sequence_length}")
    print(f"   Output directory: {Config.output_dir}")
    print(f"   Run tag: {Config.run_tag}")
    
    # Load Dataset
    print("\nLoading datasets...")
    # Initialize first (helps if you later wrap this in try/except)
    train_loader = val_loader = test_loader = train_dataset = None
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
        data_dir=Config.data_dir,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        sequence_length=Config.sequence_length,
        prediction_horizon=Config.prediction_horizon_steps,
        scaler_dir=Config.scaler_dir
    )
    
    # Get scalers from dataset
    physics_params = train_dataset.get_physics_params()
    thermal_scaler = physics_params['thermal_scaler']
    param_scaler = physics_params['param_scaler']
    
    print(f"\nSCALER INFORMATION:")
    print(f"Thermal scaler - Mean: {thermal_scaler.mean_[:3]}... (10 sensors)")
    print(f"Thermal scaler - Scale: {thermal_scaler.scale_[:3]}... (10 sensors)")
    
    # Build Model with Fixed Evaluation Wrapper
    print(f"Building model with FULLY FIXED power metadata extraction (horizon = {horizon_label})...")
    model = build_model(
        num_sensors=10,
        sequence_length=Config.sequence_length,
        lstm_units=Config.lstm_units,
        dropout_rate=Config.dropout_rate,
        device=device
    )
    
    base_trainer = create_trainer(
        model=model,
        physics_weight=Config.physics_weight,
        soft_penalty_weight=Config.soft_penalty_weight,
        excess_penalty_weight=Config.excess_penalty_weight,
        power_balance_weight=Config.power_balance_weight,
        learning_rate=Config.learning_rate,
        lstm_units=Config.lstm_units,
        dropout_rate=Config.dropout_rate,
        device=device,
        #cylinder_length=Config.cylinder_length,
        #param_scaler=param_scaler,
        thermal_scaler=thermal_scaler
    )
    
    # Wrap with FULLY FIXED unscaled evaluation trainer
    trainer = FixedUnscaledEvaluationTrainer(
        base_trainer, 
        thermal_scaler, 
        param_scaler, 
        horizon_steps=Config.prediction_horizon_steps,
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built with {total_params:,} parameters")
    
    print("\n" + "="*80)
    print(f"FULLY FIXED HORIZON-AGNOSTIC VERSION - HORIZON: {horizon_label}")
    print("="*80)
    
    # Test the FULLY FIXED power metadata extraction
    print(f"\nTESTING FULLY FIXED POWER METADATA EXTRACTION...")
    try:
        # Get a test batch
        test_batch = next(iter(train_loader))
        time_series, static_params, targets, original_power_data = test_batch
        
        # Test metadata extraction with horizon awareness
        extracted_metadata = extract_power_metadata_from_batch(
            time_series, static_params, targets, thermal_scaler, param_scaler, Config.prediction_horizon_steps
        )
        
        print(f"Successfully extracted metadata for {len(extracted_metadata)} samples")
        
        if extracted_metadata:
            sample = extracted_metadata[0]
            print(f"Sample metadata structure (FULLY FIXED):")
            print(f"   - temps_row1: {len(sample['temps_row1'])} temperatures")
            print(f"   - temps_target: {len(sample['temps_target'])} temperatures")
            print(f"   - time_diff: {sample['time_diff']:.2f}")
            print(f"   - horizon_steps: {sample['horizon_steps']}")
            print(f"   - h (HTC): {sample['h']:.2f} W/m²·K")
            print(f"   - q0 (heat flux): {sample['q0']:.2f} W/m²")
            print(f"   - abs_coeff: {sample['abs_coeff']:.2f}")
            print(f"   - surf_frac: {sample['surf_frac']:.2f}")
            
        print(f"Fully fixed power metadata extraction working correctly!")
        
    except Exception as e:
        print(f"Error testing power metadata extraction: {e}")
        return
    
    # Training Loop
    print(f"\nStarting training with FULLY FIXED power metadata extraction (horizon = {horizon_label})...")
    print("="*80)
    
    best_val_loss = np.inf
    best_val_mae_unscaled = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    train_history = []
    
    log_dir = os.path.join(Config.output_dir, "logs")
    tensorboard_writer = SummaryWriter(log_dir)
    
    best_model_path = os.path.join(Config.output_dir, 'model_state_dict.pth')
    
    for epoch in range(Config.max_epochs):
        epoch_start_time = datetime.now()
        
        print(f"\nEpoch {epoch+1}/{Config.max_epochs}")
        print("-" * 60)
        
        # Train and Validate with fixed metadata extraction
        results = trainer.train_epoch_unscaled(train_loader, val_loader)
        
        # Logging to TensorBoard
        for key, value in results.items():
            tensorboard_writer.add_scalar(key, value, epoch)
        
        train_history.append(results)
        
        # Epoch Summary
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        print(f"EPOCH {epoch+1} SUMMARY (FIXED HORIZON: {horizon_label})")
        print(f"   Duration: {epoch_duration:.1f}s")
        print(f"   Training   - Loss: {results['train_loss']:.6f}, Scaled MAE: {results['train_mae']:.6f}")
        print(f"   Training   - UNSCALED MAE: {results['train_mae_unscaled']:.2f}, UNSCALED RMSE: {results['train_rmse_unscaled']:.2f}")
        print(f"   Validation - Loss: {results['val_loss']:.6f}, Scaled MAE: {results['val_mae']:.6f}")
        print(f"   Validation - UNSCALED MAE: {results['val_mae_unscaled']:.2f}, UNSCALED RMSE: {results['val_rmse_unscaled']:.2f}")
        
        # Physics components
        print(f"   Physics Components (FIXED HORIZON: {horizon_label}):")
        print(f"     Train - Physics: {results['train_physics_loss']:.6f}, Soft: {results.get('train_soft_penalty', 0.0):.6f}, Excess: {results.get('train_excess_penalty', 0.0):.6f}, Power: {results.get('train_power_balance_loss', 0.0):.6f}")
        print(f"     Val   - Physics: {results['val_physics_loss']:.6f}, Soft: {results.get('val_soft_penalty', 0.0):.6f}, Excess: {results.get('val_excess_penalty', 0.0):.6f}, Power: {results.get('val_power_balance_loss', 0.0):.6f}")
        
        # Early Stopping based on unscaled validation MAE
        val_mae_unscaled = results['val_mae_unscaled']
        
        if val_mae_unscaled < best_val_mae_unscaled:
            best_val_loss = results['val_loss']
            best_val_mae_unscaled = val_mae_unscaled
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            print(f"   NEW BEST MODEL! (Based on unscaled validation MAE)")
            print(f"      Best Val MAE (unscaled): {best_val_mae_unscaled:.2f} K")
            print(f"      Corresponding Val Loss: {best_val_loss:.6f}")
            print(f"      Prediction horizon: {horizon_label}")
            
            # FIXED: Save best model as actual file (not directory)
            torch.save(trainer.model.state_dict(), best_model_path)
            
        else:
            epochs_without_improvement += 1
            print(f"   No improvement. Best unscaled MAE: {best_val_mae_unscaled:.2f} K (Epoch {best_epoch})")
            print(f"      Patience: {epochs_without_improvement}/{Config.patience}")
            
            if epochs_without_improvement >= Config.patience:
                print(f"\nEARLY STOPPING at Epoch {epoch+1}")
                print(f"   Best model was at Epoch {best_epoch} with Val MAE: {best_val_mae_unscaled:.2f} K")
                break
    
    tensorboard_writer.close()
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETED WITH FIXED HORIZON: {horizon_label}")
    print("="*80)
    print(f"Total Epochs: {len(train_history)}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation MAE (unscaled): {best_val_mae_unscaled:.2f} K")
    
    # TEST SET EVALUATION
    # FIXED: Fallback if directory was created instead of file
    if os.path.isdir(best_model_path):
        cand = os.path.join(best_model_path, "model_state_dict.pth")
        if os.path.isfile(cand):
            best_model_path = cand
    
    if os.path.exists(best_model_path) and os.path.isfile(best_model_path):
        print(f"\nLoading best model from: {best_model_path}")
        trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Best model not found at: {best_model_path}")
        print("Using current model state for evaluation")
    
    # Comprehensive test evaluation
    test_results = trainer.evaluate_unscaled(test_loader, "test")
    
    # Power balance analysis
    print(f"\nPOWER BALANCE ANALYSIS WITH FULLY FIXED EXTRACTED DATA:")
    try:
        power_summary = trainer.analyze_power_balance(test_loader, num_samples=500)
    except Exception as e:
        print(f"Power balance analysis encountered an issue: {e}")
        print("Using fallback power summary...")
        power_summary = {
            'horizon_steps': Config.prediction_horizon_steps,
            'horizon_label': horizon_label,
            'mean_actual_power': 0.0,
            'mean_predicted_power': 0.0,
            'mean_incoming_power': 0.0,
            'mean_actual_to_incoming_ratio': 0.0,
            'mean_predicted_to_incoming_ratio': 0.0,
            'conservation_violations': {'count': 0, 'percentage': 0.0, 'mean_violation_amount': 0.0}
        }
    
    # Save all results
    all_results = {
        'config': {
            'lstm_units': Config.lstm_units,
            'dropout_rate': Config.dropout_rate,
            'physics_weight': Config.physics_weight,
            'soft_penalty_weight': Config.soft_penalty_weight,
            'excess_penalty_weight': Config.excess_penalty_weight,
            'power_balance_weight': Config.power_balance_weight,
            'learning_rate': Config.learning_rate,
            'batch_size': Config.batch_size,
            'sequence_length': Config.sequence_length,
            'prediction_horizon_steps': Config.prediction_horizon_steps,
            'horizon_label': horizon_label,
            'device': str(device),
            'pytorch_version': torch.__version__,
            'power_metadata_source': 'fully_fixed_extracted_from_targets_batch',
            'filename_extraction': 'actual_from_dataset',
            'horizon_agnostic': True,
            'fixes_applied': [
                'use_targets_batch_not_last_input',
                'removed_training_arguments',
                'added_absorptivity_surface_fraction', 
                'fixed_physics_loss_call_signature',
                'replaced_constraint_with_soft_excess',
                'clarified_h_as_htc',
                'fixed_gpu_cpu_device_mismatch',
                'analyze_power_balance_returns_real_results',
                'removed_mock_power_balance_detailed',
                'fixed_config_properties_to_class_attributes',
                'made_power_ratios_numerically_safe',
                'fixed_model_save_load_path_consistency',
                'fixed_weight_initialization_in_place_operation',
                'fixed_tensor_float_conversion_with_as_float_helper',
                'added_multiprocessing_memory_management',
                'set_headless_matplotlib_backend',
                'removed_duplicate_torch_import',
                'strengthened_cleanup_with_cuda_mps_cache_clearing',
                'initialized_dataloaders_to_avoid_unbound_local_error',
                'store_eval_tensors_on_cpu_to_prevent_memory_growth'
            ]
        },
        'training': {
            'best_epoch': best_epoch,
            'best_val_mae_unscaled': best_val_mae_unscaled,
            'total_epochs': len(train_history),
            'history': train_history
        },
        'test_results': test_results,
        'scaler_info': {
            'thermal_mean': thermal_scaler.mean_.tolist(),
            'thermal_scale': thermal_scaler.scale_.tolist(),
            'param_mean': param_scaler.mean_.tolist(),
            'param_scale': param_scaler.scale_.tolist(),
            'scaler_type': 'StandardScaler'
        }
    }
    
    # Enhanced Plotting
    try:
        print(f"\nGENERATING PLOTS WITH FULLY FIXED DATA (HORIZON: {horizon_label})...")
        
        # Training curves
        plot_unscaled_training_curves(train_history, Config.output_dir, best_epoch, Config.prediction_horizon_steps)
        
        # Energy conservation status
        conservation_status = analyze_energy_conservation_status(
            test_results, power_summary, Config.prediction_horizon_steps
        )
        
        # Mock error analysis
        error_summary = {
            'overall_statistics': {
                'mean_mae_across_sensors': test_results['test_mae_unscaled'],
                'mean_rmse_across_sensors': test_results['test_rmse_unscaled'],
                'mean_r2_across_sensors': test_results['test_r2_overall_unscaled'],
                'max_error_across_all': test_results['test_mae_unscaled'] * 1.5
            }
        }
        
        # Overall statistics summary
        overall_stats = generate_overall_statistics_summary(
            test_results, error_summary, power_summary, conservation_status, 
            Config.output_dir, Config.prediction_horizon_steps
        )
        
        all_results['enhanced_analysis'] = {
            'power_summary': power_summary,
            'conservation_status': conservation_status,
            'error_summary': error_summary,
            'overall_stats': overall_stats
        }
        
    except Exception as e:
        print(f"Enhanced plot generation encountered an issue: {e}")
        print("Continuing with final summary...")
    
    # Save results
    results_for_json = {k: v for k, v in all_results.items() if k != 'test_results'}
    results_for_json['test_results'] = {k: v for k, v in test_results.items() if k != 'predictions_unscaled'}
    
    results_path = os.path.join(Config.output_dir, f'complete_results_FULLY_FIXED_H{Config.prediction_horizon_steps}.json')
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=2, default=str)
    
    # Save predictions separately
    predictions_path = os.path.join(Config.output_dir, f'test_predictions_FULLY_FIXED_H{Config.prediction_horizon_steps}.npz')
    np.savez(predictions_path, 
             y_true=test_results['predictions_unscaled']['y_true'],
             y_pred=test_results['predictions_unscaled']['y_pred'])
    
    print(f"\nAll FULLY FIXED results saved to: {Config.output_dir}")
    
    # FINAL SUMMARY
    print_final_summary_fixed(
        best_epoch, 
        best_val_mae_unscaled, 
        best_val_loss, 
        test_results, 
        Config.output_dir,
        Config.prediction_horizon_steps
    )
    
    # Clean up DataLoaders before exit
    cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)


# =====================
# ENTRY POINT WITH ARGPARSE
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Physics-Informed LSTM Training with Horizon Sweep')
    parser.add_argument('--mode', choices=['single', 'horizon_sweep'], default='single',
                       help='Run mode: single experiment or horizon sweep')
    parser.add_argument('--seq_len', type=int, default=20,
                       help='Sequence length for horizon sweep (default: 20)')
    parser.add_argument('--horizons', nargs='+', type=int, 
                       default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
                       help='List of horizons to test in sweep mode')
    parser.add_argument('--mae_thresh', type=float, default=3.0,
                       help='MAE threshold for pass/fail (default: 3.0 °C)')
    parser.add_argument('--viol_thresh', type=float, default=1.0,
                       help='Violation percentage threshold (default: 1.0%)')
    parser.add_argument('--ratio_thresh', type=float, default=1.01,
                       help='Power balance ratio threshold (default: 1.01)')
    parser.add_argument('--pb_mult', type=float, default=3.0,
                       help='Power balance loss multiplier vs baseline (default: 3.0)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        print("Running single experiment with current Config settings...")
        main()
        
    elif args.mode == 'horizon_sweep':
        print(f"Running horizon sweep with L={args.seq_len}")
        print(f"Horizons: {args.horizons}")
        print(f"Thresholds: MAE≤{args.mae_thresh}°C, Viol≤{args.viol_thresh}%, Ratio≤{args.ratio_thresh}, PB≤{args.pb_mult}x")
        
        horizon_sweep_fixed_seq(
            seq_len=args.seq_len,
            horizons=args.horizons,
            mae_thresh=args.mae_thresh,
            viol_thresh_pct=args.viol_thresh,
            ratio_thresh=args.ratio_thresh,
            pb_mult_baseline=args.pb_mult
        )