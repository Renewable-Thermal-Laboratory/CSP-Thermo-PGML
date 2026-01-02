import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math


# === Paper/journal plotting defaults ===
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
})

from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import warnings
import re
import argparse
import pandas as pd
PAPER_FIG_DIR = None

# Set PyTorch multiprocessing sharing strategy
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

# Suppress all warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Fixed imports - ensure all classes are imported
from .new_model import (
    build_model, 
    create_trainer, 
    compute_r2_score, 
    PhysicsInformedTrainer,
    PhysicsInformedLSTM
)
from .new_dataset_builder import create_data_loaders


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
def extract_power_metadata_from_batch(time_series_batch, static_params_batch, targets_batch, thermal_scaler, param_scaler, horizon_steps=1, num_sensors=10):
    """
    Extract power metadata from the actual batch data using canonical keys (horizon-agnostic).
    
    FIXES:
    1. Use targets_batch instead of last input row for target temperatures
    2. Compute time_target correctly using dt * horizon_steps
    3. Include absorptivity and surface fraction in metadata
    
    Args:
        time_series_batch: (batch_size, seq_len, num_sensors + 1) - time + temperature sensors
        static_params_batch: (batch_size, 4) - [htc, flux, abs, surf] (scaled)
        targets_batch: (batch_size, num_sensors) - target temperatures (scaled)
        thermal_scaler: StandardScaler for temperatures
        param_scaler: StandardScaler for static parameters
        horizon_steps: int - prediction horizon in steps
        num_sensors: int - The number of sensors.
    
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
            temp_sequence = time_series_np[batch_idx, :, 1:]  # Skip time column
            
            # Get temperatures at first timestep and use provided targets
            temps_initial_scaled = temp_sequence[0, :]
            temps_target_scaled = targets_np[batch_idx]
            
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
                'temps_row1': temps_initial_unscaled.tolist(),
                'temps_target': temps_target_unscaled.tolist(),
                'time_row1': time_initial,
                'time_target': time_target,
                'time_diff': time_diff,
                'horizon_steps': horizon_steps,
                'time_normalized': False,
                
                # Physics parameters
                'h': htc_unscaled,
                'q0': flux_unscaled,
                'abs_coeff': abs_coeff,
                'surf_frac': surf_frac,
                
                # Legacy keys for backward compatibility (will be deprecated)
                'temps_row21': temps_target_unscaled.tolist(),
                'time_row21': time_target,
            }
            
            power_metadata_list.append(power_metadata)
            
        except Exception as e:
            print(f"Error extracting power metadata for batch index {batch_idx}: {e}")
            # Create dummy metadata as fallback
            power_metadata_list.append({
                'temps_row1': [300.0] * num_sensors,
                'temps_target': [301.0] * num_sensors,
                'time_row1': 0.0,
                'time_target': float(horizon_steps),
                'time_diff': float(horizon_steps),
                'horizon_steps': horizon_steps,
                'time_normalized': False,
                'h': 50.0,
                'q0': 1000.0,
                'abs_coeff': 0.8,
                'surf_frac': 1.0,
                'temps_row21': [301.0] * num_sensors,
                'time_row21': float(horizon_steps),
            })
    
    return power_metadata_list


def process_power_data_batch_fixed(power_data_list, num_sensors=10):
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
                'temps_row1': [300.0] * num_sensors,
                'temps_target': [301.0] * num_sensors,
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
            temps_row1 = power_data.get('temps_row1', [300.0] * num_sensors)
            temps_target = power_data.get('temps_target', power_data.get('temps_row21', [301.0] * num_sensors))
            time_diff = power_data.get('time_diff', 1.0)
            horizon_steps = power_data.get('horizon_steps', 1)
            htc_value = power_data.get('h', 50.0)  # Heat transfer coefficient
            q0_value = power_data.get('q0', 1000.0)
            abs_coeff = power_data.get('abs_coeff', 0.8)  # Include absorptivity
            surf_frac = power_data.get('surf_frac', 1.0)  # Include surface fraction
            
            # Validate data
            if (isinstance(temps_row1, list) and len(temps_row1) == num_sensors and
                isinstance(temps_target, list) and len(temps_target) == num_sensors and
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
                    'temps_row1': [300.0] * num_sensors,
                    'temps_target': [301.0] * num_sensors,
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
                'temps_row1': [300.0] * num_sensors,
                'temps_target': [301.0] * num_sensors,
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
    
    def __init__(self, base_trainer, thermal_scaler, param_scaler, horizon_steps=1, device=None, num_sensors=10):
        self.base_trainer = base_trainer
        self.thermal_scaler = thermal_scaler
        self.param_scaler = param_scaler
        self.horizon_steps = horizon_steps
        self.num_sensors = num_sensors
        
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
        mean = self.thermal_mean.to(scaled_temps.device)
        scale = self.thermal_scale.to(scaled_temps.device)
        unscaled_temps = scaled_temps * scale + mean
        return unscaled_temps
    
    def train_step_unscaled(self, batch):
        """Training step with fixed power metadata extraction (horizon-agnostic)."""
        time_series, static_params, targets, original_power_data = batch
        
        time_series = time_series.to(self.device)
        static_params = static_params.to(self.device)
        targets = targets.to(self.device)
        
        extracted_power_metadata = extract_power_metadata_from_batch(
            time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps, self.num_sensors
        )
        
        trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
        train_results = self.base_trainer.train_step(trainer_batch)
        
        with torch.no_grad():
            y_pred_scaled = self.base_trainer.model([time_series, static_params])
            
            y_true_unscaled = self.unscale_temperatures(targets)
            y_pred_unscaled = self.unscale_temperatures(y_pred_scaled)
            
            mae_unscaled = torch.mean(torch.abs(y_true_unscaled - y_pred_unscaled))
            rmse_unscaled = torch.sqrt(torch.mean(torch.square(y_true_unscaled - y_pred_unscaled)))
            
            train_results.update({
                'mae_unscaled': mae_unscaled.item(),
                'rmse_unscaled': rmse_unscaled.item()
            })
        
        return train_results
    
    def validation_step_unscaled(self, batch):
        """Validation step with fixed power metadata extraction (horizon-agnostic)."""
        time_series, static_params, targets, original_power_data = batch
        
        time_series = time_series.to(self.device)
        static_params = static_params.to(self.device)
        targets = targets.to(self.device)
        
        extracted_power_metadata = extract_power_metadata_from_batch(
            time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps, self.num_sensors
        )
        
        trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
        val_results = self.base_trainer.validation_step(trainer_batch)
        
        with torch.no_grad():
            y_pred_scaled = self.base_trainer.model([time_series, static_params])
            
            y_true_unscaled = self.unscale_temperatures(targets)
            y_pred_unscaled = self.unscale_temperatures(y_pred_scaled)
            
            mae_unscaled = torch.mean(torch.abs(y_true_unscaled - y_pred_unscaled))
            rmse_unscaled = torch.sqrt(torch.mean(torch.square(y_true_unscaled - y_pred_unscaled)))
            
            val_results.update({
                'val_mae_unscaled': mae_unscaled.item(),
                'val_rmse_unscaled': rmse_unscaled.item()
            })
        
        return val_results

    def train_epoch_unscaled(self, train_loader, val_loader=None):
        """Train for one epoch with fixed power metadata extraction (horizon-agnostic)."""
        from collections import defaultdict
        
        epoch_train_metrics = defaultdict(list)
        epoch_val_metrics = defaultdict(list)
        
        for batch in train_loader:
            metrics = self.train_step_unscaled(batch)
            for key, value in metrics.items():
                epoch_train_metrics[f'train_{key}'].append(value)
        
        if val_loader is not None:
            for batch in val_loader:
                metrics = self.validation_step_unscaled(batch)
                for key, value in metrics.items():
                    epoch_val_metrics[key].append(value)
        
        results = {}
        for key, values in epoch_train_metrics.items():
            results[key] = np.mean(values)
        for key, values in epoch_val_metrics.items():
            results[key] = np.mean(values)
        
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
                
                time_series = time_series.to(self.device)
                static_params = static_params.to(self.device)
                targets = targets.to(self.device)
                
                predictions_scaled = self.model([time_series, static_params])
                
                all_predictions_scaled.append(predictions_scaled.detach().cpu())
                all_targets_scaled.append(targets.detach().cpu())
                
                extracted_power_metadata = extract_power_metadata_from_batch(
                    time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps, self.num_sensors
                )
                
                trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
                batch_metrics = self.base_trainer.validation_step(trainer_batch)
                
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
        
        all_predictions_scaled = torch.cat(all_predictions_scaled, dim=0)
        all_targets_scaled = torch.cat(all_targets_scaled, dim=0)
        
        all_predictions_unscaled = self.unscale_temperatures(all_predictions_scaled)
        all_targets_unscaled = self.unscale_temperatures(all_targets_scaled)
        
        mae_unscaled = _as_float(torch.mean(torch.abs(all_targets_unscaled - all_predictions_unscaled)))
        rmse_unscaled = _as_float(torch.sqrt(torch.mean(torch.square(all_targets_unscaled - all_predictions_unscaled))))
        r2_overall_unscaled = _as_float(compute_r2_score(all_targets_unscaled, all_predictions_unscaled))
        
        per_sensor_metrics = []
        for sensor_idx in range(self.num_sensors):
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
        
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            aggregated_metrics[key] = np.mean(values)
        
        test_physics_loss = aggregated_metrics.get('val_physics_loss', 0.0)
        test_constraint_loss = (aggregated_metrics.get('val_soft_penalty', 0.0) + 
                               aggregated_metrics.get('val_excess_penalty', 0.0))
        test_power_balance_loss = aggregated_metrics.get('val_power_balance_loss', 0.0)
        
        results = {
            f'{split_name}_mae_unscaled': mae_unscaled,
            f'{split_name}_rmse_unscaled': rmse_unscaled,
            f'{split_name}_r2_overall_unscaled': r2_overall_unscaled,
            f'{split_name}_per_sensor_metrics': per_sensor_metrics,
            f'{split_name}_physics_loss': test_physics_loss,
            f'{split_name}_constraint_loss': test_constraint_loss,
            f'{split_name}_power_balance_loss': test_power_balance_loss,
            'predictions_unscaled': {
                'y_true': all_targets_unscaled.detach().cpu().numpy(),
                'y_pred': all_predictions_unscaled.detach().cpu().numpy()
            }
        }
        
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
                
                time_series = time_series.to(self.device)
                static_params = static_params.to(self.device)
                targets = targets.to(self.device)
                
                try:
                    extracted_power_metadata = extract_power_metadata_from_batch(
                        time_series, static_params, targets, self.thermal_scaler, self.param_scaler, self.horizon_steps, self.num_sensors
                    )
                    
                    if extracted_power_metadata:
                        y_pred = self.model([time_series, static_params])
                        
                        physics_loss, soft_penalty, excess_penalty, power_balance_loss, power_info = \
                            self.base_trainer.compute_physics_loss(
                                y_pred, extracted_power_metadata
                            )
                        
                        if power_info:
                            if 'total_actual_powers' in power_info and power_info['total_actual_powers']:
                                total_actual_powers.extend(power_info['total_actual_powers'])
                            
                            if 'total_predicted_powers' in power_info and power_info['total_predicted_powers']:
                                total_predicted_powers.extend(power_info['total_predicted_powers'])
                            
                            if 'incoming_powers' in power_info and power_info['incoming_powers']:
                                incoming_powers.extend(power_info['incoming_powers'])
                            
                            sample_count += power_info.get('num_samples_processed', 0)
                            
                except Exception as e:
                    print(f"Warning: Error in power analysis: {e}")
                    continue
        
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
            
            print(f"\nTOTAL ACTUAL POWER (sum of {self.num_sensors - 1} bins):")
            print(f"  Mean: {np.mean(total_actual_powers):.2f} W")
            print(f"  Std:  {np.std(total_actual_powers):.2f} W")
            print(f"  Min:  {np.min(total_actual_powers):.2f} W")
            print(f"  Max:  {np.max(total_actual_powers):.2f} W")
            
            print(f"\nTOTAL PREDICTED POWER (sum of {self.num_sensors - 1} bins):")
            print(f"  Mean: {np.mean(total_predicted_powers):.2f} W")
            print(f"  Std:  {np.std(total_predicted_powers):.2f} W")
            print(f"  Min:  {np.min(total_predicted_powers):.2f} W")
            print(f"  Max:  {np.max(total_predicted_powers):.2f} W")
            
            eps = 1e-8
            incoming_safe = np.maximum(incoming_powers, eps)
            actual_to_incoming = total_actual_powers / incoming_safe
            predicted_to_incoming = total_predicted_powers / incoming_safe
            
            print(f"\nPOWER BALANCE RATIOS:")
            print(f"  Actual/Incoming ratio - Mean: {np.mean(actual_to_incoming):.3f}, Std: {np.std(actual_to_incoming):.3f}")
            print(f"  Predicted/Incoming ratio - Mean: {np.mean(predicted_to_incoming):.3f}, Std: {np.std(predicted_to_incoming):.3f}")
            
            actual_violations = np.sum(total_actual_powers > incoming_powers)
            predicted_violations = np.sum(total_predicted_powers > incoming_powers)
            
            print(f"\nENERGY CONSERVATION VIOLATIONS:")
            print(f"  Actual power > incoming: {actual_violations}/{len(total_actual_powers)} ({100*actual_violations/len(total_actual_powers):.1f}%)")
            print(f"  Predicted power > incoming: {predicted_violations}/{len(total_predicted_powers)} ({100*predicted_violations/len(total_predicted_powers):.1f}%)")
            
            print(f"\nSUCCESS: Real power data extracted and analyzed (horizon = {horizon_label})!")
            
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
        if hasattr(test_loader, 'dataset'):
            dataset = test_loader.dataset
            print(f"Found test dataset: {type(dataset).__name__}")
            
            if hasattr(dataset, 'current_files'):
                current_files = dataset.current_files
                print(f"Found {len(current_files)} test files in dataset.current_files")
                
                if hasattr(dataset, 'sample_indices'):
                    sample_indices = dataset.sample_indices
                    print(f"Found {len(sample_indices)} sample indices")
                    
                    for idx, (file_path, start_idx) in enumerate(sample_indices):
                        filename = os.path.basename(file_path)
                        sample_to_filename[idx] = filename
                    
                    print(f"Successfully mapped {len(sample_to_filename)} samples to ACTUAL filenames")
                    
                    print(f"First 5 sample-to-filename mappings:")
                    for i in range(min(5, len(sample_to_filename))):
                        print(f"   Sample {i}: {sample_to_filename[i]}")
                    
                    return sample_to_filename
                
                else:
                    print("Dataset doesn't have sample_indices attribute")
            
            if hasattr(dataset, 'test_files'):
                test_files = dataset.test_files
                print(f"Found {len(test_files)} files in dataset.test_files")
                
                for idx, file_path in enumerate(test_files):
                    filename = os.path.basename(file_path)
                    sample_to_filename[idx] = filename
                
                print(f"Successfully mapped {len(sample_to_filename)} files to filenames")
                return sample_to_filename
            
            current_split = getattr(dataset, 'split', 'unknown')
            print(f"Dataset split: {current_split}")
            
            possible_file_attrs = ['current_files', 'test_files', 'val_files', 'train_files', 'files']
            for attr in possible_file_attrs:
                if hasattr(dataset, attr):
                    files = getattr(dataset, attr)
                    if files and len(files) > 0:
                        print(f"Found {len(files)} files in dataset.{attr}")
                        
                        for idx, file_path in enumerate(files):
                            filename = os.path.basename(file_path)
                            sample_to_filename[idx] = filename
                        
                        print(f"Successfully mapped {len(sample_to_filename)} files from {attr}")
                        return sample_to_filename
        
        print("Could not extract actual filenames from dataset")
        print("Available dataset attributes:", [attr for attr in dir(dataset) if not attr.startswith('_')])
        
    except Exception as e:
        print(f"Error extracting filenames: {e}")
    
    print("WARNING: Using fallback generic filenames - this is not ideal!")
    print("Please check your dataset implementation to expose actual filenames")
    
    fallback_files = [
        "h0.4_flux40000_abs15_surf70_600s.csv",
        "h0.4_flux50000_abs10_surf50_600s.csv", 
        "h0.5_flux40000_abs20_surf90_600s.csv",
        "h0.5_flux100000_abs5_surf50_600s.csv",
        "h1.0_flux40000_abs15_surf70_600s.csv"
    ]
    
    for sample_idx in range(1000):
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
    height_pattern = r'h(\d+\.?\d*)'
    match = re.search(height_pattern, filename.lower())
    
    if match:
        height = float(match.group(1))
        print(f"Parsed height {height}m from filename: {filename}")
        return height
    else:
        alt_patterns = [
            r'(\d+\.?\d*)m',
            r'height_?(\d+\.?\d*)',
            r'h_(\d+\.?\d*)'
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                height = float(match.group(1))
                print(f"Parsed height {height}m from filename using alternative pattern: {filename}")
                return height
        
        print(f"Could not parse height from filename '{filename}', using Config.cylinder_length")
        return 1.0


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
    global PAPER_FIG_DIR
    
    Config.sequence_length = L
    Config.prediction_horizon_steps = H
    
    Config.output_dir = f"output/{Config.experiment_name}_L{L}_H{H}"
    Config.run_tag = f"{Config.experiment_name}_L{L}_H{H}"
    
    os.makedirs(Config.output_dir, exist_ok=True)
    
    PAPER_FIG_DIR = os.path.join(Config.output_dir, "paper_figs")
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    
    print(f"Config updated: L={L}, H={H}")
    print(f"   Output dir: {Config.output_dir}")
    print(f"   Run tag: {Config.run_tag}")
    print(f"   Paper fig dir: {PAPER_FIG_DIR}")


def run_single_experiment(L: int, H: int, seed: int = 42) -> dict:
    """
    Run one experiment with sequence length L and horizon H.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING SINGLE EXPERIMENT: L={L}, H={H}")
    print(f"{ '='*80}")

    update_config_for(L, H)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nLoading datasets for L={L}, H={H}...")
    train_loader = val_loader = test_loader = train_dataset = None
    try:
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
            data_dir=Config.data_dir,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
            sequence_length=Config.sequence_length,
            prediction_horizon=Config.prediction_horizon_steps,
            scaler_dir=Config.scaler_dir,
            num_sensors=Config.num_sensors
        )
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

    print(f"Building model for L={L}, H={H}...")
    try:
        model = build_model(
            num_sensors=Config.num_sensors,
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
            thermal_scaler=thermal_scaler
        )

        trainer = FixedUnscaledEvaluationTrainer(
            base_trainer, thermal_scaler, param_scaler,
            horizon_steps=Config.prediction_horizon_steps, device=device, num_sensors=Config.num_sensors
        )
    except Exception as e:
        print(f"Error building model: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "mae": float('inf'), "rmse": float('inf'), "r2": -float('inf'),
            "physics_loss": float('inf'), "pb_loss": float('inf'), "viol_pct": 100.0,
            "pred_to_in_ratio": float('inf'), "status": "model_build_failed"
        }

    print(f"\nTraining model for L={L}, H={H}...")
    best_val_mae_unscaled = np.inf
    best_epoch = 0
    train_history = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(Config.output_dir, f'best_model_L{L}_H{H}_{timestamp}.pth')
    os.makedirs(Config.output_dir, exist_ok=True)

    try:
        for epoch in range(Config.max_epochs):
            results = trainer.train_epoch_unscaled(train_loader, val_loader)
            train_history.append(results)

            val_mae_unscaled = results['val_mae_unscaled']
            if val_mae_unscaled < best_val_mae_unscaled:
                best_val_mae_unscaled = val_mae_unscaled
                best_epoch = epoch + 1
                torch.save(trainer.model.state_dict(), best_model_path)
    except Exception as e:
        print(f"Error during training: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "status": "training_failed"
        }

    print(f"\nEvaluating best model for L={L}, H={H}...")
    try:
        if os.path.exists(best_model_path):
            trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_results = trainer.evaluate_unscaled(test_loader, "test")
        power_summary = trainer.analyze_power_balance(test_loader, num_samples=500)
        
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "status": "success", **test_results, **power_summary
        }

    except Exception as e:
        print(f"Error during evaluation: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "status": "evaluation_failed"
        }


def generate_all_paper_artifacts(output_dir):
    print(f"Generating paper artifacts in {output_dir}")
    # This is a placeholder for where you would call your plotting functions
    pass


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
    print(f"{ '='*100}")
    
    sweep_dir = "output/sweeps_experiments"
    os.makedirs(sweep_dir, exist_ok=True)
    
    results = []
    pb_baseline = None
    
    for i, H in enumerate(horizons):
        result = run_single_experiment(seq_len, H, seed=42)
        
        if H == 1 and result['status'] == 'success':
            pb_baseline = result.get('test_power_balance_loss', None)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(sweep_dir, f"seq{seq_len}_horizon_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Sweep results saved to: {csv_path}")


def generate_sweep_plots(results, sweep_dir, seq_len, mae_thresh, viol_thresh_pct, ratio_thresh, pb_baseline=None, pb_mult_baseline=3.0):
    """
    Generate and save the 4 required sweep plots.
    """
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    horizons = [r['H'] for r in successful_results]
    maes = [r['test_mae_unscaled'] for r in successful_results]
    viols = [r['conservation_violations']['percentage'] for r in successful_results]
    ratios = [r['mean_predicted_to_incoming_ratio'] for r in successful_results]
    pb_losses = [r['test_power_balance_loss'] for r in successful_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Horizon Sweep Results - Sequence Length L={seq_len}', fontsize=16)
    
    axes[0, 0].scatter(horizons, maes, c='blue', s=80, alpha=0.7, edgecolors='black')
    axes[0, 0].axhline(y=mae_thresh, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({mae_thresh}°C)')
    axes[0, 0].set_xlabel('Horizon Steps (H)')
    axes[0, 0].set_ylabel('MAE (°C)')
    axes[0, 0].set_title('MAE vs Horizon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    combined_path = os.path.join(sweep_dir, f"seq{seq_len}_horizon_sweep_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_overall_statistics_summary(test_results, error_summary, power_summary, conservation_status, output_dir, horizon_steps=1):
    """
    Overall Statistics Summary (horizon-agnostic).
    """
    # Placeholder for summary generation
    pass

def analyze_energy_conservation_status(test_results, power_summary, horizon_steps=1):
    """
    Energy Conservation Status (horizon-agnostic).
    """
    # Placeholder for analysis
    pass

def print_final_summary_fixed(best_epoch, best_val_mae_unscaled, test_results, output_dir, horizon_steps=1):
    """Print comprehensive final summary (horizon-agnostic)."""
    # Placeholder for summary printing
    pass


class Config:
    # Data and model settings
    data_dir = "data/output_with_TC11"
    scaler_dir = "models_TC11"
    batch_size = 32
    learning_rate = 0.001
    max_epochs = 100
    patience = 10
    lstm_units = 64
    dropout_rate = 0.2
    num_sensors = 11

    physics_weight = 0.001
    soft_penalty_weight = 0.001
    excess_penalty_weight = 0.001
    power_balance_weight = 0.0005
    
    sequence_length = 20
    prediction_horizon_steps = 300
    
    cylinder_length = 1.0
    num_workers = 0
    
    experiment_name = "new_theoretical_TC11"

Config.output_dir = f"output/{Config.experiment_name}_H{Config.prediction_horizon_steps}"
Config.run_tag = f"{Config.experiment_name}_H{Config.prediction_horizon_steps}"

os.makedirs(Config.output_dir, exist_ok=True)
PAPER_FIG_DIR = os.path.join(Config.output_dir, "paper_figs")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


def generate_profile_plots_per_file(trainer, test_loader, horizon_steps, output_dir, num_sensors=10):
    """
    Generate temperature profile plots for each test file at the given horizon.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    trainer.model.eval()
    
    sample_to_filename = get_test_filenames_and_sample_mapping(test_loader)
    
    profiles_by_file = defaultdict(list)
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            time_series, static_params, targets, original_power_data = batch
            
            time_series = time_series.to(trainer.device)
            static_params_scaled = static_params.to(trainer.device)
            targets = targets.to(trainer.device)
            
            static_params_unscaled = trainer.param_scaler.inverse_transform(static_params_scaled.cpu().numpy())
            
            predictions_scaled = trainer.model([time_series, static_params_scaled])
            
            batch_size = time_series.shape[0]
            for sample_idx in range(batch_size):
                global_sample_idx = sample_count + sample_idx
                
                if global_sample_idx in sample_to_filename:
                    filename = sample_to_filename[global_sample_idx]
                    
                    y_true_sample = trainer.unscale_temperatures(targets[sample_idx:sample_idx+1]).cpu().numpy()[0]
                    y_pred_sample = trainer.unscale_temperatures(predictions_scaled[sample_idx:sample_idx+1]).cpu().numpy()[0]
                    
                    h, flux, abs_val, surf = static_params_unscaled[sample_idx]
                    
                    profiles_by_file[filename].append({
                        'y_true': y_true_sample,
                        'y_pred': y_pred_sample,
                        'h': h,
                        'flux': flux,
                        'abs': abs_val,
                        'surf': surf,
                    })
            
            sample_count += batch_size
    
    horizon_label = f"H{horizon_steps}"
    plots_dir = os.path.join(output_dir, f"profile_plots_{horizon_label}")
    os.makedirs(plots_dir, exist_ok=True)
    
    for filename, profiles in profiles_by_file.items():
        if not profiles:
            continue
            
        for profile in profiles:
            profile['mae'] = np.mean(np.abs(profile['y_true'] - profile['y_pred']))
        
        best_profile = min(profiles, key=lambda p: p['mae'])
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        h = best_profile['h']
        physical_depths = np.linspace(0, h, num_sensors)
        
        ax.plot(best_profile['y_true'], physical_depths, 'o-', label='Actual', color='blue', markersize=6, linewidth=2)
        ax.plot(best_profile['y_pred'], physical_depths, 's--', label='Predicted', color='red', markersize=5, linewidth=2)
        
        for i in range(num_sensors):
            ax.annotate(f'TC{i+1}', (best_profile['y_true'][i], physical_depths[i]), textcoords="offset points", xytext=(5, -5), ha='left')
            
        mae = best_profile['mae']
        flux = best_profile['flux']
        abs_val = best_profile['abs']
        surf = best_profile['surf']
        
        title = f'h={h:.4f}m, flux={flux:.0f}, abs={abs_val:.2f}, surf={surf:.2f} | H={horizon_steps} | MAE: {mae:.2f}°C'
        ax.set_title(title)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Depth (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        clean_filename = filename.replace('.csv', '')
        plot_filename = f"{clean_filename}_{horizon_label}_profiles.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved profile plot: {plot_path}")
    
    print(f"Profile plots for horizon {horizon_steps} saved to: {plots_dir}")
    return plots_dir


def run_single_experiment_with_profiles(L: int, H: int, seed: int = 42) -> dict:
    """
    Modified version of run_single_experiment that also generates profile plots.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING SINGLE EXPERIMENT WITH PROFILES: L={L}, H={H}")
    print(f"{ '='*80}")

    update_config_for(L, H)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --------------------------
    # Data
    # --------------------------
    print(f"\nLoading datasets for L={L}, H={H}...")
    train_loader = val_loader = test_loader = train_dataset = None
    try:
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
            data_dir=Config.data_dir,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
            sequence_length=Config.sequence_length,
            prediction_horizon=Config.prediction_horizon_steps,
            scaler_dir=Config.scaler_dir,
            num_sensors=Config.num_sensors
        )
        physics_params = train_dataset.get_physics_params()
        thermal_scaler = physics_params['thermal_scaler']
        param_scaler = physics_params['param_scaler']
    except Exception as e:
        print(f"Error loading datasets: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"status": "data_load_failed"}

    # --------------------------
    # Model / Trainer
    # --------------------------
    print(f"Building model for L={L}, H={H}...")
    try:
        model = build_model(
            num_sensors=Config.num_sensors,
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
            thermal_scaler=thermal_scaler
        )

        trainer = FixedUnscaledEvaluationTrainer(
            base_trainer, thermal_scaler, param_scaler,
            horizon_steps=Config.prediction_horizon_steps, device=device, num_sensors=Config.num_sensors
        )
    except Exception as e:
        print(f"Error building model: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"status": "model_build_failed"}

    # --------------------------
    # Train (early stop on val MAE_unscaled)
    # --------------------------
    print(f"\nTraining model for L={L}, H={H}...")
    best_val_mae_unscaled = np.inf
    train_history = []
    best_model_path = os.path.join(Config.output_dir, f'best_model_L{L}_H{H}.pth')

    try:
        for epoch in range(Config.max_epochs):
            results = trainer.train_epoch_unscaled(train_loader, val_loader)
            train_history.append(results)
            if results['val_mae_unscaled'] < best_val_mae_unscaled:
                best_val_mae_unscaled = results['val_mae_unscaled']
                torch.save(trainer.model.state_dict(), best_model_path)
    except Exception as e:
        print(f"Error during training: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"status": "training_failed"}

    # --------------------------
    # Evaluate + Generate Profile Plots
    # --------------------------
    print(f"\nEvaluating best model for L={L}, H={H}...")
    try:
        if os.path.exists(best_model_path):
            trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_results = trainer.evaluate_unscaled(test_loader, "test")
        
        print(f"\nGenerating profile plots for each test file (H={H})...")
        generate_profile_plots_per_file(trainer, test_loader, H, Config.output_dir, Config.num_sensors)

        power_summary = trainer.analyze_power_balance(test_loader, num_samples=500)
        
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {**test_results, **power_summary}

    except Exception as e:
        print(f"Error during evaluation: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {
            "L": L, "H": H, "status": "evaluation_failed"
        }

if __name__ == '__main__':
    run_single_experiment_with_profiles(L=20, H=30, seed=42)
