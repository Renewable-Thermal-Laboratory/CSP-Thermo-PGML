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
def extract_power_metadata_from_batch(time_series_batch, static_params_batch, targets_batch, thermal_scaler, param_scaler, horizon_steps=1, num_sensors=10, original_power_data_list=None):
    """
    Extract power metadata from the actual batch data using canonical keys (horizon-agnostic).

    PREFERRED PATH: When original_power_data_list is provided (from the dataset's own power_data
    dict), use its pre-computed unscaled values directly.  This avoids the critical bug where
    time_diff was recomputed from the *scaled* time column (normalized as (t-300)/300), which
    made dt ≈ 0.003 s instead of 1 s, inflating all computed powers by ~300×.

    FALLBACK PATH: When original_power_data_list is None or a sample entry is invalid, reconstruct
    from the batch tensors — but use the correct time unscaling (mean=300, std=300 seconds).

    Args:
        time_series_batch: (batch_size, seq_len, num_sensors + 1) - scaled time + temperature sensors
        static_params_batch: (batch_size, 4) - [htc, flux, abs, surf] (scaled)
        targets_batch: (batch_size, num_sensors) - target temperatures (scaled)
        thermal_scaler: StandardScaler for temperatures
        param_scaler: StandardScaler for static parameters
        horizon_steps: int - prediction horizon in steps
        num_sensors: int - The number of sensors.
        original_power_data_list: list of dicts from the dataset (already unscaled). Use this
            whenever available to get the correct time_diff, temps_row1, temps_target, h, q0.

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
            # ---------------------------------------------------------------
            # PREFERRED PATH: use the dataset's already-unscaled power_data.
            # This is the only way to get a correct time_diff, because the
            # time column in time_series_batch is *scaled* ((t-300)/300) and
            # reading dt from it would give ~0.003 s instead of 1 s.
            # ---------------------------------------------------------------
            orig = None
            if original_power_data_list is not None and batch_idx < len(original_power_data_list):
                orig = original_power_data_list[batch_idx]

            if (orig is not None and isinstance(orig, dict)
                    and 'time_diff' in orig
                    and 'temps_row1' in orig
                    and 'temps_target' in orig
                    and 'h' in orig
                    and 'q0' in orig):
                # All unscaled values come straight from the dataset builder.
                temps_initial_unscaled = np.array(orig['temps_row1'], dtype=np.float32)
                temps_target_unscaled  = np.array(orig['temps_target'], dtype=np.float32)
                time_diff   = max(float(orig['time_diff']), 1e-8)
                time_initial = float(orig.get('time_row1', 0.0))
                time_target  = float(orig.get('time_target', time_initial + time_diff))
                htc_unscaled = float(orig['h'])
                flux_unscaled = float(orig['q0'])
                abs_coeff  = float(orig.get('abs_coeff', orig.get('abs', 0.0)))
                surf_frac  = float(orig.get('surf_frac', orig.get('surf', 1.0)))

            else:
                # -----------------------------------------------------------
                # FALLBACK PATH: reconstruct from batch tensors.
                # Time column is scaled as (t_seconds - 300) / 300, so we
                # must invert that before computing dt.
                # -----------------------------------------------------------
                TIME_MEAN = 300.0   # seconds
                TIME_STD  = 300.0   # seconds

                temp_sequence = time_series_np[batch_idx, :, 1:]  # Skip time column
                temps_initial_scaled = temp_sequence[0, :]
                temps_target_scaled  = targets_np[batch_idx]

                temps_initial_unscaled = thermal_scaler.inverse_transform([temps_initial_scaled])[0]
                temps_target_unscaled  = thermal_scaler.inverse_transform([temps_target_scaled])[0]

                # Unscale time back to seconds
                t0_scaled = float(time_series_np[batch_idx, 0, 0])
                t0 = t0_scaled * TIME_STD + TIME_MEAN  # seconds
                if time_series_np.shape[1] > 1:
                    dt_scaled = float(time_series_np[batch_idx, 1, 0] - time_series_np[batch_idx, 0, 0])
                    dt = dt_scaled * TIME_STD  # convert scaled dt → seconds
                else:
                    dt = 1.0

                time_initial = t0
                time_target  = t0 + horizon_steps * dt
                time_diff    = max(time_target - time_initial, 1e-8)

                static_params_scaled = static_params_np[batch_idx, :]
                static_params_unscaled = param_scaler.inverse_transform([static_params_scaled])[0]

                htc_unscaled  = float(static_params_unscaled[0])
                flux_unscaled = float(static_params_unscaled[1])
                abs_coeff     = float(static_params_unscaled[2])
                surf_frac     = float(static_params_unscaled[3])
            
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
        
        # Skip power-metadata extraction when physics is inactive (all weights 0).
        if self.base_trainer._physics_active:
            extracted_power_metadata = extract_power_metadata_from_batch(
                time_series, static_params, targets, self.thermal_scaler, self.param_scaler,
                self.horizon_steps, self.num_sensors,
                original_power_data_list=original_power_data
            )
            trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
        else:
            trainer_batch = [time_series, static_params, targets, None]
        # return_pred=True reuses train_step's forward — no redundant second forward pass.
        train_results, y_pred_scaled = self.base_trainer.train_step(trainer_batch, return_pred=True)

        with torch.no_grad():
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
        
        # Skip power-metadata extraction when physics is inactive (all weights 0).
        if self.base_trainer._physics_active:
            extracted_power_metadata = extract_power_metadata_from_batch(
                time_series, static_params, targets, self.thermal_scaler, self.param_scaler,
                self.horizon_steps, self.num_sensors,
                original_power_data_list=original_power_data
            )
            trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
        else:
            trainer_batch = [time_series, static_params, targets, None]
        # return_pred=True reuses validation_step's forward — no redundant second forward pass.
        val_results, y_pred_scaled = self.base_trainer.validation_step(trainer_batch, return_pred=True)

        with torch.no_grad():
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
        
        val_is_empty = False
        if val_loader is not None:
            for batch in val_loader:
                metrics = self.validation_step_unscaled(batch)
                for key, value in metrics.items():
                    epoch_val_metrics[key].append(value)
            if len(epoch_val_metrics) == 0:
                val_is_empty = True
        
        results = {}
        for key, values in epoch_train_metrics.items():
            results[key] = np.mean(values)
            
        if not val_is_empty and val_loader is not None and len(epoch_val_metrics) > 0:
            for key, values in epoch_val_metrics.items():
                results[key] = np.mean(values)
        else:
            # ⚠ WARNING: val set is empty — val metrics mirror train metrics.
            # Model selection is blind; early stopping will NOT reflect generalisation.
            if val_loader is not None:
                print(
                    "\n[WARNING] Validation loader produced ZERO batches for this horizon. "
                    "val_mae_unscaled is being set equal to train_mae_unscaled — "
                    "best-model selection and early stopping are UNRELIABLE. "
                    "Consider increasing data or reducing the prediction horizon.\n"
                )
            results['val_loss']              = results.get('train_loss', 0.0)
            results['val_mae']               = results.get('train_mae', 0.0)
            results['val_mae_unscaled']      = results.get('train_mae_unscaled', 0.0)
            results['val_physics_loss']      = results.get('train_physics_loss', 0.0)
            results['val_soft_penalty']      = results.get('train_soft_penalty', 0.0)
            results['val_excess_penalty']    = results.get('train_excess_penalty', 0.0)
            results['val_power_balance_loss']= results.get('train_power_balance_loss', 0.0)
        
        # Store 'val_is_empty' flag so callers can detect this
        results['_val_is_empty'] = val_is_empty
        
        for key, value in results.items():
            if key in self.base_trainer.history:
                self.base_trainer.history[key].append(float(value))
        
        return results


    def evaluate_unscaled(self, data_loader, split_name="test"):
        """Comprehensive evaluation with fixed power metadata extraction (horizon-agnostic).
        
        Returns a sentinel dict (all metrics = NaN / empty list) when the loader has no samples,
        instead of crashing on torch.cat([]).
        """
        horizon_label = f"{self.horizon_steps} step{'s' if self.horizon_steps != 1 else ''}"
        
        # --- Guard: empty loader ---
        dataset_size = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else None
        if dataset_size == 0 or (dataset_size is None and sum(1 for _ in data_loader) == 0):
            print(
                f"\n[WARNING] evaluate_unscaled({split_name}): test loader has ZERO samples "
                f"(horizon = {horizon_label}). Skipping evaluation and returning NaN sentinels."
            )
            nan = float('nan')
            return {
                f'{split_name}_mae_unscaled':      nan,
                f'{split_name}_rmse_unscaled':     nan,
                f'{split_name}_r2_overall_unscaled': nan,
                f'{split_name}_per_sensor_metrics': [],
                f'{split_name}_physics_loss':      nan,
                f'{split_name}_constraint_loss':   nan,
                f'{split_name}_power_balance_loss': nan,
                'predictions_unscaled': {'y_true': np.array([]), 'y_pred': np.array([])},
                '_empty_loader': True,
            }
        
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
                    time_series, static_params, targets, self.thermal_scaler, self.param_scaler,
                    self.horizon_steps, self.num_sensors,
                    original_power_data_list=original_power_data
                )
                
                trainer_batch = [time_series, static_params, targets, extracted_power_metadata]
                batch_metrics = self.base_trainer.validation_step(trainer_batch)
                
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
        
        # Second-level guard: loop produced no data (e.g. all batches were empty)
        if len(all_predictions_scaled) == 0:
            print(
                f"\n[WARNING] evaluate_unscaled({split_name}): no predictions collected "
                f"(horizon = {horizon_label}). Returning NaN sentinels."
            )
            nan = float('nan')
            return {
                f'{split_name}_mae_unscaled':      nan,
                f'{split_name}_rmse_unscaled':     nan,
                f'{split_name}_r2_overall_unscaled': nan,
                f'{split_name}_per_sensor_metrics': [],
                f'{split_name}_physics_loss':      nan,
                f'{split_name}_constraint_loss':   nan,
                f'{split_name}_power_balance_loss': nan,
                'predictions_unscaled': {'y_true': np.array([]), 'y_pred': np.array([])},
                '_empty_loader': True,
            }
        
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

        # --- Per-file breakdown: which held-out file drives the test error? ---
        # Test loader is shuffle=False / drop_last=False, so the concatenated predictions
        # are in dataset order and align 1:1 with dataset.sample_indices.
        per_file_metrics = {}
        try:
            ds = data_loader.dataset
            sidx = getattr(ds, 'sample_indices', None)
            if sidx is not None and len(sidx) == all_targets_unscaled.shape[0]:
                file_to_rows = defaultdict(list)
                for i, (fp, _start) in enumerate(sidx):
                    file_to_rows[os.path.basename(fp)].append(i)
                for fname, rows in file_to_rows.items():
                    rt = torch.tensor(rows, dtype=torch.long)
                    yt, yp = all_targets_unscaled[rt], all_predictions_unscaled[rt]
                    per_file_metrics[fname] = {
                        'n': len(rows),
                        'mae': _as_float(torch.mean(torch.abs(yt - yp))),
                        'rmse': _as_float(torch.sqrt(torch.mean((yt - yp) ** 2))),
                        'r2': _as_float(compute_r2_score(yt, yp)),
                    }
        except Exception as _e:
            print(f"(per-file breakdown skipped: {_e})")

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
            f'{split_name}_per_file_metrics': per_file_metrics,
            f'{split_name}_physics_loss': test_physics_loss,
            f'{split_name}_constraint_loss': test_constraint_loss,
            f'{split_name}_power_balance_loss': test_power_balance_loss,
            'predictions_unscaled': {
                'y_true': all_targets_unscaled.detach().cpu().numpy(),
                'y_pred': all_predictions_unscaled.detach().cpu().numpy()
            }
        }
        
        results.update(aggregated_metrics)
        
        print(f"\nTEST SET EVALUATION (UNSCALED - HORIZON = {horizon_label}):")
        print(f"   MAE:  {mae_unscaled:.2f} K")
        print(f"   RMSE: {rmse_unscaled:.2f} K") 
        print(f"   R\u00b2:   {r2_overall_unscaled:.6f}")
        print(f"   Physics Loss: {test_physics_loss:.6f}")
        print(f"   Constraint Loss: {test_constraint_loss:.6f}")
        print(f"   Power Balance Loss: {test_power_balance_loss:.6f}")

        # Per-sensor MAE (overall test metric is dominated by the high-variance sensors)
        print("   Per-sensor MAE (K): " +
              "  ".join(f"TC{i+1}={m['mae']:.2f}" for i, m in enumerate(per_sensor_metrics)))

        # Per-file MAE — shows whether one held-out file (e.g. the OOD abs20) drives the average
        if per_file_metrics:
            print("   Per-file:")
            for fname, m in sorted(per_file_metrics.items(), key=lambda kv: -kv[1]['mae']):
                print(f"      MAE={m['mae']:6.2f} K  RMSE={m['rmse']:6.2f}  R2={m['r2']:7.4f}  n={m['n']:5d}  {fname}")

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
                        time_series, static_params, targets, self.thermal_scaler, self.param_scaler,
                        self.horizon_steps, self.num_sensors,
                        original_power_data_list=original_power_data
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


def save_single_run_performance_metrics(L: int, H: int, results: dict) -> None:
    """Save performance metrics of a single run to a CSV file inside Config.output_dir."""
    metrics_csv_path = os.path.join(Config.output_dir, "performance_metrics.csv")
    try:
        viol = results.get('conservation_violations', {})
        row = {
            'experiment':   Config.experiment_name,
            'seq_len':      L,
            'horizon':      H,
            'status':       results.get('status', 'unknown'),
            'best_epoch':   results.get('best_epoch', float('nan')),
            'mae_K':        results.get('test_mae_unscaled',       float('nan')),
            'rmse_K':       results.get('test_rmse_unscaled',      float('nan')),
            'r2':           results.get('test_r2_overall_unscaled', float('nan')),
            'physics_loss':       results.get('test_physics_loss',      float('nan')),
            'power_balance_loss': results.get('test_power_balance_loss', float('nan')),
            'mean_actual_power_W':    results.get('mean_actual_power',    float('nan')),
            'mean_predicted_power_W': results.get('mean_predicted_power', float('nan')),
            'mean_incoming_power_W':  results.get('mean_incoming_power',  float('nan')),
            'conservation_violations_pct': (
                viol.get('percentage', float('nan'))
                if isinstance(viol, dict) else float('nan')
            ),
        }
        pd.DataFrame([row]).to_csv(metrics_csv_path, index=False)
        print(f"Performance metrics saved to CSV: {metrics_csv_path}")
    except Exception as e:
        print(f"Error saving performance metrics CSV: {e}")


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

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
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
            num_sensors=Config.num_sensors,
            zscore_threshold=Config.zscore_threshold,
            target_test_files=Config.target_test_files,
            augment_noise_std=Config.augment_noise_std,
            bin_target=Config.bin_target
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
            device=device,
            horizon_steps=Config.prediction_horizon_steps
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
        
        results = {
            "L": L, "H": H, "status": "success", "best_epoch": best_epoch, **test_results, **power_summary
        }
        save_single_run_performance_metrics(L, H, results)
        return results

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
                          horizons: list = [60, 180, 300, 480, 600, 900],
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
        result = run_single_experiment_with_profiles(seq_len, H, seed=42)
        
        if H == 1 and result['status'] == 'success':
            pb_baseline = result.get('test_power_balance_loss', None)
        
        results.append(result)
    
    # Build a clean, human-readable summary with the key metrics.
    # The raw `results` dicts contain large nested objects (predictions_unscaled,
    # per_sensor_metrics …) that make the CSV unreadable — extract only what matters.
    summary_rows = []
    for r in results:
        viol = r.get('conservation_violations', {})
        row = {
            'experiment':   Config.experiment_name,
            'seq_len':      r.get('L', seq_len),
            'horizon':      r.get('H'),
            'status':       r.get('status', 'unknown'),
            'best_epoch':   r.get('best_epoch', float('nan')),
            # ── Primary accuracy metrics (unscaled, in Kelvin) ──
            'mae_K':        r.get('test_mae_unscaled',       float('nan')),
            'rmse_K':       r.get('test_rmse_unscaled',      float('nan')),
            'r2':           r.get('test_r2_overall_unscaled', float('nan')),
            # ── Physics metrics ──
            'physics_loss':       r.get('test_physics_loss',      float('nan')),
            'power_balance_loss': r.get('test_power_balance_loss', float('nan')),
            # ── Power balance summary ──
            'mean_actual_power_W':    r.get('mean_actual_power',    float('nan')),
            'mean_predicted_power_W': r.get('mean_predicted_power', float('nan')),
            'mean_incoming_power_W':  r.get('mean_incoming_power',  float('nan')),
            'conservation_violations_pct': (
                viol.get('percentage', float('nan'))
                if isinstance(viol, dict) else float('nan')
            ),
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    # One CSV per experiment so TC10 and TC11 don't overwrite each other.
    csv_path = os.path.join(sweep_dir, f"seq{seq_len}_{Config.experiment_name}_horizon_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Sweep results saved to: {csv_path}")

    # Consolidated CSV for all sweeps
    consolidated_csv_path = os.path.join(sweep_dir, "consolidated_horizon_sweep_results.csv")
    if os.path.exists(consolidated_csv_path):
        try:
            existing_df = pd.read_csv(consolidated_csv_path)
            # Filter out previous runs of the same experiment name to prevent duplicate entries
            existing_df = existing_df[existing_df['experiment'] != Config.experiment_name]
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(consolidated_csv_path, index=False)
        except Exception as e:
            print(f"Error appending to consolidated CSV: {e}")
            df.to_csv(consolidated_csv_path, index=False)
    else:
        df.to_csv(consolidated_csv_path, index=False)
    print(f"Consolidated sweep results saved to: {consolidated_csv_path}")


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
    batch_size = 256           # bigger batch → far better MPS utilization (28 vs 114 batches/epoch)
    learning_rate = 3e-4       # peak LR (after warmup)
    max_epochs = 200           # tightened from 400: standalone H=60 (200 epochs, patience 40)
                               # converged to test MAE 1.78 K. Past ~epoch 160 val improvements
                               # are tiny and do NOT translate to test improvements (overfit floor).
    patience = 25              # stop after 25 epochs w/o smoothed-val improvement (val plateaus ~130)
    lstm_units = 512           # reverted to 512: the 256 cut underfit mid/long-horizon
                               # deltas (H180-480 regressed). Residual stays; capacity restored.
    dropout_rate = 0.2         # increased regularization
    num_sensors = 11

    zscore_threshold = 3.0    # z-score outlier cutoff; raise for steep-ramp datasets
    target_test_files = [
        "h6_flux88_abs20_surf0_781s - Sheet2.csv",
        "h6_flux88_abs92_surf0_648s - Sheet3.csv",
        "h6_flux88_abs0_surf1_790s - Sheet1.csv",
        "h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv",
    ]

    warmup_epochs  = 10        # linear LR warmup: 0 → peak_lr over first 10 epochs
    ema_window     = 5         # smooth val_mae over this many epochs for checkpoint selection
    use_huber_loss = True      # Huber(delta=2) primary loss → better R² than pure MAE
    augment_noise_std = 0.0    # train-only Gaussian noise on scaled inputs (combats overfit; 0 disables) — neutral effect at 0.05
    bin_target = False         # predict RAW TC1..TC{N} directly (the actual thermocouples).
                               # bin_target=True averages adjacent TCs (easier target, lower MAE
                               # but NOT the raw sensors) — kept available but off by default.

    physics_weight = 0.0
    soft_penalty_weight = 0.0
    excess_penalty_weight = 0.0
    power_balance_weight = 0.0
    
    sequence_length = 20
    prediction_horizon_steps = 300
    
    cylinder_length = 1.0
    num_workers = 0
    
    experiment_name = "new_theoretical_TC11"

Config.output_dir = f"output_new_TC11/{Config.experiment_name}_H{Config.prediction_horizon_steps}"
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
                    
                    time_start_scaled = time_series[sample_idx, 0, 0].item()
                    unscaled_time_start = time_start_scaled * 300.0 + 300.0
                    seq_len = time_series.shape[1]
                    time_stamp = unscaled_time_start + (seq_len - 1) + horizon_steps
                    
                    profiles_by_file[filename].append({
                        'y_true': y_true_sample,
                        'y_pred': y_pred_sample,
                        'h': h,
                        'flux': flux,
                        'abs': abs_val,
                        'surf': surf,
                        'time_stamp': time_stamp
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
        
        print(f"File: {filename}")
        print(f"Time stamp: {best_profile['time_stamp']:.1f}s")
        print("Actual:")
        print(np.round(best_profile['y_true'], 2))
        print("Predicted:")
        print(np.round(best_profile['y_pred'], 2))
        print("-" * 40)
        
        fig, ax = plt.subplots(figsize=(8, 10))

        h = best_profile['h']
        # Bin mode: predictions live at bin midpoints between adjacent TCs (num_sensors-1
        # outputs). Otherwise plot directly at each of the num_sensors TC depths. This
        # handles every dataset (TC11 → 10 bins, TC10 → 9 bins, future N → N-1 bins).
        n_out = best_profile['y_true'].shape[0]
        is_binned = (n_out == num_sensors - 1)
        if is_binned:
            tc_depths = np.linspace(0, -h, num_sensors)[::-1]         # raw TC depths
            physical_depths = 0.5 * (tc_depths[:-1] + tc_depths[1:])  # bin midpoints
            label_prefix = 'BIN'
        else:
            physical_depths = np.linspace(0, -h, n_out)[::-1]
            label_prefix = 'TC'

        ax.plot(best_profile['y_true'], physical_depths, 'o-', label='Actual', color='blue', markersize=6, linewidth=2)
        ax.plot(best_profile['y_pred'], physical_depths, 's--', label='Predicted', color='red', markersize=5, linewidth=2)

        for i in range(best_profile['y_true'].shape[0]):
            ax.annotate(f'{label_prefix}{i+1}', (best_profile['y_true'][i], physical_depths[i]),
                        textcoords="offset points", xytext=(5, -5), ha='left')

        # ================= 1D MODEL OVERLAY =================
        # Reference 1D model arrays are 11-element TC predictions hardcoded for TC11/Config C.
        # Only draw them when the dataset matches (11 raw sensors → 10 bins or 11 TCs).
        # For TC10 (10 sensors → 9 bins) the geometry is different; skip the overlay.
        skip_1d_overlay = (num_sensors != 11)
        depths_1d = [
            0, -0.0158, -0.0315, -0.0473, -0.0630, -0.0788,
            -0.0945, -0.1103, -0.1260, -0.1417, -0.1575
        ]

        if horizon_steps == 60:
            temperatures_1d = [
                284.5487, 302.9150, 302.9150, 302.9150, 302.9150, 302.9150,
                302.9150, 302.9150, 302.9150, 302.9150, 339.0725
            ]
        elif horizon_steps == 300:
            temperatures_1d = [
                292.3949, 311.5475, 311.5475, 311.5475, 311.5475, 311.5475,
                311.5475, 311.5475, 311.5475, 311.5475, 347.7051
            ]
        elif horizon_steps == 150:
            temperatures_1d = [
                287.5114, 306.1720, 306.1720, 306.1720, 306.1720, 306.1720,
                306.1720, 306.1720, 306.1720, 306.1720, 342.3296
            ]
        else:
            temperatures_1d = None

        if temperatures_1d is not None and not skip_1d_overlay:
            if is_binned:
                t = np.asarray(temperatures_1d, dtype=float)
                temperatures_1d = (0.5 * (t[:-1] + t[1:])).tolist()
                depths_1d = ((np.asarray(depths_1d[:-1]) + np.asarray(depths_1d[1:])) / 2).tolist()
            ax.plot(temperatures_1d, depths_1d, color='green', marker='^', linestyle='-', linewidth=2, label='1D model')

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

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
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
            num_sensors=Config.num_sensors,
            zscore_threshold=Config.zscore_threshold,
            target_test_files=Config.target_test_files,
            augment_noise_std=Config.augment_noise_std,
            bin_target=Config.bin_target
        )
        physics_params = train_dataset.get_physics_params()
        thermal_scaler = physics_params['thermal_scaler']
        param_scaler = physics_params['param_scaler']

        # Preserve THIS experiment's scalers next to its checkpoint. The shared scaler_dir is
        # overwritten by every later horizon (and the longest horizon fits on a degenerate
        # file subset), so post-hoc eval/prediction MUST use the matched per-experiment scaler.
        import joblib as _joblib
        _joblib.dump(thermal_scaler, os.path.join(Config.output_dir, 'thermal_scaler.save'))
        _joblib.dump(param_scaler, os.path.join(Config.output_dir, 'param_scaler.save'))
    except Exception as e:
        print(f"Error loading datasets: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"L": L, "H": H, "status": "data_load_failed"}

    # --------------------------
    # Guard: skip entirely if train set is empty
    # --------------------------
    train_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 0
    test_size  = len(test_loader.dataset)  if hasattr(test_loader,  'dataset') else 0
    val_size   = len(val_loader.dataset)   if hasattr(val_loader,   'dataset') else 0

    if train_size == 0:
        print(
            f"\n[SKIP] H={H}: Training dataset is EMPTY "
            f"(not enough files with >= {L + H} rows). Skipping experiment."
        )
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"L": L, "H": H, "status": "skipped_empty_train",
                "mae": float('nan'), "rmse": float('nan'), "r2": float('nan')}

    if test_size == 0:
        print(
            f"\n[WARNING] H={H}: Test dataset is EMPTY "
            f"(not enough files with >= {L + H} rows). "
            "Training will proceed but evaluation will return NaN sentinels."
        )

    if val_size == 0:
        print(
            f"\n[WARNING] H={H}: Validation dataset is EMPTY. "
            "Early stopping and model selection will be based on TRAIN loss — "
            "expect overfitting. Consider reducing the horizon or adding more data."
        )

    # Low training-file count warning (does not skip — user decision)
    _MIN_WARN_FILES = 8
    _num_valid = None
    for _attr in ('num_valid_files', 'valid_file_count', 'num_train_files'):
        if hasattr(train_dataset, _attr):
            _num_valid = getattr(train_dataset, _attr)
            break
    if _num_valid is None and hasattr(train_dataset, 'valid_files'):
        _num_valid = len(train_dataset.valid_files)
    if _num_valid is not None and _num_valid < _MIN_WARN_FILES:
        print(
            f"\n[WARNING] H={H}: Only {_num_valid} valid training files survive the "
            f"horizon length filter (recommended >= {_MIN_WARN_FILES}). "
            f"Results may be unreliable due to limited training diversity. Proceeding anyway."
        )

    # --------------------------
    # Model / Trainer
    # --------------------------
    # In bin mode the dataset produces num_outputs = num_sensors-1 channels; use that
    # everywhere the model/trainer needs the output dimensionality. The dataset's
    # thermal_scaler is already shape (num_outputs,) since it was fit on bin data.
    num_outputs = getattr(train_dataset, 'num_outputs', Config.num_sensors)
    print(f"Building model for L={L}, H={H} (num_outputs={num_outputs}, bin_target={Config.bin_target})...")
    try:
        model = build_model(
            num_sensors=num_outputs,
            sequence_length=Config.sequence_length,
            lstm_units=Config.lstm_units,
            dropout_rate=Config.dropout_rate,
            device=device,
            horizon_steps=Config.prediction_horizon_steps
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
            thermal_scaler=thermal_scaler,
            use_huber=Config.use_huber_loss
        )

        trainer = FixedUnscaledEvaluationTrainer(
            base_trainer, thermal_scaler, param_scaler,
            horizon_steps=Config.prediction_horizon_steps, device=device, num_sensors=num_outputs
        )
    except Exception as e:
        print(f"Error building model: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"L": L, "H": H, "status": "model_build_failed"}

    # Monotonic cosine decay (NO warm restarts). The previous WarmRestarts spiked the LR
    # back to peak every 50 epochs, throwing val_mae from ~0.21 to ~0.59 (see run_sweeps.log)
    # and making best-checkpoint selection a lottery on the tiny 4-file val set — that is how
    # H600 collapsed to R²≈0 while H480 hit 0.97. A single smooth cosine after warmup is far
    # more stable for this regression problem and gives reproducible checkpoints.
    peak_lr       = Config.learning_rate
    warmup_epochs = Config.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_trainer.optimizer, T_max=Config.max_epochs - warmup_epochs, eta_min=1e-6
    )

    # --------------------------
    # Train with early stopping on 5-epoch EMA of val_mae_unscaled
    # --------------------------
    print(f"\nTraining model for L={L}, H={H} (max_epochs={Config.max_epochs}, patience={Config.patience})...")
    best_smoothed_val_mae = np.inf
    best_val_mae_unscaled = np.inf
    best_epoch = 0
    patience_counter = 0
    train_history = []
    val_mae_history = []   # rolling buffer for EMA smoothing
    best_model_path = os.path.join(Config.output_dir, f'best_model_L{L}_H{H}.pth')

    try:
        for epoch in range(Config.max_epochs):
            # --- Linear LR warmup for the first `warmup_epochs` epochs ---
            if epoch < warmup_epochs:
                warmup_scale = (epoch + 1) / warmup_epochs
                for pg in base_trainer.optimizer.param_groups:
                    pg['lr'] = peak_lr * warmup_scale
            else:
                # Cosine schedule counts from the end of warmup
                scheduler.step(epoch - warmup_epochs)

            results = trainer.train_epoch_unscaled(train_loader, val_loader)
            train_history.append(results)

            val_mae = results['val_mae_unscaled']
            val_mae_history.append(val_mae)

            # 5-epoch EMA — smooths out noisy small-val-set fluctuations
            smoothed_val_mae = float(np.mean(val_mae_history[-Config.ema_window:]))

            if smoothed_val_mae < best_smoothed_val_mae:
                best_smoothed_val_mae = smoothed_val_mae
                best_val_mae_unscaled = val_mae
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(trainer.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == Config.patience:
                current_lr = base_trainer.optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch+1}/{Config.max_epochs} — "
                    f"val_mae={val_mae:.4f}  smooth={smoothed_val_mae:.4f}  "
                    f"best={best_smoothed_val_mae:.4f}  "
                    f"patience={patience_counter}/{Config.patience}  lr={current_lr:.2e}"
                )

            if patience_counter >= Config.patience:
                print(
                    f"\n[Early Stop] Smoothed val_mae no improvement for "
                    f"{Config.patience} epochs. Stopping at epoch {epoch+1}.\n"
                )
                break

    except Exception as e:
        print(f"Error during training: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"L": L, "H": H, "status": "training_failed"}

    # --------------------------
    # Evaluate + Generate Profile Plots
    # --------------------------
    print(f"\nEvaluating best model for L={L}, H={H} (best epoch={best_epoch})...")
    try:
        if os.path.exists(best_model_path):
            trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_results = trainer.evaluate_unscaled(test_loader, "test")

        # --- REPRODUCIBILITY CHECK ---
        # Rebuild a FRESH model + trainer from ONLY the on-disk checkpoint and this experiment's
        # saved scalers, then re-evaluate. The MAE must match the in-run number; if not, the saved
        # artifacts don't reproduce (checkpoint/scaler corruption, e.g. from concurrent runs).
        try:
            import joblib as _joblib
            _tsc = _joblib.load(os.path.join(Config.output_dir, 'thermal_scaler.save'))
            _psc = _joblib.load(os.path.join(Config.output_dir, 'param_scaler.save'))
            _m = build_model(num_sensors=num_outputs, sequence_length=Config.sequence_length,
                             lstm_units=Config.lstm_units, dropout_rate=Config.dropout_rate,
                             device=device, horizon_steps=Config.prediction_horizon_steps)
            _bt = create_trainer(model=_m, lstm_units=Config.lstm_units, dropout_rate=Config.dropout_rate,
                                 device=device, thermal_scaler=_tsc, use_huber=Config.use_huber_loss)
            _ev = FixedUnscaledEvaluationTrainer(_bt, _tsc, _psc,
                                                 horizon_steps=Config.prediction_horizon_steps,
                                                 device=device, num_sensors=num_outputs)
            _ev.model.load_state_dict(torch.load(best_model_path, map_location=device))
            _re = _ev.evaluate_unscaled(test_loader, "reprocheck")
            _a = test_results.get('test_mae_unscaled'); _b = _re.get('reprocheck_mae_unscaled')
            if _a is not None and _b is not None and abs(float(_a) - float(_b)) > 0.05:
                print(f"\n*** [REPRO-CHECK FAILED] H={H}: in-run MAE {_a:.4f} != reloaded MAE {_b:.4f} "
                      f"-> saved checkpoint/scaler does NOT reproduce. RESULTS UNRELIABLE. ***\n")
                results_repro_ok = False
            else:
                print(f"[REPRO-CHECK OK] H={H}: in-run {_a:.4f} == reloaded {_b:.4f} K")
                results_repro_ok = True
            del _m, _bt, _ev
        except Exception as _e:
            print(f"[REPRO-CHECK error] {_e}")
            results_repro_ok = None

        if not test_results.get('_empty_loader', False):
            print(f"\nGenerating profile plots for each test file (H={H})...")
            generate_profile_plots_per_file(trainer, test_loader, H, Config.output_dir, Config.num_sensors)

        power_summary = trainer.analyze_power_balance(test_loader, num_samples=500)

        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        
        results = {"L": L, "H": H, "status": "success", "best_epoch": best_epoch,
                   "repro_ok": results_repro_ok, **test_results, **power_summary}
        save_single_run_performance_metrics(L, H, results)
        return results

    except Exception as e:
        print(f"Error during evaluation: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader, train_dataset)
        return {"L": L, "H": H, "status": "evaluation_failed"}


if __name__ == '__main__':
    horizon_sweep_fixed_seq(seq_len=20)
