import torch
import numpy as np

def get_adaptive_physics_weight(physics_loss_value, data_loss_value, epoch=0):
    """
    Adjust physics weight based on current loss balance and epoch
    """
    if data_loss_value < 1e-8:  # Avoid division by zero
        return 2.0
    
    ratio = physics_loss_value / data_loss_value
    
    # Base weight based on loss ratio
    if ratio > 10:    # Physics loss dominating
        base_weight = 0.5
    elif ratio > 5:   # Physics loss strong
        base_weight = 1.0
    elif ratio > 1:   # Balanced
        base_weight = 2.0
    else:             # Data loss dominating, boost physics
        base_weight = 5.0
    
    # Slight epoch-based adjustment (gradual increase)
    epoch_multiplier = min(1.5, 1.0 + epoch * 0.02)
    
    final_weight = base_weight * epoch_multiplier
    
    # Clamp to reasonable range
    return max(0.1, min(10.0, final_weight))

def compute_energy_conservation_loss(
    predicted_temps,
    X_batch,
    thermal_scaler,
    param_scaler,
    rho,
    cp,
    radius,
    delta_t=20.0,
    penalty_weight=10.0,
    debug=False
):
    """
    Compute energy conservation constraint loss
    Ensures predicted energy stored <= incoming energy
    
    Args:
        predicted_temps: [batch, 10] predicted temperatures (scaled)
        X_batch: [batch, seq_len, features] input batch containing q0 and other params
        thermal_scaler: scaler for temperature data
        param_scaler: scaler for parameter data
        penalty_weight: weight for constraint violation penalty
    
    Returns:
        conservation_loss: penalty for violating energy conservation
        conservation_info: dict with detailed conservation metrics
    """
    batch_size = predicted_temps.shape[0]
    
    # Extract q0 from X_batch (assuming it's in the parameter features)
    # X_batch shape: [batch, sequence_length, features]
    # Features are typically [T1, T2, ..., T10, h_total, q0, q1, q2, q3] (scaled)
    
    # Get q0 from the last timestep (most recent parameters)
    q0_scaled = X_batch[:, -1, 11]  # q0 is at index 11 (after T1-T10 and h_total)
    
    # Get h_total from the last timestep
    h_total_scaled = X_batch[:, -1, 10]  # h_total is at index 10
    
    # Inverse transform to get real values
    # Create parameter tensor for inverse transformation
    zeros_params = torch.zeros((batch_size, 3), device=X_batch.device)  # For q1, q2, q3
    scaled_params = torch.cat([
        h_total_scaled.unsqueeze(1), 
        q0_scaled.unsqueeze(1), 
        zeros_params
    ], dim=1)
    
    # Inverse transform parameters
    params_raw = param_scaler.inverse_transform(scaled_params.detach().cpu().numpy())
    h_total_raw = torch.tensor(params_raw[:, 0], device=X_batch.device, dtype=torch.float32)
    q0_raw = torch.tensor(params_raw[:, 1], device=X_batch.device, dtype=torch.float32)
    
    # Calculate A_rec (cross-sectional area)
    A_rec = torch.pi * radius ** 2
    
    # Calculate incoming energy: q0 * A_rec
    incoming_energy = q0_raw * A_rec
    
    # Calculate predicted energy stored
    # Get initial temperatures (at t=0, 20 seconds ago)
    initial_temps = X_batch[:, 0, :10]  # T at beginning of sequence
    
    # Convert temperatures to raw values
    predicted_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(predicted_temps.detach().cpu().numpy()),
        device=predicted_temps.device
    )
    initial_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(initial_temps.detach().cpu().numpy()),
        device=initial_temps.device
    )
    
    # Calculate energy stored using the same method as physics loss
    predicted_energy_stored = calculate_total_energy_stored(
        predicted_temps_raw, initial_temps_raw, h_total_raw, rho, cp, radius, delta_t
    )
    
    # Energy conservation constraint: predicted_energy_stored <= incoming_energy
    # Calculate violations (positive values indicate violations)
    energy_violations = torch.clamp(predicted_energy_stored - incoming_energy, min=0.0)
    
    # Penalty for violations
    conservation_loss = penalty_weight * torch.mean(energy_violations)
    
    # Calculate metrics for monitoring
    violation_ratio = torch.mean((predicted_energy_stored > incoming_energy).float())
    max_violation = torch.max(energy_violations)
    energy_efficiency = torch.mean(predicted_energy_stored / (incoming_energy + 1e-8))
    
    conservation_info = {
        'incoming_energy_avg': torch.mean(incoming_energy).item(),
        'predicted_energy_avg': torch.mean(predicted_energy_stored).item(),
        'violation_ratio': violation_ratio.item(),
        'max_violation': max_violation.item(),
        'energy_efficiency': energy_efficiency.item(),
        'q0_avg': torch.mean(q0_raw).item(),
        'conservation_loss': conservation_loss.item()
    }
    
    if debug:
        print(f"\n=== ENERGY CONSERVATION DEBUG ===")
        print(f"Batch size: {batch_size}")
        print(f"q0 range: {q0_raw.min().item():.3f} to {q0_raw.max().item():.3f} W/m²")
        print(f"h_total range: {h_total_raw.min().item():.3f} to {h_total_raw.max().item():.3f} m")
        print(f"A_rec: {A_rec:.6f} m²")
        print(f"Incoming energy range: {incoming_energy.min().item():.3f} to {incoming_energy.max().item():.3f} J/s")
        print(f"Predicted energy range: {predicted_energy_stored.min().item():.3f} to {predicted_energy_stored.max().item():.3f} J/s")
        print(f"Violations: {torch.sum(predicted_energy_stored > incoming_energy).item()}/{batch_size} samples")
        print(f"Max violation: {max_violation.item():.3f} J/s")
        print(f"Conservation loss: {conservation_loss.item():.3f}")
    
    return conservation_loss, conservation_info

def calculate_total_energy_stored(temp_final, temp_initial, h_total, rho, cp, radius, delta_t):
    """
    Calculate total energy stored across all bins
    """
    batch_size = temp_final.shape[0]
    
    # Calculate bin height (9 bins from 10 temperature points)
    delta_h = h_total / 9  # [batch]
    
    # Calculate cross-sectional area and mass per bin
    A = torch.pi * radius ** 2
    mass_bin = rho * A * delta_h  # [batch]
    
    total_energy = torch.zeros(batch_size, device=temp_final.device)
    
    # Loop through 9 bins (using adjacent temperature pairs)
    for i in range(9):
        # Get temperature at bin boundaries
        temp_top_final = temp_final[:, i]      # Top of bin at final time
        temp_bottom_final = temp_final[:, i+1]  # Bottom of bin at final time
        temp_top_initial = temp_initial[:, i]   # Top of bin at initial time
        temp_bottom_initial = temp_initial[:, i+1] # Bottom of bin at initial time
        
        # Calculate average temperature in the bin
        T_avg_final = (temp_top_final + temp_bottom_final) / 2
        T_avg_initial = (temp_top_initial + temp_bottom_initial) / 2
        
        # Temperature change in this bin over delta_t seconds
        delta_T = T_avg_final - T_avg_initial
        
        # Energy storage rate in this bin (J/s)
        bin_energy = mass_bin * cp * delta_T / delta_t
        
        total_energy += bin_energy
    
    return total_energy

def compute_energy_storage_loss_20sec(
    predicted_temps,
    actual_temps,
    initial_temps,  # T at t (20 seconds ago)
    thermal_scaler,
    rho,
    cp,
    h_total,
    radius,
    delta_t=20.0,  # 20 seconds
    debug=False
):
    """
    Compute physics loss using 20-second temperature changes
    
    predicted_temps: [batch, 10]  → predicted T at t+20
    actual_temps:    [batch, 10]  → actual T at t+20
    initial_temps:   [batch, 10]  → actual T at t (20 seconds ago)
    """
    
    # Convert scaled temperatures back to raw temperatures
    predicted_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(predicted_temps.detach().cpu().numpy()),
        device=predicted_temps.device
    )
    actual_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(actual_temps.detach().cpu().numpy()),
        device=actual_temps.device
    )
    initial_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(initial_temps.detach().cpu().numpy()),
        device=initial_temps.device
    )
    
    batch_size = predicted_temps_raw.shape[0]
    
    # Ensure h_total is a tensor
    if isinstance(h_total, (float, int)):
        h_total = torch.full((batch_size,), h_total, device=predicted_temps_raw.device)
    
    # Calculate bin height (9 bins from 10 temperature points)
    delta_h = h_total / 9  # [batch]
    
    # Calculate cross-sectional area and mass per bin
    A = torch.pi * radius ** 2
    mass_bin = rho * A * delta_h  # [batch]
    
    def calculate_total_energy_with_debug(temp_final, temp_initial, label=""):
        """Calculate total energy stored across all bins with detailed tracking"""
        total_energy = torch.zeros(batch_size, device=predicted_temps_raw.device)
        bin_energy_details = []
        
        if debug and label:
            print(f"\n{label} Energy Calculation (20-second intervals):")
        
        # Loop through 9 bins (using adjacent temperature pairs)
        for i in range(9):
            # Get temperature at bin boundaries
            temp_top_final = temp_final[:, i]      # Top of bin at final time
            temp_bottom_final = temp_final[:, i+1]  # Bottom of bin at final time
            temp_top_initial = temp_initial[:, i]   # Top of bin at initial time
            temp_bottom_initial = temp_initial[:, i+1] # Bottom of bin at initial time
            
            # Calculate average temperature in the bin
            T_avg_final = (temp_top_final + temp_bottom_final) / 2
            T_avg_initial = (temp_top_initial + temp_bottom_initial) / 2
            
            # Temperature change in this bin over 20 seconds
            delta_T = T_avg_final - T_avg_initial
            
            # Energy storage rate in this bin (J/s)
            # Energy = mass * cp * delta_T / delta_t
            bin_energy = mass_bin * cp * delta_T / delta_t
            
            if debug and label:
                print(f"  Bin {i+1}: T_avg_final={T_avg_final[0].item():.3f}°C, "
                      f"T_avg_initial={T_avg_initial[0].item():.3f}°C, "
                      f"delta_T={delta_T[0].item():.3f}°C, "
                      f"bin_energy={bin_energy[0].item():.3e} J/s")
            
            bin_energy_details.append({
                'bin': i+1,
                'delta_T': delta_T[0].item() if batch_size > 0 else 0,
                'energy': bin_energy[0].item() if batch_size > 0 else 0
            })
            
            total_energy += bin_energy
        
        if debug and label:
            print(f"Total {label} Energy: {total_energy[0].item():.3e} J/s")
        
        return total_energy, bin_energy_details
    
    # Calculate total energy for predicted and actual temperatures
    E_predicted, pred_bin_details = calculate_total_energy_with_debug(
        predicted_temps_raw, initial_temps_raw, "Predicted" if debug else ""
    )
    E_actual, actual_bin_details = calculate_total_energy_with_debug(
        actual_temps_raw, initial_temps_raw, "Actual" if debug else ""
    )
    
    # Calculate physics loss as mean absolute difference
    energy_diff = torch.abs(E_predicted - E_actual)
    physics_loss = torch.mean(energy_diff)
    
    if debug:
        print(f"\nEnergy Difference (first sample): {energy_diff[0].item():.3e} J/s")
        print(f"Physics Loss: {physics_loss.item():.3e} J/s")
    
    return physics_loss, {
        'predicted_energy': E_predicted[0].item() if batch_size > 0 else 0,
        'actual_energy': E_actual[0].item() if batch_size > 0 else 0,
        'energy_difference': energy_diff[0].item() if batch_size > 0 else 0,
        'pred_bin_details': pred_bin_details,
        'actual_bin_details': actual_bin_details
    }

def track_physics_metrics(predicted_temps, actual_temps, initial_temps, thermal_scaler, 
                         rho, cp, h_total, radius, delta_t=20.0):
    """
    Track detailed physics metrics for monitoring
    """
    physics_loss, debug_info = compute_energy_storage_loss_20sec(
        predicted_temps, actual_temps, initial_temps, thermal_scaler,
        rho, cp, h_total, radius, delta_t, debug=False
    )
    
    # Calculate bin-wise errors
    bin_errors = []
    for pred_bin, actual_bin in zip(debug_info['pred_bin_details'], debug_info['actual_bin_details']):
        bin_error = abs(pred_bin['energy'] - actual_bin['energy'])
        bin_errors.append(bin_error)
    
    # Energy balance ratio
    if debug_info['actual_energy'] != 0:
        energy_balance_ratio = debug_info['predicted_energy'] / debug_info['actual_energy']
    else:
        energy_balance_ratio = 1.0
    
    return {
        'total_physics_loss': physics_loss.item(),
        'max_bin_error': max(bin_errors) if bin_errors else 0,
        'avg_bin_error': np.mean(bin_errors) if bin_errors else 0,
        'energy_balance_ratio': energy_balance_ratio,
        'predicted_total_energy': debug_info['predicted_energy'],
        'actual_total_energy': debug_info['actual_energy'],
        'bin_errors': bin_errors
    }

def compute_energy_storage_loss_debug_20sec(
    predicted_temps,
    actual_temps,
    initial_temps,
    thermal_scaler,
    rho,
    cp,
    h_total,
    radius,
    delta_t=20.0
):
    """
    Debug version with detailed prints for 20-second intervals
    """
    
    batch_size = predicted_temps.shape[0]
    
    print(f"\n=== PHYSICS LOSS DEBUG (20-second intervals) ===")
    print(f"Batch size: {batch_size}")
    print(f"Device: {predicted_temps.device}")
    print(f"Delta t: {delta_t} seconds")
    print(f"Physical parameters: rho={rho}, cp={cp}, radius={radius}")
    
    if isinstance(h_total, (float, int)):
        h_total_val = h_total
    else:
        h_total_val = h_total[0].item()
    
    print(f"h_total: {h_total_val:.6f} m")
    print(f"Bin height: {h_total_val/9:.6f} m")
    print(f"Cross-sectional area: {torch.pi * radius**2:.6f} m²")
    print(f"Mass per bin: {rho * torch.pi * radius**2 * (h_total_val/9):.6f} kg")
    
    # Convert to raw temperatures for meaningful debug output
    predicted_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(predicted_temps.detach().cpu().numpy()),
        device=predicted_temps.device
    )
    actual_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(actual_temps.detach().cpu().numpy()),
        device=actual_temps.device
    )
    initial_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(initial_temps.detach().cpu().numpy()),
        device=initial_temps.device
    )
    
    print(f"\nTemperature ranges (first sample, raw °C):")
    print(f"  Initial (t=0): min={initial_temps_raw[0].min().item():.1f}°C, max={initial_temps_raw[0].max().item():.1f}°C")
    print(f"  Actual (t=20): min={actual_temps_raw[0].min().item():.1f}°C, max={actual_temps_raw[0].max().item():.1f}°C")
    print(f"  Predicted (t=20): min={predicted_temps_raw[0].min().item():.1f}°C, max={predicted_temps_raw[0].max().item():.1f}°C")
    
    # Calculate physics loss with debug
    physics_loss, debug_info = compute_energy_storage_loss_20sec(
        predicted_temps, actual_temps, initial_temps, thermal_scaler,
        rho, cp, h_total, radius, delta_t, debug=True
    )
    
    print(f"\nFinal Physics Loss: {physics_loss.item():.3e} J/s")
    print(f"Energy Balance Ratio: {debug_info['predicted_energy'] / debug_info['actual_energy']:.3f}")
    
    return physics_loss