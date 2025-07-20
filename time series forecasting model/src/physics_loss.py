import torch
import numpy as np

def get_adaptive_physics_weight(physics_loss_value, data_loss_value, epoch=0):
    """
    Adjust physics weight based on current loss balance and epoch
    """
    if data_loss_value < 1e-8:  # Avoid division by zero
        return 2.0
    
    ratio = physics_loss_value / data_loss_value
    
    # Base weight based on loss ratio - UPDATED for better balance
    if ratio > 20:    # Physics loss dominating heavily
        base_weight = 0.1
    elif ratio > 10:  # Physics loss dominating
        base_weight = 0.3
    elif ratio > 5:   # Physics loss strong
        base_weight = 0.8
    elif ratio > 1:   # Balanced
        base_weight = 2.0
    elif ratio > 0.1: # Data loss stronger
        base_weight = 5.0
    else:             # Data loss dominating heavily, boost physics significantly
        base_weight = 10.0
    
    # More aggressive epoch-based adjustment
    epoch_multiplier = min(2.0, 1.0 + epoch * 0.05)
    
    final_weight = base_weight * epoch_multiplier
    
    # Clamp to reasonable range
    return max(0.01, min(20.0, final_weight))

def compute_physics_loss_improved(
    predicted_temps,
    actual_temps,
    initial_temps,
    thermal_scaler,
    rho,
    cp,
    h_total,
    radius,
    delta_t=20.0,
    debug=False
):
    """
    IMPROVED physics loss with better numerical stability and multiple constraints
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
    
    # CONSTRAINT 1: Energy Storage Physics Loss
    def calculate_energy_storage_rate(temp_final, temp_initial):
        """Calculate energy storage rate for each bin"""
        total_energy = torch.zeros(batch_size, device=predicted_temps_raw.device)
        
        # Loop through 9 bins (using adjacent temperature pairs)
        for i in range(9):
            # Get temperature at bin boundaries
            temp_top_final = temp_final[:, i]
            temp_bottom_final = temp_final[:, i+1]
            temp_top_initial = temp_initial[:, i]
            temp_bottom_initial = temp_initial[:, i+1]
            
            # Calculate average temperature in the bin
            T_avg_final = (temp_top_final + temp_bottom_final) / 2
            T_avg_initial = (temp_top_initial + temp_bottom_initial) / 2
            
            # Temperature change in this bin over delta_t seconds
            delta_T = T_avg_final - T_avg_initial
            
            # Energy storage rate in this bin (J/s)
            bin_energy = mass_bin * cp * delta_T / delta_t
            total_energy += bin_energy
        
        return total_energy
    
    # Calculate energy storage rates
    E_predicted = calculate_energy_storage_rate(predicted_temps_raw, initial_temps_raw)
    E_actual = calculate_energy_storage_rate(actual_temps_raw, initial_temps_raw)
    
    # Energy storage physics loss - use relative error for better scaling
    energy_diff = torch.abs(E_predicted - E_actual)
    energy_scale = torch.abs(E_actual) + 1.0  # Add 1.0 to avoid division by zero
    energy_loss = torch.mean(energy_diff / energy_scale)
    
    # CONSTRAINT 2: Temperature Gradient Physics Loss
    # Physical expectation: temperature should generally decrease with depth
    pred_temp_gradients = []
    actual_temp_gradients = []
    
    for i in range(9):  # 9 temperature differences between adjacent points
        pred_grad = predicted_temps_raw[:, i] - predicted_temps_raw[:, i+1]
        actual_grad = actual_temps_raw[:, i] - actual_temps_raw[:, i+1]
        pred_temp_gradients.append(pred_grad)
        actual_temp_gradients.append(actual_grad)
    
    pred_gradients = torch.stack(pred_temp_gradients, dim=1)  # [batch, 9]
    actual_gradients = torch.stack(actual_temp_gradients, dim=1)  # [batch, 9]
    
    # Gradient physics loss
    gradient_diff = torch.abs(pred_gradients - actual_gradients)
    gradient_loss = torch.mean(gradient_diff)
    
    # CONSTRAINT 3: Temperature Bounds Physics Loss
    # Temperatures should be within reasonable physical bounds
    temp_bounds_loss = 0.0
    
    # Penalize extreme temperatures (outside 0-100°C range)
    extreme_cold = torch.clamp(0.0 - predicted_temps_raw, min=0.0)
    extreme_hot = torch.clamp(predicted_temps_raw - 100.0, min=0.0)
    temp_bounds_loss = torch.mean(extreme_cold + extreme_hot)
    
    # CONSTRAINT 4: Temporal Consistency Loss
    # Temperature changes should be smooth and physically reasonable
    temp_change_pred = predicted_temps_raw - initial_temps_raw
    temp_change_actual = actual_temps_raw - initial_temps_raw
    
    # Penalize unreasonably large temperature changes (>50°C in 20 seconds)
    extreme_changes_pred = torch.clamp(torch.abs(temp_change_pred) - 50.0, min=0.0)
    temporal_consistency_loss = torch.mean(extreme_changes_pred)
    
    # Combine all physics losses with appropriate weights
    total_physics_loss = (
        1.0 * energy_loss +                    # Primary energy physics
        0.3 * gradient_loss +                  # Gradient consistency
        10.0 * temp_bounds_loss +              # Strong penalty for unphysical temps
        2.0 * temporal_consistency_loss        # Temporal smoothness
    )
    
    if debug:
        print(f"\n=== IMPROVED PHYSICS LOSS DEBUG ===")
        print(f"Energy Loss: {energy_loss.item():.6f}")
        print(f"Gradient Loss: {gradient_loss.item():.6f}")
        print(f"Temperature Bounds Loss: {temp_bounds_loss.item():.6f}")
        print(f"Temporal Consistency Loss: {temporal_consistency_loss.item():.6f}")
        print(f"Total Physics Loss: {total_physics_loss.item():.6f}")
        
        print(f"\nEnergy Statistics:")
        print(f"  Predicted Energy Range: {E_predicted.min().item():.2f} to {E_predicted.max().item():.2f} J/s")
        print(f"  Actual Energy Range: {E_actual.min().item():.2f} to {E_actual.max().item():.2f} J/s")
        
        print(f"\nTemperature Statistics:")
        print(f"  Predicted Temp Range: {predicted_temps_raw.min().item():.1f} to {predicted_temps_raw.max().item():.1f} °C")
        print(f"  Actual Temp Range: {actual_temps_raw.min().item():.1f} to {actual_temps_raw.max().item():.1f} °C")
        
        # Check for extreme temperature changes
        max_temp_change = torch.max(torch.abs(temp_change_pred)).item()
        print(f"  Max Predicted Temp Change: {max_temp_change:.1f} °C")
    
    debug_info = {
        'predicted_energy': E_predicted[0].item() if batch_size > 0 else 0,
        'actual_energy': E_actual[0].item() if batch_size > 0 else 0,
        'energy_loss': energy_loss.item(),
        'gradient_loss': gradient_loss.item(),
        'temp_bounds_loss': temp_bounds_loss.item(),
        'temporal_consistency_loss': temporal_consistency_loss.item(),
        'total_physics_loss': total_physics_loss.item()
    }
    
    return total_physics_loss, debug_info

def compute_enhanced_energy_conservation_loss(
    predicted_temps,
    X_batch,
    thermal_scaler,
    param_scaler,
    rho,
    cp,
    radius,
    delta_t=20.0,
    penalty_weight=5.0,  # Reduced from 10.0
    debug=False
):
    """
    ENHANCED energy conservation with more realistic constraints
    """
    batch_size = predicted_temps.shape[0]
    
    # Extract q0 and h_total from X_batch
    q0_scaled = X_batch[:, -1, 11]  # q0 is at index 11
    h_total_scaled = X_batch[:, -1, 10]  # h_total is at index 10
    
    # Inverse transform to get real values
    zeros_params = torch.zeros((batch_size, 3), device=X_batch.device)
    scaled_params = torch.cat([
        h_total_scaled.unsqueeze(1), 
        q0_scaled.unsqueeze(1), 
        zeros_params
    ], dim=1)
    
    params_raw = param_scaler.inverse_transform(scaled_params.detach().cpu().numpy())
    h_total_raw = torch.tensor(params_raw[:, 0], device=X_batch.device, dtype=torch.float32)
    q0_raw = torch.tensor(params_raw[:, 1], device=X_batch.device, dtype=torch.float32)
    
    # Calculate incoming energy
    A_rec = torch.pi * radius ** 2
    incoming_energy = q0_raw * A_rec  # J/s
    
    # Calculate predicted energy stored
    initial_temps = X_batch[:, 0, :10]
    
    predicted_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(predicted_temps.detach().cpu().numpy()),
        device=predicted_temps.device
    )
    initial_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(initial_temps.detach().cpu().numpy()),
        device=initial_temps.device
    )
    
    predicted_energy_stored = calculate_total_energy_stored(
        predicted_temps_raw, initial_temps_raw, h_total_raw, rho, cp, radius, delta_t
    )
    
    # ENHANCED CONSERVATION CONSTRAINTS
    
    # 1. Energy cannot exceed incoming energy (soft constraint with tolerance)
    energy_tolerance = 0.1 * incoming_energy  # 10% tolerance
    energy_violations = torch.clamp(
        predicted_energy_stored - (incoming_energy + energy_tolerance), 
        min=0.0
    )
    
    # 2. Energy efficiency should be reasonable (between 0 and 100%)
    energy_efficiency = predicted_energy_stored / (incoming_energy + 1e-8)
    efficiency_penalty = torch.clamp(energy_efficiency - 1.0, min=0.0)  # Penalize >100% efficiency
    
    # 3. Energy should be positive (can't remove more energy than available)
    negative_energy_penalty = torch.clamp(-predicted_energy_stored, min=0.0)
    
    # Combined conservation loss with different weights
    conservation_loss = (
        penalty_weight * torch.mean(energy_violations) +           # Primary constraint
        2.0 * torch.mean(efficiency_penalty) +                    # Efficiency constraint
        5.0 * torch.mean(negative_energy_penalty)                 # Physical realizability
    )
    
    # Calculate metrics
    violation_ratio = torch.mean((predicted_energy_stored > incoming_energy + energy_tolerance).float())
    max_violation = torch.max(energy_violations)
    avg_efficiency = torch.mean(energy_efficiency)
    
    conservation_info = {
        'incoming_energy_avg': torch.mean(incoming_energy).item(),
        'predicted_energy_avg': torch.mean(predicted_energy_stored).item(),
        'violation_ratio': violation_ratio.item(),
        'max_violation': max_violation.item(),
        'energy_efficiency': avg_efficiency.item(),
        'q0_avg': torch.mean(q0_raw).item(),
        'conservation_loss': conservation_loss.item(),
        'negative_energy_samples': torch.sum(predicted_energy_stored < 0).item()
    }
    
    if debug:
        print(f"\n=== ENHANCED CONSERVATION DEBUG ===")
        print(f"Incoming energy range: {incoming_energy.min().item():.1f} to {incoming_energy.max().item():.1f} J/s")
        print(f"Predicted energy range: {predicted_energy_stored.min().item():.1f} to {predicted_energy_stored.max().item():.1f} J/s")
        print(f"Energy efficiency range: {energy_efficiency.min().item():.2f} to {energy_efficiency.max().item():.2f}")
        print(f"Violations (with tolerance): {torch.sum(predicted_energy_stored > incoming_energy + energy_tolerance).item()}/{batch_size}")
        print(f"Negative energy samples: {conservation_info['negative_energy_samples']}")
        print(f"Conservation loss: {conservation_loss.item():.4f}")
    
    return conservation_loss, conservation_info

def calculate_total_energy_stored(temp_final, temp_initial, h_total, rho, cp, radius, delta_t):
    """
    Calculate total energy stored across all bins (same as original)
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
        temp_top_final = temp_final[:, i]
        temp_bottom_final = temp_final[:, i+1]
        temp_top_initial = temp_initial[:, i]
        temp_bottom_initial = temp_initial[:, i+1]
        
        # Calculate average temperature in the bin
        T_avg_final = (temp_top_final + temp_bottom_final) / 2
        T_avg_initial = (temp_top_initial + temp_bottom_initial) / 2
        
        # Temperature change in this bin over delta_t seconds
        delta_T = T_avg_final - T_avg_initial
        
        # Energy storage rate in this bin (J/s)
        bin_energy = mass_bin * cp * delta_T / delta_t
        
        total_energy += bin_energy
    
    return total_energy

def track_enhanced_physics_metrics(predicted_temps, actual_temps, initial_temps, thermal_scaler, 
                                 rho, cp, h_total, radius, delta_t=20.0):
    """
    Track enhanced physics metrics including all constraint components
    """
    physics_loss, debug_info = compute_physics_loss_improved(
        predicted_temps, actual_temps, initial_temps, thermal_scaler,
        rho, cp, h_total, radius, delta_t, debug=False
    )
    
    return {
        'total_physics_loss': physics_loss.item(),
        'energy_loss': debug_info['energy_loss'],
        'gradient_loss': debug_info['gradient_loss'],
        'temp_bounds_loss': debug_info['temp_bounds_loss'],
        'temporal_consistency_loss': debug_info['temporal_consistency_loss'],
        'predicted_total_energy': debug_info['predicted_energy'],
        'actual_total_energy': debug_info['actual_energy']
    }