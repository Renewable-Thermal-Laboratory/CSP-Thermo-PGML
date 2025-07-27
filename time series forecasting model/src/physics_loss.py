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
    
    # CONSTRAINT 1: Power Storage Physics Loss (CORRECTED UNITS)
    def calculate_power_storage_rate(temp_final, temp_initial):
        """Calculate power storage rate for each bin (J/s)"""
        total_power = torch.zeros(batch_size, device=predicted_temps_raw.device)
        
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
            
            # Power storage rate in this bin (J/s)
            bin_power = mass_bin * cp * delta_T / delta_t
            total_power += bin_power
        
        return total_power
    
    # Calculate power storage rates
    P_predicted = calculate_power_storage_rate(predicted_temps_raw, initial_temps_raw)
    P_actual = calculate_power_storage_rate(actual_temps_raw, initial_temps_raw)
    
    # Physics-based scaling for power loss
    P_char = torch.maximum(torch.abs(P_actual), 0.01 * torch.max(torch.abs(P_actual)))
    power_diff = torch.abs(P_predicted - P_actual)
    power_loss = torch.mean(power_diff / P_char)
    
    # CONSTRAINT 2: Heat Conduction Physics Loss (NEW)
    # Based on Fourier's law: q = -k * dT/dx
    # Temperature gradient should be consistent with heat flux
    thermal_conductivity = 0.6  # W/(m·K) - typical value for water/thermal storage
    
    pred_temp_gradients = []
    actual_temp_gradients = []
    
    for i in range(9):  # 9 temperature differences between adjacent points
        # Calculate temperature gradient (K/m)
        pred_grad = (predicted_temps_raw[:, i] - predicted_temps_raw[:, i+1]) / (delta_h + torch.tensor(1e-8, device=delta_h.device))
        actual_grad = (actual_temps_raw[:, i] - actual_temps_raw[:, i+1]) / (delta_h + torch.tensor(1e-8, device=delta_h.device))
        pred_temp_gradients.append(pred_grad)
        actual_temp_gradients.append(actual_grad)
    
    pred_gradients = torch.stack(pred_temp_gradients, dim=1)  # [batch, 9]
    actual_gradients = torch.stack(actual_temp_gradients, dim=1)  # [batch, 9]
    
    # Physics-based scaling for gradient loss
    grad_char = torch.maximum(torch.abs(actual_gradients), 
                             torch.tensor(0.1, device=actual_gradients.device))  # 0.1 K/m minimum
    gradient_diff = torch.abs(pred_gradients - actual_gradients)
    gradient_loss = torch.mean(gradient_diff / grad_char)
    
    # CONSTRAINT 3: Temperature Bounds Physics Loss (IMPROVED)
    # Context-dependent bounds based on initial conditions
    initial_temp_range = torch.max(initial_temps_raw) - torch.min(initial_temps_raw)
    max_reasonable_change = 2.0 * initial_temp_range + torch.tensor(10.0, device=predicted_temps_raw.device)  # More realistic bounds
    
    temp_change_pred = torch.abs(predicted_temps_raw - initial_temps_raw)
    temp_bounds_violations = torch.clamp(temp_change_pred - max_reasonable_change, min=0.0)
    temp_bounds_loss = torch.mean(temp_bounds_violations)
    
    # CONSTRAINT 4: Temporal Consistency Loss (IMPROVED)
    # Based on thermal diffusivity: α = k/(ρ*cp)
    alpha = thermal_conductivity / (rho * cp)  # m²/s
    
    # Maximum temperature change based on thermal diffusion
    alpha_tensor = torch.tensor(alpha, device=predicted_temps_raw.device)
    delta_t_tensor = torch.tensor(delta_t, device=predicted_temps_raw.device)
    max_diffusion_change = 2.0 * torch.sqrt(alpha_tensor * delta_t_tensor)  # Diffusion length scale
    
    temp_change_pred = torch.abs(predicted_temps_raw - initial_temps_raw)
    temp_change_actual = torch.abs(actual_temps_raw - initial_temps_raw)
    
    # Penalize predictions that exceed physical diffusion limits
    diffusion_violations = torch.clamp(temp_change_pred - max_diffusion_change, min=0.0)
    temporal_consistency_loss = torch.mean(diffusion_violations)
    
    # CONSTRAINT 5: Heat Flux Continuity (NEW)
    # Heat flux should be continuous between adjacent bins
    thermal_conductivity_tensor = torch.tensor(thermal_conductivity, device=predicted_temps_raw.device)
    heat_flux_pred = -thermal_conductivity_tensor * pred_gradients  # W/m²
    heat_flux_actual = -thermal_conductivity_tensor * actual_gradients  # W/m²
    
    # Check continuity between adjacent bins
    if pred_gradients.shape[1] > 1:
        flux_continuity_pred = torch.abs(heat_flux_pred[:, :-1] - heat_flux_pred[:, 1:])
        flux_continuity_actual = torch.abs(heat_flux_actual[:, :-1] - heat_flux_actual[:, 1:])
        
        # Scale by characteristic heat flux
        flux_char = torch.maximum(torch.abs(heat_flux_actual), 
                                 torch.tensor(1.0, device=heat_flux_actual.device))  # 1 W/m² minimum
        flux_continuity_diff = torch.abs(flux_continuity_pred - flux_continuity_actual)
        flux_continuity_loss = torch.mean(flux_continuity_diff / flux_char[:, :-1])
    else:
        flux_continuity_loss = torch.tensor(0.0, device=predicted_temps_raw.device)
    
    # Combine all physics losses with physics-based weights
    total_physics_loss = (
        1.0 * power_loss +                     # Primary power balance
        0.5 * gradient_loss +                  # Heat conduction consistency
        2.0 * temp_bounds_loss +               # Physical temperature bounds
        1.0 * temporal_consistency_loss +      # Thermal diffusion limits
        0.3 * flux_continuity_loss            # Heat flux continuity
    )
    
    if debug:
        print(f"\n=== IMPROVED PHYSICS LOSS DEBUG ===")
        print(f"Power Loss: {power_loss.item():.6f}")
        print(f"Gradient Loss: {gradient_loss.item():.6f}")
        print(f"Temperature Bounds Loss: {temp_bounds_loss.item():.6f}")
        print(f"Temporal Consistency Loss: {temporal_consistency_loss.item():.6f}")
        print(f"Flux Continuity Loss: {flux_continuity_loss.item():.6f}")
        print(f"Total Physics Loss: {total_physics_loss.item():.6f}")
        
        print(f"\nPower Statistics:")
        print(f"  Predicted Power Range: {P_predicted.min().item():.2f} to {P_predicted.max().item():.2f} J/s")
        print(f"  Actual Power Range: {P_actual.min().item():.2f} to {P_actual.max().item():.2f} J/s")
        
        print(f"\nTemperature Statistics:")
        print(f"  Predicted Temp Range: {predicted_temps_raw.min().item():.1f} to {predicted_temps_raw.max().item():.1f} °C")
        print(f"  Actual Temp Range: {actual_temps_raw.min().item():.1f} to {actual_temps_raw.max().item():.1f} °C")
        
        # Check for extreme temperature changes
        max_temp_change = torch.max(temp_change_pred).item()
        print(f"  Max Predicted Temp Change: {max_temp_change:.1f} °C")
        print(f"  Max Reasonable Change Limit: {max_reasonable_change.max().item():.1f} °C")
    
    debug_info = {
        'predicted_power': P_predicted[0].item() if batch_size > 0 else 0,
        'actual_power': P_actual[0].item() if batch_size > 0 else 0,
        'power_loss': power_loss.item(),
        'gradient_loss': gradient_loss.item(),
        'temp_bounds_loss': temp_bounds_loss.item(),
        'temporal_consistency_loss': temporal_consistency_loss.item(),
        'flux_continuity_loss': flux_continuity_loss.item(),
        'total_physics_loss': total_physics_loss.item()
    }
    
    return total_physics_loss, debug_info

def compute_enhanced_power_conservation_loss(
    predicted_temps,
    X_batch,
    thermal_scaler,
    param_scaler,
    rho,
    cp,
    radius,
    delta_t=20.0,
    penalty_weight=5.0,
    debug=False
):
    """
    ENHANCED power conservation with physics-based scaling (CORRECTED NAMING)
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
    
    # Calculate incoming power (J/s)
    A_rec = torch.pi * radius ** 2
    incoming_power = q0_raw * A_rec  # J/s (Watts)
    
    # Calculate predicted power stored (J/s)
    initial_temps = X_batch[:, 0, :10]
    
    predicted_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(predicted_temps.detach().cpu().numpy()),
        device=predicted_temps.device
    )
    initial_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(initial_temps.detach().cpu().numpy()),
        device=initial_temps.device
    )
    
    predicted_power_stored = calculate_total_power_stored(
        predicted_temps_raw, initial_temps_raw, h_total_raw, rho, cp, radius, delta_t
    )
    
    # ENHANCED CONSERVATION CONSTRAINTS with physics-based scaling
    
    # Characteristic power scale for normalization
    P_char = torch.maximum(torch.abs(incoming_power), 0.01 * torch.max(torch.abs(incoming_power)))
    
    # 1. Power cannot exceed incoming power (soft constraint with tolerance)
    power_tolerance = 0.15 * incoming_power  # 15% tolerance for transient effects
    power_violations = torch.clamp(
        predicted_power_stored - (incoming_power + power_tolerance), 
        min=0.0
    )
    
    # 2. Power efficiency should be reasonable (between 0 and 120% for transients)
    power_efficiency = predicted_power_stored / (incoming_power + 1e-8)
    efficiency_penalty = torch.clamp(power_efficiency - 1.2, min=0.0)  # Allow 120% for transients
    
    # 3. Power should be physically reasonable (prevent extreme negative values)
    # Allow some negative power for cooling/heat loss
    extreme_negative_power = torch.clamp(-predicted_power_stored - 0.5 * incoming_power, min=0.0)
    
    # 4. Power balance consistency check
    power_imbalance = torch.abs(predicted_power_stored - incoming_power)
    normalized_imbalance = power_imbalance / P_char
    
    # Combined conservation loss with physics-based scaling
    conservation_loss = (
        penalty_weight * torch.mean(power_violations / P_char) +           # Scaled violation penalty
        2.0 * torch.mean(efficiency_penalty) +                           # Efficiency constraint
        3.0 * torch.mean(extreme_negative_power / P_char) +               # Scaled negative power penalty
        1.0 * torch.mean(normalized_imbalance)                           # General power balance
    )
    
    # Calculate metrics
    violation_ratio = torch.mean((predicted_power_stored > incoming_power + power_tolerance).float())
    max_violation = torch.max(power_violations)
    avg_efficiency = torch.mean(power_efficiency)
    
    conservation_info = {
        'incoming_power_avg': torch.mean(incoming_power).item(),
        'predicted_power_avg': torch.mean(predicted_power_stored).item(),
        'violation_ratio': violation_ratio.item(),
        'max_violation': max_violation.item(),
        'power_efficiency': avg_efficiency.item(),
        'q0_avg': torch.mean(q0_raw).item(),
        'conservation_loss': conservation_loss.item(),
        'extreme_negative_samples': torch.sum(predicted_power_stored < -0.5 * incoming_power).item()
    }
    
    if debug:
        print(f"\n=== ENHANCED POWER CONSERVATION DEBUG ===")
        print(f"Incoming power range: {incoming_power.min().item():.1f} to {incoming_power.max().item():.1f} J/s")
        print(f"Predicted power range: {predicted_power_stored.min().item():.1f} to {predicted_power_stored.max().item():.1f} J/s")
        print(f"Power efficiency range: {power_efficiency.min().item():.2f} to {power_efficiency.max().item():.2f}")
        print(f"Violations (with tolerance): {torch.sum(predicted_power_stored > incoming_power + power_tolerance).item()}/{batch_size}")
        print(f"Extreme negative power samples: {conservation_info['extreme_negative_samples']}")
        print(f"Conservation loss: {conservation_loss.item():.4f}")
    
    return conservation_loss, conservation_info

def calculate_total_power_stored(temp_final, temp_initial, h_total, rho, cp, radius, delta_t):
    """
    Calculate total power stored across all bins (J/s) - CORRECTED NAMING
    """
    batch_size = temp_final.shape[0]
    
    # Calculate bin height (9 bins from 10 temperature points)
    delta_h = h_total / 9  # [batch]
    
    # Calculate cross-sectional area and mass per bin
    A = torch.pi * radius ** 2
    mass_bin = rho * A * delta_h  # [batch]
    
    total_power = torch.zeros(batch_size, device=temp_final.device)
    
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
        
        # Power storage rate in this bin (J/s)
        bin_power = mass_bin * cp * delta_T / delta_t
        
        total_power += bin_power
    
    return total_power

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
        'power_loss': debug_info['power_loss'],
        'gradient_loss': debug_info['gradient_loss'],
        'temp_bounds_loss': debug_info['temp_bounds_loss'],
        'temporal_consistency_loss': debug_info['temporal_consistency_loss'],
        'flux_continuity_loss': debug_info['flux_continuity_loss'],
        'predicted_total_power': debug_info['predicted_power'],
        'actual_total_power': debug_info['actual_power']
    }