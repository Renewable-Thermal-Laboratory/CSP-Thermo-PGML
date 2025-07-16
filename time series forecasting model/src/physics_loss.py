import torch

def compute_energy_storage_loss(
    predicted_temps,
    actual_temps,
    prev_actual_temps,
    rho,
    cp,
    h_total,
    radius,
    delta_t
):
    """
    Compute physics loss based on difference in total stored energy
    between predicted and actual temperature profiles.

    predicted_temps: [batch, 10]  → predicted T at t+1
    actual_temps:    [batch, 10]  → actual T at t+1
    prev_actual_temps: [batch, 10] → actual T at t
    
    Formula: total_energy_stored = mass * cp * (T_final - T_initial) / (time_final - time_initial)
    """
    
    batch_size = predicted_temps.shape[0]
    
    # Ensure h_total is a tensor
    if isinstance(h_total, float) or isinstance(h_total, int):
        h_total = torch.full((batch_size,), h_total, device=predicted_temps.device)
    
    # Calculate bin height (9 bins from 10 temperature points)
    delta_h = h_total / 9  # [batch]
    
    # Calculate cross-sectional area and mass per bin
    A = torch.pi * radius ** 2
    mass_bin = rho * A * delta_h  # [batch]
    
    def calculate_total_energy(temp_current, temp_previous):
        """Calculate total energy stored across all bins"""
        total_energy = torch.zeros(batch_size, device=predicted_temps.device)
        
        # Loop through 9 bins (using adjacent temperature pairs)
        for i in range(9):
            # Get temperature at bin boundaries
            # Assuming TC1 is at surface (index 0) and TC10 is at bottom (index 9)
            temp_top_current = temp_current[:, i]      # Top of bin at current time
            temp_bottom_current = temp_current[:, i+1]  # Bottom of bin at current time
            temp_top_previous = temp_previous[:, i]     # Top of bin at previous time
            temp_bottom_previous = temp_previous[:, i+1] # Bottom of bin at previous time
            
            # Calculate average temperature in the bin
            T_avg_current = (temp_top_current + temp_bottom_current) / 2
            T_avg_previous = (temp_top_previous + temp_bottom_previous) / 2
            
            # Temperature change in this bin
            delta_T = T_avg_current - T_avg_previous
            
            # Energy stored in this bin
            # Using: energy = mass * cp * delta_T / delta_t
            bin_energy = mass_bin * cp * delta_T / delta_t
            
            total_energy += bin_energy
        
        return total_energy
    
    # Calculate total energy for predicted and actual temperatures
    E_predicted = calculate_total_energy(predicted_temps, prev_actual_temps)
    E_actual = calculate_total_energy(actual_temps, prev_actual_temps)
    
    # Calculate physics loss as mean absolute difference
    physics_loss = torch.mean(torch.abs(E_predicted - E_actual))
    
    return physics_loss

def compute_energy_storage_loss_debug_detailed(
    predicted_temps,
    actual_temps,
    prev_actual_temps,
    rho,
    cp,
    h_total,
    radius,
    delta_t
):
    """
    Detailed debug version that prints all intermediate values
    """
    
    batch_size = predicted_temps.shape[0]
    
    # Ensure h_total is a tensor
    if isinstance(h_total, float) or isinstance(h_total, int):
        h_total = torch.full((batch_size,), h_total, device=predicted_temps.device)
    
    delta_h = h_total / 9
    A = torch.pi * radius ** 2
    mass_bin = rho * A * delta_h
    
    print(f"\n=== PHYSICS LOSS DEBUG ===")
    print(f"Batch size: {batch_size}")
    print(f"Device: {predicted_temps.device}")
    print(f"h_total: {h_total[0].item():.6f}")
    print(f"delta_h: {delta_h[0].item():.6f}")
    print(f"Area: {A:.6f}")
    print(f"Mass per bin: {mass_bin[0].item():.6f}")
    print(f"rho: {rho}, cp: {cp}, radius: {radius}, delta_t: {delta_t}")
    
    # Print temperature ranges for first sample
    print(f"\nTemperature ranges (first sample):")
    print(f"  Previous actual: min={prev_actual_temps[0].min().item():.3f}, max={prev_actual_temps[0].max().item():.3f}")
    print(f"  Current actual: min={actual_temps[0].min().item():.3f}, max={actual_temps[0].max().item():.3f}")
    print(f"  Current predicted: min={predicted_temps[0].min().item():.3f}, max={predicted_temps[0].max().item():.3f}")
    
    # Check if temperatures are scaled
    print(f"\nTemperature values (first sample):")
    print(f"  Previous actual: {prev_actual_temps[0].tolist()}")
    print(f"  Current actual: {actual_temps[0].tolist()}")
    print(f"  Current predicted: {predicted_temps[0].tolist()}")
    
    def calculate_total_energy_debug(temp_current, temp_previous, label):
        total_energy = torch.zeros(batch_size, device=predicted_temps.device)
        
        print(f"\n{label} Energy Calculation:")
        for i in range(9):
            temp_top_current = temp_current[:, i]
            temp_bottom_current = temp_current[:, i+1]
            temp_top_previous = temp_previous[:, i]
            temp_bottom_previous = temp_previous[:, i+1]
            
            T_avg_current = (temp_top_current + temp_bottom_current) / 2
            T_avg_previous = (temp_top_previous + temp_bottom_previous) / 2
            
            delta_T = T_avg_current - T_avg_previous
            bin_energy = mass_bin * cp * delta_T / delta_t
            
            print(f"  Bin {i+1}: T_avg_curr={T_avg_current[0].item():.6f}, "
                  f"T_avg_prev={T_avg_previous[0].item():.6f}, "
                  f"delta_T={delta_T[0].item():.6f}, "
                  f"bin_energy={bin_energy[0].item():.6e}")
            
            total_energy += bin_energy
        
        print(f"Total {label} Energy: {total_energy[0].item():.6e}")
        return total_energy
    
    E_predicted = calculate_total_energy_debug(predicted_temps, prev_actual_temps, "Predicted")
    E_actual = calculate_total_energy_debug(actual_temps, prev_actual_temps, "Actual")
    
    energy_diff = torch.abs(E_predicted - E_actual)
    physics_loss = torch.mean(energy_diff)
    
    print(f"\nEnergy Difference (first sample): {energy_diff[0].item():.6e}")
    print(f"Physics Loss: {physics_loss.item():.6e}")
    
    # Check if the difference is too small
    if physics_loss.item() < 1e-10:
        print("WARNING: Physics loss is very small! This might indicate:")
        print("1. Temperatures are scaled (normalized)")
        print("2. Temperature differences are too small")
        print("3. Model predictions are too close to actual values")
        print("4. Need to work with raw (unscaled) temperatures")
    
    return physics_loss

def compute_energy_storage_loss_raw_temps(
    predicted_temps_scaled,
    actual_temps_scaled,
    prev_actual_temps_scaled,
    thermal_scaler,
    rho,
    cp,
    h_total,
    radius,
    delta_t
):
    """
    Physics loss function that works with raw (unscaled) temperatures
    """
    
    # Convert scaled temperatures back to raw temperatures
    predicted_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(predicted_temps_scaled.detach().cpu().numpy()),
        device=predicted_temps_scaled.device
    )
    actual_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(actual_temps_scaled.detach().cpu().numpy()),
        device=actual_temps_scaled.device
    )
    prev_actual_temps_raw = torch.tensor(
        thermal_scaler.inverse_transform(prev_actual_temps_scaled.detach().cpu().numpy()),
        device=prev_actual_temps_scaled.device
    )
    
    batch_size = predicted_temps_raw.shape[0]
    
    # Ensure h_total is a tensor
    if isinstance(h_total, float) or isinstance(h_total, int):
        h_total = torch.full((batch_size,), h_total, device=predicted_temps_raw.device)
    
    delta_h = h_total / 9
    A = torch.pi * radius ** 2
    mass_bin = rho * A * delta_h
    
    def calculate_total_energy(temp_current, temp_previous):
        total_energy = torch.zeros(batch_size, device=predicted_temps_raw.device)
        
        for i in range(9):
            temp_top_current = temp_current[:, i]
            temp_bottom_current = temp_current[:, i+1]
            temp_top_previous = temp_previous[:, i]
            temp_bottom_previous = temp_previous[:, i+1]
            
            T_avg_current = (temp_top_current + temp_bottom_current) / 2
            T_avg_previous = (temp_top_previous + temp_bottom_previous) / 2
            
            delta_T = T_avg_current - T_avg_previous
            bin_energy = mass_bin * cp * delta_T / delta_t
            
            total_energy += bin_energy
        
        return total_energy
    
    E_predicted = calculate_total_energy(predicted_temps_raw, prev_actual_temps_raw)
    E_actual = calculate_total_energy(actual_temps_raw, prev_actual_temps_raw)
    
    physics_loss = torch.mean(torch.abs(E_predicted - E_actual))
    
    return physics_loss