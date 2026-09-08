import os
import sys

# Add src to python path if needed
sys.path.append(os.path.abspath("src"))

from src.train import Config, horizon_sweep_fixed_seq

def main():
    horizons = [60, 180, 300, 480, 600, 900]
    
    # ---------------------------------------------
    # 1. Run TC11 Sweep (11 sensors)
    # ---------------------------------------------
    print("\n" + "="*80)
    print("STARTING SWEEP: TC11 (11 SENSORS)")
    print("="*80)
    
    Config.data_dir = "data/output_with_TC11"
    Config.scaler_dir = "models_TC11"
    Config.num_sensors = 11
    Config.experiment_name = "new_theoretical_TC11"
    Config.zscore_threshold = 1e9   # row-dropping OFF (it corrupted horizon timing + deleted ramp transients)
    Config.dropout_rate = 0.1       # dataset-specific dropout override
    Config.target_test_files = [
        "h6_flux88_abs20_surf0_781s - Sheet2.csv",
        "h6_flux88_abs92_surf0_648s - Sheet3.csv",
        "h6_flux88_abs0_surf1_790s - Sheet1.csv",
        "h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv",
    ]
    tc11_horizons = [60, 180, 300, 480, 600]
    try:
        horizon_sweep_fixed_seq(seq_len=20, horizons=tc11_horizons)
        print("\nSuccessfully completed TC11 Sweep!")
    except Exception as e:
        print(f"\nError during TC11 Sweep: {e}")
        
    # ---------------------------------------------
    # 2. Run TC10 Sweep (10 sensors)
    # ---------------------------------------------
    print("\n" + "="*80)
    print("STARTING SWEEP: TC10 (10 SENSORS)")
    print("="*80)
    
    Config.data_dir = "data/processed_H6"
    Config.scaler_dir = "models_TC10"
    Config.num_sensors = 10
    Config.experiment_name = "new_theoretical_TC10"
    # TC10 has steep temperature ramps — raise z-score threshold so startup rows
    # aren't mistakenly flagged as outliers and dropped from training.
    Config.zscore_threshold = 1e9   # row-dropping OFF (same reason as TC11)
    Config.dropout_rate = 0.2       # dataset-specific dropout override
    Config.target_test_files = [
        "h6_flux88_abs0_surf1_790s - Sheet1.csv",
        "h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv",
        "h6_flux88_abs20_surf0_longRun_612s - Sheet2.csv",
    ]
    
    tc10_horizons = [60, 180, 300, 480, 600, 900]
    try:
        horizon_sweep_fixed_seq(seq_len=20, horizons=tc10_horizons)
        print("\nSuccessfully completed TC10 Sweep!")
    except Exception as e:
        print(f"\nError during TC10 Sweep: {e}")

if __name__ == "__main__":
    main()
