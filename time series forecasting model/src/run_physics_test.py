"""Test: TC10 H=480 with PHYSICS GUIDANCE ON + proper (longer) training + fixed seed.

H=480 is one of the unstable horizons (logged 1.80, re-evaluated 9.72). This checks whether
physics guidance + full training produces a STABLE, reproducible model. Writes to a separate
experiment name so it doesn't touch the real results.
"""
import warnings; warnings.filterwarnings('ignore')
from train import Config, run_single_experiment_with_profiles

Config.data_dir = "../data/processed_H6"
Config.scaler_dir = "../models_TC10"
Config.num_sensors = 10
Config.experiment_name = "physics_test_TC10"
Config.zscore_threshold = 1e9
Config.dropout_rate = 0.2
Config.target_test_files = [
    "h6_flux88_abs0_surf1_790s - Sheet1.csv",
    "h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv",
    "h6_flux88_abs20_surf0_longRun_612s - Sheet2.csv",
]

# --- PHYSICS GUIDANCE ON (energy-conservation regularizer) ---
Config.physics_weight = 0.1
Config.power_balance_weight = 0.05

# --- proper training for stability (no aggressive early stop) ---
Config.max_epochs = 250
Config.patience = 50

res = run_single_experiment_with_profiles(L=20, H=480, seed=42)
print("\n" + "=" * 60)
print("PHYSICS-GUIDED H=480 TEST RESULT")
print("=" * 60)
print(f"  status     : {res.get('status')}")
print(f"  best_epoch : {res.get('best_epoch')}")
print(f"  test MAE   : {res.get('test_mae_unscaled', float('nan')):.3f} K   (old unstable: 9.72 | want: <2)")
print(f"  test R^2   : {res.get('test_r2_overall_unscaled', float('nan')):.4f}")
print(f"  repro_ok   : {res.get('repro_ok')}   (in-run vs reloaded match)")
print("=" * 60)
