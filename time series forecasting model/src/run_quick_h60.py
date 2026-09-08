"""Quick confirmation run: TC11, H=60 only, to validate the residual-forecasting change.

Reduced epoch budget for a fast-but-trustworthy read (residual learning converges fast).
Run from the src/ directory:  python3 run_quick_h60.py
"""
from train import Config, run_single_experiment_with_profiles

# --- TC11 dataset config (mirrors run_all.py) ---
Config.data_dir = "../data/output_with_TC11"
Config.scaler_dir = "../models_TC11"
Config.num_sensors = 11
Config.experiment_name = "residual_test_TC11"
Config.zscore_threshold = 1e9   # effectively OFF: dropping rows corrupts horizon timing
                                # (H steps != H seconds) and deletes legit heated-end transients.
                                # NaN rows are still removed by dropna() in _process_file.
Config.dropout_rate = 0.1
Config.target_test_files = [
    "h6_flux88_abs20_surf0_781s - Sheet2.csv",
    "h6_flux88_abs92_surf0_648s - Sheet3.csv",
    "h6_flux88_abs0_surf1_790s - Sheet1.csv",
    "h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv",
]

# Faster than the 400/100 default; residual model + smooth cosine converge quickly.
Config.max_epochs = 200
Config.patience = 60

res = run_single_experiment_with_profiles(L=20, H=60, seed=42)

print("\n" + "=" * 60)
print("QUICK H=60 RESULT (residual forecasting)")
print("=" * 60)
print(f"  status   : {res.get('status')}")
print(f"  best_epoch: {res.get('best_epoch')}")
print(f"  TEST MAE : {res.get('test_mae_unscaled'):.3f} K   (old run: 3.28 | persistence: 1.85 | target: <1.0)")
print(f"  TEST RMSE: {res.get('test_rmse_unscaled'):.3f} K")
print(f"  TEST R^2 : {res.get('test_r2_overall_unscaled'):.4f}   (old run: 0.913 | target: >0.97)")
psm = res.get('test_per_sensor_metrics', [])
if psm:
    print("  per-sensor MAE (K):", [round(m['mae'], 2) for m in psm])
print("=" * 60)
