"""Clean, single-process re-run of the TC10 sweep ONLY (TC11 is already trustworthy).

Uses the fixed train.py, which now (a) saves each experiment's scalers next to its
checkpoint so post-hoc eval uses the matched scaler, and (b) runs a reproducibility
check after every experiment — printing [REPRO-CHECK OK] or [REPRO-CHECK FAILED].

Run from project root:  python3 -u run_tc10.py
"""
import os
import sys

sys.path.append(os.path.abspath("src"))
from src.train import Config, horizon_sweep_fixed_seq


def main():
    print("\n" + "=" * 80)
    print("STARTING SWEEP: TC10 (10 SENSORS) — CLEAN RE-RUN with per-experiment scalers + repro check")
    print("=" * 80)

    Config.data_dir = "data/processed_H6"
    Config.scaler_dir = "models_TC10"
    Config.num_sensors = 10
    Config.experiment_name = "new_theoretical_TC10"
    Config.zscore_threshold = 1e9   # row-dropping OFF (preserve exact horizon timing)
    Config.dropout_rate = 0.2
    Config.target_test_files = [
        "h6_flux88_abs0_surf1_790s - Sheet1.csv",
        "h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv",
        "h6_flux88_abs20_surf0_longRun_612s - Sheet2.csv",
    ]
    # PROPER / STABLE training — revert the aggressive early-stopping speed tweaks that
    # under-converged the model into a fragile, non-reproducible state. Physics stays OFF
    # (all weights 0) so epochs run at normal speed.
    Config.max_epochs = 400
    Config.patience = 80

    tc10_horizons = [60, 180, 300, 480, 600, 900]
    try:
        horizon_sweep_fixed_seq(seq_len=20, horizons=tc10_horizons)
        print("\nSuccessfully completed TC10 Sweep!")
    except Exception as e:
        print(f"\nError during TC10 Sweep: {e}")


if __name__ == "__main__":
    main()
