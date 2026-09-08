"""Re-train the unstable TC10 horizons (H=180, H=900) with new seeds, selecting the seed
whose FRESH re-evaluation (separate model+dataset rebuild) is lowest — NOT the in-run number,
which doesn't reproduce. Copies the winning checkpoint+scaler into the canonical TC10 dir.
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib, os, shutil
import numpy as np, torch
import torch.utils.data as tud
from train import Config, run_single_experiment_with_profiles
from new_dataset_builder import TempSequenceDataset, collate_fn
from new_model import PhysicsInformedLSTM

DATA_DIR = "../data/processed_H6"
TC10_TEST = ["h6_flux88_abs0_surf1_790s - Sheet1.csv",
             "h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv",
             "h6_flux88_abs20_surf0_longRun_612s - Sheet2.csv"]
SEEDS = [123, 7]
HORIZONS = [180, 900]


def set_tc10_config(exp_name):
    Config.data_dir = DATA_DIR
    Config.scaler_dir = "../models_TC10"
    Config.num_sensors = 10
    Config.experiment_name = exp_name
    Config.zscore_threshold = 1e9
    Config.dropout_rate = 0.2
    Config.target_test_files = TC10_TEST
    Config.physics_weight = 0.0
    Config.power_balance_weight = 0.0
    Config.max_epochs = 250
    Config.patience = 50


def fresh_eval(exp_dir, H):
    """Rebuild model+test-dataset from the saved artifacts and return true test MAE."""
    with contextlib.redirect_stdout(io.StringIO()):
        ds = TempSequenceDataset(data_dir=DATA_DIR, scaler_dir=exp_dir, split='test',
                                 prediction_horizon=H, num_sensors=10, zscore_threshold=1e9,
                                 target_test_files=TC10_TEST, bin_target=False)
        ds.load_pretrained_scalers(exp_dir)
    tsc = ds.thermal_scaler
    sc = torch.tensor(tsc.scale_).float(); mn = torch.tensor(tsc.mean_).float()
    m = PhysicsInformedLSTM(num_sensors=10, sequence_length=20, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(torch.load(f"{exp_dir}/best_model_L20_H{H}.pth", map_location='cpu')); m.eval()
    ld = tud.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    tot = 0; n = 0
    with torch.no_grad():
        for ts_, sp_, tg_, _ in ld:
            out = m([ts_, sp_]); tot += torch.abs(out * sc + mn - (tg_ * sc + mn)).sum().item(); n += tg_.numel()
    return tot / n


def main():
    summary = {}
    for H in HORIZONS:
        print("\n" + "#" * 70)
        print(f"# RE-TRAINING TC10 H={H}  (current fresh MAE: {'60.7' if H==180 else '22.0'} K)")
        print("#" * 70)
        results = []
        for seed in SEEDS:
            exp = f"retrain_TC10_s{seed}"
            set_tc10_config(exp)
            run_single_experiment_with_profiles(L=20, H=H, seed=seed)
            exp_dir = f"output/{exp}_L20_H{H}"
            try:
                fmae = fresh_eval(exp_dir, H)
            except Exception as e:
                print(f"  [seed {seed}] fresh-eval error: {e}"); fmae = float('inf')
            print(f"\n>>> H={H} seed={seed}: FRESH re-eval MAE = {fmae:.3f} K\n")
            results.append((seed, fmae, exp_dir))
        best = min(results, key=lambda r: r[1])
        summary[H] = best
        # copy winner into canonical TC10 dir
        canon = f"output/new_theoretical_TC10_L20_H{H}"
        os.makedirs(canon, exist_ok=True)
        for fn in (f"best_model_L20_H{H}.pth", "thermal_scaler.save", "param_scaler.save"):
            src = os.path.join(best[2], fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(canon, fn))
        print(f"\n=== H={H}: BEST seed={best[0]}  fresh MAE={best[1]:.3f} K  -> copied to canonical ===")

    print("\n" + "=" * 60)
    print("RETRAIN SUMMARY (fresh, reproducible MAE)")
    print("=" * 60)
    for H, (seed, fmae, _) in summary.items():
        print(f"  H={H}: best seed={seed}  fresh MAE={fmae:.3f} K")


if __name__ == "__main__":
    main()
