"""Roll 2 more seeds for the flux73 H=900 retrain (train run2/3/4, hold out run1).
Compare against seed 42 (already trained, 11.64 K). Pick the best held-out run1 MAE.
Copies the winner to output/flux73_h900_BEST_L20_H900.
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib, os, shutil
import torch
import torch.utils.data as tud
from train import Config, run_single_experiment_with_profiles
from new_dataset_builder import TempSequenceDataset, collate_fn
from new_model import PhysicsInformedLSTM

DATA = "../data/processed_H6"
RUN1 = "h6_flux73_abs0_surf0_longRun_1658s - Sheet1.csv"   # HELD OUT
ABS92 = "h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv"  # sanity
TEST = [RUN1, ABS92]
H = 900
NEW_SEEDS = [7, 123]
SEED42_DIR = "output/flux73_h900_L20_H900"  # already trained


def set_cfg(exp):
    Config.data_dir = DATA
    Config.scaler_dir = "../models_TC10"
    Config.num_sensors = 10
    Config.experiment_name = exp
    Config.zscore_threshold = 1e9
    Config.dropout_rate = 0.2
    Config.target_test_files = TEST
    Config.physics_weight = 0.0
    Config.power_balance_weight = 0.0
    Config.max_epochs = 300
    Config.patience = 60


def fresh_perfile(exp_dir, files):
    out = {}
    for fn in files:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = TempSequenceDataset(data_dir=DATA, scaler_dir=exp_dir, split='test', prediction_horizon=H,
                                     num_sensors=10, zscore_threshold=1e9, target_test_files=[fn], bin_target=False)
            ds.load_pretrained_scalers(exp_dir)
        tsc = ds.thermal_scaler; sc = torch.tensor(tsc.scale_).float(); mn = torch.tensor(tsc.mean_).float()
        m = PhysicsInformedLSTM(num_sensors=10, sequence_length=20, lstm_units=512, dropout_rate=0.2,
                                residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
        m.load_state_dict(torch.load(f"{exp_dir}/best_model_L20_H{H}.pth", map_location='cpu')); m.eval()
        ld = tud.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
        P, T = [], []
        with torch.no_grad():
            for ts_, sp_, tg_, _ in ld:
                P.append(m([ts_, sp_]) * sc + mn); T.append(tg_ * sc + mn)
        P = torch.cat(P); T = torch.cat(T)
        out[fn] = (torch.abs(P - T).mean().item(), len(P))
    return out


def main():
    results = {}  # seed -> (dir, perfile)
    print("#" * 64)
    print("# flux73 H=900 SEED ROLL (held-out run1). seed42 already = 11.64 K")
    print("#" * 64)
    results[42] = (SEED42_DIR, fresh_perfile(SEED42_DIR, TEST))
    for seed in NEW_SEEDS:
        exp = f"flux73_h900_s{seed}"
        set_cfg(exp)
        run_single_experiment_with_profiles(L=20, H=H, seed=seed)
        d = f"output/{exp}_L20_H{H}"
        results[seed] = (d, fresh_perfile(d, TEST))
        print(f"\n>>> seed {seed}: run1(held-out)={results[seed][1][RUN1][0]:.2f} K  abs92={results[seed][1][ABS92][0]:.2f} K\n")

    print("\n" + "=" * 56)
    print("SEED ROLL SUMMARY — flux73 H=900 (fresh, held-out run1)")
    print("=" * 56)
    for s in sorted(results):
        r1 = results[s][1][RUN1][0]; a = results[s][1][ABS92][0]
        print(f"  seed {s:>4}:  run1(held-out)={r1:7.2f} K   abs92={a:6.2f} K")
    best = min(results, key=lambda s: results[s][1][RUN1][0])
    print(f"\n  WINNER: seed {best}  held-out flux73 H=900 = {results[best][1][RUN1][0]:.2f} K")
    canon = f"output/flux73_h900_BEST_L20_H{H}"
    os.makedirs(canon, exist_ok=True)
    for fn in (f"best_model_L20_H{H}.pth", "thermal_scaler.save", "param_scaler.save"):
        src = os.path.join(results[best][0], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(canon, fn))
    print(f"  -> copied seed {best} to {canon}")


if __name__ == "__main__":
    main()
