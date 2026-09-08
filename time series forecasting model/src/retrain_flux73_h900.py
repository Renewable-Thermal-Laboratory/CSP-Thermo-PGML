"""Retrain TC10 H=900 now that 4 long flux73_abs0_surf0 runs exist.

Train on run2/3/4 (+ all other data); HOLD OUT run1 as an honest flux73 H=900 test.
Also keep abs92_618 held out as a sanity check. Reports FRESH per-file MAE.
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib, os, shutil
import numpy as np, torch
import torch.utils.data as tud
from train import Config, run_single_experiment_with_profiles
from new_dataset_builder import TempSequenceDataset, collate_fn
from new_model import PhysicsInformedLSTM

DATA = "../data/processed_H6"
RUN1 = "h6_flux73_abs0_surf0_longRun_1658s - Sheet1.csv"   # HELD OUT (honest flux73 test)
RUN2 = "h6_flux73_abs0_surf0_longRun2_1309s - Sheet1.csv"  # train
RUN3 = "h6_flux73_abs0_surf0_longRun3_1376s - Sheet1.csv"  # train
RUN4 = "h6_flux73_abs0_surf0_longRun4_1224s - Sheet1.csv"  # train
ABS92 = "h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv"  # sanity
TEST = [RUN1, ABS92]
H = 900
SEED = 42


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
    Config.data_dir = DATA
    Config.scaler_dir = "../models_TC10"
    Config.num_sensors = 10
    Config.experiment_name = "flux73_h900"
    Config.zscore_threshold = 1e9
    Config.dropout_rate = 0.2
    Config.target_test_files = TEST
    Config.physics_weight = 0.0
    Config.power_balance_weight = 0.0
    Config.max_epochs = 300
    Config.patience = 60

    print("#" * 70)
    print(f"# RETRAIN H=900 with long flux73 data. Train: run2/3/4. HOLD OUT: run1.")
    print(f"# (old held-out flux73 H=900 = 118 K collapse)")
    print("#" * 70)
    run_single_experiment_with_profiles(L=20, H=H, seed=SEED)
    exp_dir = f"output/flux73_h900_L20_H{H}"

    # honest held-out + in-sample replicates + abs92 sanity
    res = fresh_perfile(exp_dir, [RUN1, RUN2, RUN3, RUN4, ABS92])
    print("\n" + "=" * 64)
    print("FRESH per-file MAE @ H=900 (after retrain with flux73 long data)")
    print("=" * 64)
    label = {RUN1: "run1  HELD-OUT (honest)", RUN2: "run2  (in train)", RUN3: "run3  (in train)",
             RUN4: "run4  (in train)", ABS92: "abs92_618 (sanity)"}
    for fn in [RUN1, RUN2, RUN3, RUN4, ABS92]:
        mae, n = res[fn]
        print(f"  {label[fn]:28} MAE={mae:8.2f} K   ({n} windows)")
    print(f"\n  >>> HONEST flux73 H=900 (run1, held-out) = {res[RUN1][0]:.2f} K   (was 118.44)")


if __name__ == "__main__":
    main()
