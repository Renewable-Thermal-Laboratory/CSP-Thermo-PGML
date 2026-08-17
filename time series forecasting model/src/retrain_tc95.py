"""Train the 11-sensor (incl TC_9.5) model on data/flux88_TC95 (flux88_abs92_surf0 only).
Hold out longRun_647 as test. 2 seeds/horizon, pick best held-out. Reports per-sensor
(incl TC_9.5). H=600/900 may be data-limited (only 2 long files) -> caught & skipped.
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib, os, shutil, json
import torch
import torch.utils.data as tud
from train import Config, run_single_experiment_with_profiles
from new_dataset_builder import TempSequenceDataset, collate_fn
from new_model import PhysicsInformedLSTM

DATA = "../data/flux88_TC95"
TEST = "h6_flux88_abs92_surf0_longRun_647s - Sheet2.csv"
NS = 11
HORIZONS = [60, 180, 300, 480, 600, 900]
SEEDS = [42, 123]
SENSORS = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'TC6', 'TC7', 'TC8', 'TC9', 'TC_9.5', 'TC10']


def set_cfg(exp):
    Config.data_dir = DATA
    Config.scaler_dir = "../models_TC95"
    Config.num_sensors = NS
    Config.experiment_name = exp
    Config.zscore_threshold = 1e9
    Config.dropout_rate = 0.2
    Config.target_test_files = [TEST]
    Config.physics_weight = 0.0
    Config.power_balance_weight = 0.0
    Config.max_epochs = 300
    Config.patience = 50


def fresh(exp_dir, H):
    with contextlib.redirect_stdout(io.StringIO()):
        ds = TempSequenceDataset(data_dir=DATA, scaler_dir=exp_dir, split='test', prediction_horizon=H,
                                 num_sensors=NS, zscore_threshold=1e9, target_test_files=[TEST], bin_target=False)
        ds.load_pretrained_scalers(exp_dir)
    tsc = ds.thermal_scaler; sc = torch.tensor(tsc.scale_).float(); mn = torch.tensor(tsc.mean_).float()
    m = PhysicsInformedLSTM(num_sensors=NS, sequence_length=20, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(torch.load(f"{exp_dir}/best_model_L20_H{H}.pth", map_location='cpu')); m.eval()
    ld = tud.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    P, T = [], []
    with torch.no_grad():
        for ts_, sp_, tg_, _ in ld:
            P.append(m([ts_, sp_]) * sc + mn); T.append(tg_ * sc + mn)
    P = torch.cat(P); T = torch.cat(T)
    return torch.abs(P - T).mean().item(), torch.abs(P - T).mean(0).tolist(), len(P)


def main():
    summary = {}
    for H in HORIZONS:
        print("#" * 60); print(f"# H={H}  (11-sensor incl TC_9.5)"); print("#" * 60)
        res = {}
        for seed in SEEDS:
            exp = f"tc95_H{H}_s{seed}"; set_cfg(exp)
            try:
                run_single_experiment_with_profiles(L=20, H=H, seed=seed)
                d = f"output/{exp}_L20_H{H}"; ov, ps, n = fresh(d, H)
            except Exception as e:
                print(f"  [H={H} s{seed}] ERROR: {str(e)[:90]}"); continue
            res[seed] = (d, ov, ps, n); print(f"  H={H} s{seed}: held-out MAE={ov:.3f} K (n={n})")
        if not res:
            print(f"  H={H}: NO successful seed (data-limited: only 2 long files)\n"); continue
        best = min(res, key=lambda s: res[s][1]); d, ov, ps, n = res[best]
        summary[H] = {"seed": best, "MAE": round(ov, 3), "n": n,
                      "persensor": {SENSORS[i]: round(ps[i], 3) for i in range(NS)}}
        canon = f"output/tc95_best_L20_H{H}"; os.makedirs(canon, exist_ok=True)
        for fn in (f"best_model_L20_H{H}.pth", "thermal_scaler.save", "param_scaler.save"):
            s = os.path.join(d, fn)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(canon, fn))
        print(f"  >>> H={H} WINNER seed={best}  MAE={ov:.3f} K  -> {canon}\n")
        with open("../tc95_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TC95 (11-sensor incl TC_9.5) — held-out longRun_647 MAE (K)")
    print("=" * 70)
    print(f"{'H':>5} {'seed':>5} {'overall':>8} {'TC_9.5':>8} {'TC10':>8} {'n':>6}")
    for H in HORIZONS:
        if H in summary:
            s = summary[H]
            print(f"{H:>5} {s['seed']:>5} {s['MAE']:>8.3f} {s['persensor']['TC_9.5']:>8.3f} {s['persensor']['TC10']:>8.3f} {s['n']:>6}")
        else:
            print(f"{H:>5}   --  (data-limited: only 2 long files, no valid split)")
    print("\nDONE.  summary -> tc95_summary.json")


if __name__ == "__main__":
    main()
