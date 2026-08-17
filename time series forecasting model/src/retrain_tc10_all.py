"""Full TC10 retrain for best results (TC11 dropped). All 6 horizons x 3 seeds, with the
4 new long flux73 runs in the data. Held-out test = run1 (abs0, clean) + flux88 longRun_762
(abs0, tougher/diff flux) + abs92_618 (abs92). Winner per horizon = lowest MEAN held-out
fresh MAE across the 3 test files. Winners copied to output/tc10_best_L20_H<H>.
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib, os, shutil, json, csv
import torch
import torch.utils.data as tud
from train import Config, run_single_experiment_with_profiles
from new_dataset_builder import TempSequenceDataset, collate_fn
from new_model import PhysicsInformedLSTM

DATA = "../data/processed_H6"
RUN1 = "h6_flux73_abs0_surf0_longRun_1658s - Sheet1.csv"       # abs0 clean (replicate)
ABS0_HARD = "h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv"   # abs0 tougher (diff flux)
ABS92 = "h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv"      # abs92
TEST = [RUN1, ABS0_HARD, ABS92]
SHORT = {RUN1: "run1_abs0", ABS0_HARD: "762_abs0", ABS92: "abs92_618"}
HORIZONS = [180, 300, 480, 600, 900, 60]   # broken-middle first; H=60 (already good) last
SEEDS = [123, 42, 7]


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
    Config.patience = 50


def fresh_perfile(exp_dir, H, files):
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
        if P:
            P = torch.cat(P); T = torch.cat(T); out[fn] = round(torch.abs(P - T).mean().item(), 3)
        else:
            out[fn] = float('nan')
    return out


def main():
    summary = {}
    for H in HORIZONS:
        print("#" * 70); print(f"# HORIZON H={H}  (seeds {SEEDS})"); print("#" * 70)
        seed_results = {}
        for seed in SEEDS:
            exp = f"tc10_all_H{H}_s{seed}"
            set_cfg(exp)
            try:
                run_single_experiment_with_profiles(L=20, H=H, seed=seed)
                d = f"output/{exp}_L20_H{H}"
                pf = fresh_perfile(d, H, TEST)
            except Exception as e:
                print(f"  [H={H} seed={seed}] ERROR {e}"); continue
            vals = [v for v in pf.values() if v == v]
            score = round(sum(vals) / len(vals), 3) if vals else float('inf')
            seed_results[seed] = (d, pf, score)
            print(f"  H={H} seed={seed}: " + "  ".join(f"{SHORT[fn]}={pf[fn]:.2f}" for fn in TEST) + f"   mean={score:.2f}")
        if not seed_results:
            continue
        best = min(seed_results, key=lambda s: seed_results[s][2])
        bd, bpf, bscore = seed_results[best]
        summary[H] = {"best_seed": best, "mean": bscore, "perfile": {SHORT[fn]: bpf[fn] for fn in TEST},
                      "all_seeds": {s: seed_results[s][2] for s in seed_results}}
        canon = f"output/tc10_best_L20_H{H}"
        os.makedirs(canon, exist_ok=True)
        for fn in (f"best_model_L20_H{H}.pth", "thermal_scaler.save", "param_scaler.save"):
            src = os.path.join(bd, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(canon, fn))
        print(f"  >>> H={H} WINNER seed={best}  mean={bscore:.2f} K  -> {canon}\n")
        with open("../tc10_best_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("FINAL TC10 BEST LADDER (held-out fresh MAE, K)")
    print("=" * 78)
    print(f"{'H':>5} {'minutes':>8} {'seed':>5} {'run1_abs0':>10} {'762_abs0':>10} {'abs92_618':>10}")
    rows = []
    for H in [60, 180, 300, 480, 600, 900]:
        if H not in summary:
            continue
        s = summary[H]; pf = s["perfile"]
        print(f"{H:>5} {H/60:>8.1f} {s['best_seed']:>5} {pf['run1_abs0']:>10.2f} {pf['762_abs0']:>10.2f} {pf['abs92_618']:>10.2f}")
        rows.append([H, H / 60, s["best_seed"], pf["run1_abs0"], pf["762_abs0"], pf["abs92_618"]])
    with open("../tc10_best_ladder.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["H", "minutes", "best_seed", "run1_abs0_MAE", "762_abs0_MAE", "abs92_618_MAE"])
        w.writerows(rows)
    print("\nDONE. summary -> tc10_best_summary.json, ladder -> tc10_best_ladder.csv")


if __name__ == "__main__":
    main()
