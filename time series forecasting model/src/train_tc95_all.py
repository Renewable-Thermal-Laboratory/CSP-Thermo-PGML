"""11-sensor (incl TC_9.5) trainer for data/flux88_TC95 — ALL horizons 60..900.

Data is one condition (flux88_abs92_surf0) with only 2 long files, so a file-level val
split fails at H>=600. Instead: TEST = longRun_647 (held out, whole file); TRAIN/VAL =
window-level 85/15 split over every other file's windows. Scaler fit on all non-test
files. Reuses PhysicsInformedLSTM (residual/persistence) + project hyperparams
(AdamW wd=1e-4, Huber/SmoothL1, CosineAnnealingLR, grad-clip 1.0, early stop on val).
"""
import warnings; warnings.filterwarnings('ignore')
import os, glob, re, json, copy
import numpy as np, pandas as pd, torch, joblib
from sklearn.preprocessing import StandardScaler
import sys; sys.path.insert(0, '.')
from new_model import PhysicsInformedLSTM

DATA = '../data/flux88_TC95'
TEST = 'h6_flux88_abs92_surf0_longRun_647s - Sheet2.csv'
NS, SEQ, TM, TSC = 11, 20, 300.0, 300.0
PARAM = ['h', 'flux', 'abs', 'surf']
SENSORS = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'TC6', 'TC7', 'TC8', 'TC9', 'TC_9.5', 'TC10']
HORIZONS = [60, 180, 300, 480, 600, 900]
SEEDS = [42, 123]
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def tc_cols(df):
    return sorted([c for c in df.columns if 'TC' in c.upper()], key=lambda c: int(re.findall(r'\d+', c)[0]))[:NS]


def load(f):
    return pd.read_csv(f).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)


def fit_scalers(files):
    tcd, pcd = [], []
    for f in files:
        df = load(f); tcd.append(df[tc_cols(df)].values); pcd.append(df[PARAM].iloc[[0]].values)
    return StandardScaler().fit(np.vstack(tcd)), StandardScaler().fit(np.vstack(pcd))


def windows(files, H, tsc, psc):
    sc, mn = tsc.scale_, tsc.mean_; X, S, Y = [], [], []
    for f in files:
        df = load(f); cols = tc_cols(df); n = len(df)
        if n < SEQ + H + 1:
            continue
        tcv = df[cols].values.astype(float); tv = df['Time'].values.astype(float)
        sp = psc.transform([[float(df[c].iloc[0]) for c in PARAM]])[0]
        for off in range(n - SEQ - H + 1):
            X.append(np.hstack([((tv[off:off + SEQ] - TM) / TSC).reshape(-1, 1), (tcv[off:off + SEQ] - mn) / sc]))
            Y.append((tcv[off + SEQ - 1 + H] - mn) / sc); S.append(sp)
    if not X:
        return None
    return (torch.tensor(np.array(X)).float(), torch.tensor(np.array(S)).float(), torch.tensor(np.array(Y)).float())


def eval_mae(model, X, S, Y, tsc):
    model.eval(); sc = torch.tensor(tsc.scale_).float(); mn = torch.tensor(tsc.mean_).float()
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            P.append(model([X[i:i + 256].to(DEV), S[i:i + 256].to(DEV)]).cpu())
    P = torch.cat(P) * sc + mn; Yu = Y * sc + mn
    return torch.abs(P - Yu).mean().item(), torch.abs(P - Yu).mean(0).tolist()


def train_one(H, seed, tsc, psc, tr, te):
    torch.manual_seed(seed); np.random.seed(seed)
    X, S, Y = tr
    idx = np.random.RandomState(seed).permutation(len(X)); nval = max(1, int(0.15 * len(X)))
    vi, ti = idx[:nval], idx[nval:]
    Xt, St, Yt = X[ti], S[ti], Y[ti]; Xv, Sv, Yv = X[vi], S[vi], Y[vi]
    m = PhysicsInformedLSTM(num_sensors=NS, sequence_length=SEQ, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-6)
    lossfn = torch.nn.SmoothL1Loss(); best, best_state, bad = 1e9, None, 0
    for epoch in range(300):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]; opt.zero_grad()
            loss = lossfn(m([Xt[b].to(DEV), St[b].to(DEV)]), Yt[b].to(DEV))
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        vmae, _ = eval_mae(m, Xv, Sv, Yv, tsc)
        if vmae < best - 1e-4:
            best, best_state, bad = vmae, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
        if bad >= 50:
            break
    m.load_state_dict(best_state)
    tmae, tps = eval_mae(m, te[0], te[1], te[2], tsc)
    return tmae, tps, best, m


def main():
    pool = [f for f in sorted(glob.glob(DATA + '/*.csv')) if os.path.basename(f) != TEST]
    test_f = os.path.join(DATA, TEST)
    print(f"device {DEV}  | train-pool {len(pool)} files, test = {TEST}")
    summary = {}
    for H in HORIZONS:
        print("#" * 60); print(f"# H={H}"); print("#" * 60)
        tsc, psc = fit_scalers(pool)
        tr = windows(pool, H, tsc, psc); te = windows([test_f], H, tsc, psc)
        if tr is None or te is None:
            print(f"  H={H}: no windows, skip"); continue
        print(f"  train-pool windows={len(tr[0])}  test windows={len(te[0])}")
        res = {}
        for seed in SEEDS:
            tmae, tps, vmae, m = train_one(H, seed, tsc, psc, tr, te)
            res[seed] = (tmae, tps, m)
            print(f"  H={H} s{seed}: test MAE={tmae:.3f} K (val {vmae:.3f})")
        best = min(res, key=lambda s: res[s][0]); tmae, tps, m = res[best]
        summary[H] = {"seed": best, "MAE": round(tmae, 3),
                      "persensor": {SENSORS[i]: round(tps[i], 3) for i in range(NS)}}
        d = f"output/tc95_best_L20_H{H}"; os.makedirs(d, exist_ok=True)
        torch.save(m.state_dict(), f"{d}/best_model_L20_H{H}.pth")
        joblib.dump(tsc, f"{d}/thermal_scaler.save"); joblib.dump(psc, f"{d}/param_scaler.save")
        print(f"  >>> H={H} WINNER seed={best} test MAE={tmae:.3f} K -> {d}\n")
        with open("../tc95_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TC95 (11-sensor incl TC_9.5) — held-out longRun_647 MAE (K)")
    print("=" * 70)
    print(f"{'H':>5} {'seed':>5} {'overall':>8} {'TC_9.5':>8} {'TC10':>8}")
    for H in HORIZONS:
        if H in summary:
            s = summary[H]
            print(f"{H:>5} {s['seed']:>5} {s['MAE']:>8.3f} {s['persensor']['TC_9.5']:>8.3f} {s['persensor']['TC10']:>8.3f}")
    print("\nDONE.  summary -> tc95_summary.json")


if __name__ == "__main__":
    main()
