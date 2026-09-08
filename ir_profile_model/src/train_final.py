#!/usr/bin/env python3
"""Final static profile model: one IR snapshot + TC1 + conditions -> TC2..TC10 (+TC9.5).

Architecture: DEPTH-CONDITIONED regression. Instead of nine separate output heads,
the rake position z is an input and the model predicts T(z):

    [ir features..., TC1, flux, abs, surf, z]  ->  T at position z

Why: one model shares strength across depths, TC9.5 is naturally included as
z = 9.5 on the runs that have it (abs92), and the profile can be evaluated at ANY
z, giving a continuous curve. Positions use the TC index as the depth coordinate
(TC1 bottom ... TC10 surface), normalized to [0, 1].

The target is predicted as an OFFSET from TC1 (the model learns the profile shape
relative to the known bottom temperature), which removes the absolute-temperature
degree of freedom and helps generalization.

Two learners are evaluated leave-one-run-out and the better is kept:
    * HistGradientBoosting on the depth-conditioned long format
    * MLP (standardized features) on the same format

Runs with known-bad IR/TC alignment are excluded. The chosen learner is refit on
ALL data and saved to models/profile_model.joblib for predict.py.

Usage:
    python train_final.py            # evaluate (LORO) + fit + save final model
    python train_final.py --fast     # skip the MLP comparison
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mlp_model import MLPModel

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data" / "dataset.csv"
MODELS = PROJ / "models"

IR_FEATS = ["ir_center", "ir_bulk", "ir_p95", "ir_p99", "ir_p25", "ir_std", "ir_grad"]
COND = ["flux", "abs", "surf"]
BASE = IR_FEATS + ["TC1"] + COND
BAD_ALIGN = ["h6_abs0_flux73_surf0_run4"]      # corr(ir_bulk, TC10) = -0.91

CHANNELS = {"TC2": 2, "TC3": 3, "TC4": 4, "TC5": 5, "TC6": 6, "TC7": 7,
            "TC8": 8, "TC9": 9, "TC9.5": 9.5, "TC10": 10}
ZNORM = lambda z: (np.asarray(z, dtype=float) - 1.0) / 9.0     # TC1..TC10 -> 0..1


def long_format(df):
    """Wide rows -> (X, y, run, channel) long format with z as a feature.
    y is the offset from TC1."""
    Xb = df[BASE].values
    run = df["run_key"].values
    xs, ys, rs, cs = [], [], [], []
    for ch, z in CHANNELS.items():
        mask = df[ch].notna().values
        if not mask.any():
            continue
        xz = np.column_stack([Xb[mask], np.full(mask.sum(), ZNORM(z))])
        xs.append(xz)
        ys.append(df.loc[mask, ch].values - df.loc[mask, "TC1"].values)
        rs.append(run[mask])
        cs.append(np.full(mask.sum(), ch))
    return (np.vstack(xs), np.concatenate(ys), np.concatenate(rs), np.concatenate(cs))


def make_hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                         max_depth=None, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)


def loro_eval(name, make_model, X, y, runs, chans):
    uruns = np.unique(runs)
    rows = []
    for r in uruns:
        tr, te = runs != r, runs == r
        m = make_model().fit(X[tr], y[tr])
        err = np.abs(m.predict(X[te]) - y[te])
        for ch in np.unique(chans[te]):
            rows.append((r, ch, err[chans[te] == ch].mean()))
    per = pd.DataFrame(rows, columns=["run", "chan", "mae"])
    print(f"\n[{name}] leave-one-run-out MAE (offset-from-TC1, deg C):")
    order = list(CHANNELS)
    ch_mae = per.groupby("chan")["mae"].mean().reindex(order)
    print("   " + "".join(f"{c:>7}" for c in order))
    print("   " + "".join(f"{v:7.2f}" for v in ch_mae.values) + f"   ALL={per['mae'].mean():.2f}")
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip the MLP comparison")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    df = df[~df["run_key"].isin(BAD_ALIGN)].reset_index(drop=True)
    X, y, runs, chans = long_format(df)
    print(f"long-format samples={len(X)}  (runs={df['run_key'].nunique()}, "
          f"channels incl TC9.5 where present)")

    per_hgb = loro_eval("HistGB", make_hgb, X, y, runs, chans)
    best_name, best_make, best_per = "hgb", make_hgb, per_hgb
    if not args.fast:
        per_mlp = loro_eval("MLP", MLPModel, X, y, runs, chans)
        if per_mlp["mae"].mean() < per_hgb["mae"].mean():
            best_name, best_make, best_per = "mlp", MLPModel, per_mlp
    print(f"\nselected learner: {best_name}")

    # abs-level breakdown for the selected learner
    absmap = df.groupby("run_key")["abs"].first()
    best_per["abs"] = best_per["run"].map(absmap)
    print("MAE by absorber level:")
    for a, g in best_per.groupby("abs"):
        print(f"  abs{a:<3d}: {g['mae'].mean():.2f} C   (runs={g['run'].nunique()})")

    # final fit on ALL data
    final = best_make().fit(X, y)
    MODELS.mkdir(exist_ok=True)
    payload = {
        "model": final, "learner": best_name,
        "base_features": BASE, "ir_features": IR_FEATS,
        "channels": CHANNELS, "znorm": "z01 = (z - 1) / 9",
        "target": "offset from TC1 (add TC1 back at predict time)",
        "excluded_runs": BAD_ALIGN,
        "loro_mae_overall": float(best_per["mae"].mean()),
    }
    out = MODELS / "profile_model.joblib"
    joblib.dump(payload, out)
    print(f"\nfinal model (trained on all {len(X)} samples) -> {out}")


if __name__ == "__main__":
    main()
