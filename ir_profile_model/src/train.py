#!/usr/bin/env python3
"""Baseline static IR -> vertical-profile model.

Predicts the rake profile TC2..TC10 from a SINGLE snapshot (no time history):
    inputs  = IR surface features + TC1 (bottom) + flux, abs, surf
    targets = TC2 .. TC10

Evaluation is leave-one-run-OUT (every second of a held-out run is unseen), and
optionally leave-one-condition-out (a whole flux/abs/surf combo held out) -- the
harder, industrial "unseen regime" test. Reports per-channel MAE and, as an
ablation, the same model WITHOUT the IR features, to quantify what the camera buys.

This is the baseline number to beat; a depth-conditioned MLP / physics loss come next.

Usage:
    python train.py                     # leave-one-run-out
    python train.py --cv condition      # leave-one-condition-out
    python train.py --drop-bad          # exclude runs with known bad IR/TC alignment
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data" / "dataset.csv"

IR_FEATS = ["ir_center", "ir_bulk", "ir_p95", "ir_p99", "ir_p25", "ir_std", "ir_grad"]
COND = ["flux", "abs", "surf"]
TARGETS = [f"TC{i}" for i in range(2, 11)]
BAD_ALIGN = ["h6_abs0_flux73_surf0_run4"]     # corr(ir_bulk,TC10) = -0.91


def evaluate(df, feats, group_col):
    """Leave-one-group-out RF; returns per-channel MAE averaged over groups."""
    groups = df[group_col].unique()
    per = []
    X = df[feats].values
    Y = df[TARGETS].values
    g = df[group_col].values
    for gv in groups:
        tr, te = g != gv, g == gv
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
        rf.fit(X[tr], Y[tr])
        per.append(np.abs(rf.predict(X[te]) - Y[te]).mean(0))
    return np.array(per), groups           # (n_groups, n_targets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", choices=["run", "condition"], default="run")
    ap.add_argument("--drop-bad", action="store_true", help="exclude known bad-alignment runs")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    if args.drop_bad:
        df = df[~df["run_key"].isin(BAD_ALIGN)].reset_index(drop=True)
    df["cond_key"] = df["flux"].astype(str) + "_" + df["abs"].astype(str) + "_" + df["surf"].astype(str)
    group_col = "run_key" if args.cv == "run" else "cond_key"
    print(f"rows={len(df)}  runs={df['run_key'].nunique()}  CV=leave-one-{args.cv}-out "
          f"({df[group_col].nunique()} folds)\n")

    full = IR_FEATS + ["TC1"] + COND
    noir = ["TC1"] + COND
    mae_full, groups = evaluate(df, full, group_col)
    mae_noir, _ = evaluate(df, noir, group_col)

    print("per-channel MAE (deg C):    interior <------------------------> surface")
    print(f"  {'':14s}" + "".join(f"{t:>7}" for t in TARGETS) + f"{'ALL':>8}")
    print(f"  IR+TC+cond   " + "".join(f"{v:7.2f}" for v in mae_full.mean(0)) + f"{mae_full.mean():8.2f}")
    print(f"  TC+cond only " + "".join(f"{v:7.2f}" for v in mae_noir.mean(0)) + f"{mae_noir.mean():8.2f}")
    print(f"  IR gain      " + "".join(f"{v:7.2f}" for v in (mae_noir.mean(0) - mae_full.mean(0)))
          + f"{mae_noir.mean()-mae_full.mean():8.2f}")

    # per-abs breakdown of the full model
    absr = np.array([df[df[group_col] == gv]["abs"].iloc[0] for gv in groups])
    print("\n  full-model MAE by absorber level:")
    for a in [0, 20, 92]:
        sel = absr == a
        if sel.any():
            print(f"    abs{a:<2d} (n={sel.sum():2d} folds): {mae_full[sel].mean():.2f} C")

    # worst folds (surface for bad alignment/detection)
    allmae = mae_full.mean(1)
    order = np.argsort(allmae)[::-1][:5]
    print("\n  worst folds (mean MAE):")
    for i in order:
        print(f"    {str(groups[i]).replace('h6_',''):40s} {allmae[i]:.2f} C")


if __name__ == "__main__":
    main()
