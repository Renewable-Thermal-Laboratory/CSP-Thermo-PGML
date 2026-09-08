#!/usr/bin/env python3
"""Leave-one-CONDITION-out test: the honest 'unseen regime' number.

Leave-one-run-out (train_final.py) still leaves other runs of the SAME condition
in the training set. Here we hold out every run of a whole flux/abs/surf combo,
train on the other conditions, and predict the held-out one -- true extrapolation
to a regime the model has never seen.

Some held-out conditions are interpolation (e.g. flux78, with flux73 and flux88
still in training) and some are extrapolation off the edge of an input's range
(e.g. abs92 is the ONLY high-absorber, so holding it out means the model never
saw abs=92 at all). We label each so the numbers are read correctly.

Uses the same depth-conditioned MLP and offset-from-TC1 target as train_final.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_final import CHANNELS, BASE, ZNORM, BAD_ALIGN
from mlp_model import MLPModel

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data" / "dataset.csv"

# flux 73/78/88 seen; abs 0/20/92 seen; surf 0/1 seen. A held-out condition is
# "extrapolation" if it removes the only instance of a value from an input axis.
FLUX_SEEN, ABS_SEEN = {73, 78, 88}, {0, 20, 92}


def long_format_grouped(df):
    Xb = df[BASE].values
    cond = df["cond_key"].values
    xs, ys, gs, cs = [], [], [], []
    for ch, z in CHANNELS.items():
        m = df[ch].notna().values
        if not m.any():
            continue
        xs.append(np.column_stack([Xb[m], np.full(m.sum(), ZNORM(z))]))
        ys.append(df.loc[m, ch].values - df.loc[m, "TC1"].values)
        gs.append(cond[m])
        cs.append(np.full(m.sum(), ch))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(cs)


def extrap_reason(df, cond):
    """Would holding out this condition remove the only copy of an input value?"""
    sub = df[df.cond_key == cond].iloc[0]
    rest = df[df.cond_key != cond]
    reasons = []
    if sub["abs"] not in set(rest["abs"]):
        reasons.append(f"only abs={sub['abs']}")
    if sub["flux"] not in set(rest["flux"]):
        reasons.append(f"only flux={sub['flux']}")
    if sub["surf"] not in set(rest["surf"]):
        reasons.append(f"only surf={sub['surf']}")
    return ", ".join(reasons) if reasons else "interpolation"


def main():
    df = pd.read_csv(DATA)
    df = df[~df["run_key"].isin(BAD_ALIGN)].reset_index(drop=True)
    df["cond_key"] = df["flux"].astype(str) + "_" + df["abs"].astype(str) + "_" + df["surf"].astype(str)
    X, y, g, chans = long_format_grouped(df)
    conds = sorted(df["cond_key"].unique(),
                   key=lambda c: (int(c.split("_")[1]), int(c.split("_")[0]), int(c.split("_")[2])))
    nruns = df.groupby("cond_key")["run_key"].nunique()
    print(f"leave-one-CONDITION-out  ({len(conds)} conditions, {df['run_key'].nunique()} runs)\n")

    order = list(CHANNELS)
    rows = []
    for c in conds:
        tr, te = g != c, g == c
        m = MLPModel().fit(X[tr], y[tr])
        err = np.abs(m.predict(X[te]) - y[te])
        ch_mae = {ch: err[chans[te] == ch].mean() for ch in np.unique(chans[te])}
        overall = err.mean()
        rows.append((c, nruns[c], overall, ch_mae, extrap_reason(df, c)))
        fl, ab, su = c.split("_")
        print(f"abs{ab:>2} flux{fl} surf{su}  (n={nruns[c]})  MAE={overall:5.2f} C   [{rows[-1][4]}]")

    print("\nper-channel MAE by held-out condition (deg C):")
    print(f"  {'condition':18s}" + "".join(f"{ch:>6}" for ch in order) + f"{'ALL':>7}")
    for c, n, ov, chm, _ in rows:
        fl, ab, su = c.split("_")
        label = f"abs{ab} f{fl} s{su}"
        print(f"  {label:18s}" + "".join(f"{chm.get(ch, np.nan):6.1f}" if ch in chm else f"{'-':>6}" for ch in order)
              + f"{ov:7.2f}")

    interp = [r[2] for r in rows if r[4] == "interpolation"]
    extrap = [(r[0], r[2], r[4]) for r in rows if r[4] != "interpolation"]
    print(f"\nSUMMARY:")
    print(f"  interpolation conditions (seen neighbours): mean MAE {np.mean(interp):.2f} C  (n={len(interp)})")
    print(f"  extrapolation conditions (off an input edge):")
    for c, ov, why in extrap:
        print(f"     {c:14s} {ov:5.2f} C   ({why})")
    print(f"\n  for reference, leave-one-RUN-out was 1.34 C")


if __name__ == "__main__":
    main()
