#!/usr/bin/env python3
"""Join IR surface features with the thermocouple rake into one training table.

Both the IR .h5 runs and the processed TC runs are already trimmed to the same
run window (their per-run lengths match), so they align second-for-second at
index 0. For each paired run we merge:

    inputs   : ir_center ir_bulk ir_p95 ir_p99 ir_p25 ir_std ir_grad ir_area,
               TC1 (bottom), flux, abs, surf
    targets  : TC2 .. TC10   (the interior + surface profile to predict)

and write one long CSV (data/dataset.csv) with a run_key column for grouped CV.

As a sanity check it reports, per run, the correlation between the IR surface
feature (ir_bulk) and the surface thermocouple (TC10): if the join is aligned
and the feature is meaningful, these should be strongly positive. Runs whose IR
and TC lengths differ by more than TOL seconds are flagged (possible mis-trim).

Usage:
    python build_dataset.py
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
IRF_DIR = PROJ / "data" / "ir_features"
TC_DIR = PROJ / "data" / "processed_TC"
OUT = PROJ / "data" / "dataset.csv"

IR_FEATS = ["ir_center", "ir_bulk", "ir_p95", "ir_p99", "ir_p25", "ir_std", "ir_grad", "ir_area"]
TARGETS = [f"TC{i}" for i in range(2, 11)]           # TC2..TC10
OPTIONAL = ["TC9.5"]                                  # abs92 runs only; NaN elsewhere
LEN_TOL = 15                                          # flag runs whose lengths differ by more


def run_key(name):
    b = os.path.splitext(os.path.basename(name))[0].replace("_ir", "")
    return re.sub(r"_\d+s_", "_", b)


def conds(name):
    b = os.path.basename(name)
    return (int(re.search(r"flux(\d+)", b).group(1)),
            int(re.search(r"abs(\d+)", b).group(1)),
            int(re.search(r"surf(\d+)", b).group(1)))


def main():
    irf = {run_key(f): f for f in glob.glob(str(IRF_DIR / "*.csv")) if not os.path.basename(f).startswith("_")}
    tcf = {run_key(f): f for f in glob.glob(str(TC_DIR / "*.csv"))
           if os.path.basename(f) not in ("process_summary.csv", "cooldown_cuts.txt")}
    keys = sorted(set(irf) & set(tcf))
    print(f"{len(keys)} paired runs (IR {len(irf)}, TC {len(tcf)})")

    parts, report = [], []
    for k in keys:
        ir = pd.read_csv(irf[k])
        tc = pd.read_csv(tcf[k])
        n = min(len(ir), len(tc))
        flux, ab, su = conds(irf[k])
        d = pd.DataFrame({"run_key": k, "flux": flux, "abs": ab, "surf": su, "t": np.arange(n)})
        for c in IR_FEATS:
            d[c] = ir[c].values[:n]
        for c in ["TC1"] + TARGETS:
            d[c] = tc[c].values[:n]
        for c in OPTIONAL:
            d[c] = tc[c].values[:n] if c in tc.columns else np.nan
        parts.append(d)
        r = np.corrcoef(d["ir_bulk"], d["TC10"])[0, 1]
        flag = "  <-- length mismatch" if abs(len(ir) - len(tc)) > LEN_TOL else ""
        report.append((k, len(ir), len(tc), n, r, flag))

    data = pd.concat(parts, ignore_index=True)
    data.to_csv(OUT, index=False)

    print(f"\n{'run':40s} {'IRlen':>5} {'TClen':>5} {'used':>5} {'corr(ir_bulk,TC10)':>18}")
    for k, li, lt, n, r, flag in report:
        print(f"{k.replace('h6_',''):40s} {li:5d} {lt:5d} {n:5d} {r:18.2f}{flag}")
    rs = np.array([r for *_, r, _ in report])
    print(f"\nrows={len(data)}  ->  {OUT}")
    print(f"alignment check: median corr(ir_bulk,TC10) = {np.nanmedian(rs):.2f}  "
          f"({(rs > 0.5).sum()}/{len(rs)} runs > 0.5)")
    nmis = sum(1 for *_, f in report if f)
    if nmis:
        print(f"{nmis} run(s) flagged for length mismatch (review trimming).")


if __name__ == "__main__":
    main()
