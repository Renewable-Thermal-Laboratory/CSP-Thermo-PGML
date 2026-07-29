#!/usr/bin/env python3
"""Process raw thermocouple .xlsm/.xlsx runs into clean, aligned per-run CSVs.

Pipeline (per file), following the agreed spec:
  1. TC1 = TC_Bottom_rec_groove if present (else TC1_tip), renamed to TC1.
  2. Keep TC2..TC10 (present in every file).
  3. abs92 files: also keep the 9.5 channel (TC_9.5 or TC_9.5_2), renamed TC9.5,
     placed between TC9 and TC10.
  4. Order columns: time, TC1, TC2..TC9, [TC9.5], TC10.
  5/6. The number in the filename (_NNNs_) is the run-start second on the sheet's
     own Time axis: drop every row before it, then renumber a fresh 'time' from 1.
  7. Trim the cooldown tail: cut from where all TCs begin dropping continuously to
     the end. The cut second (in the new 1-based time) is logged to cooldown_cuts.txt.

Non-TC sensors (TC_botm_Ind, TC_wall_ins_ext, TC_bottom_ins_groove) are dropped.

Usage:
    python process_tc.py "/Users/bhuvan/Desktop/raw IR" -o "/Users/bhuvan/Desktop/processed_TC"
    python process_tc.py "..." -o "..." --no-cooldown-trim   # skip step 7
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TC_MAIN = [f"TC{i}" for i in range(2, 11)]          # TC2..TC10
TC1_SOURCES = ["TC_Bottom_rec_groove", "TC1_tip"]   # preference order
TC95_SOURCES = ["TC_9.5", "TC_9.5_2"]
START_RE = re.compile(r"_(\d+)s_")


def original_sheet(xl):
    for s in xl.sheet_names:
        if s.strip().lower() == "original":
            return s
    return xl.sheet_names[0]


def load_clean(path):
    """Return the 'Original' sheet as data rows only (units/blank rows removed),
    with a numeric Time column and stripped column names."""
    xl = pd.ExcelFile(path)
    df = xl.parse(original_sheet(xl))
    df.columns = [str(c).strip() for c in df.columns]
    if "Time" not in df.columns:
        raise ValueError("no 'Time' column")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df[df["Time"].notna()].reset_index(drop=True)   # drops units + blank rows
    return df


def pick_source(df, candidates, required=True):
    """First candidate column that exists and has real numeric data."""
    for c in candidates:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().mean() > 0.5:
                return c
    if required:
        raise ValueError(f"none of {candidates} present with data")
    return None


def find_cooldown_cut(tc, smooth=21, tol=0.5, min_len=5, min_drop=2.0):
    """Index from which ALL channels are past their peak and decreasing to the end.

    tc: (N, C) array of the output TC channels over the run (1 Hz).
    Returns (cut_index, info). cut_index == N means "no cooldown trim".

    Per the spec: cut where all thermocouples decrease monotonically together. For
    each channel we smooth it and find the last time it sits at its running maximum
    (within tol) -- after that point it is below its peak and heading down. The cut
    is the LATEST such point across all channels, i.e. where the last channel turns
    over; from there every channel is declining. Channels that keep rising to the
    end (longruns, abs92 absorber channels) hold the cut near the end, so little or
    nothing is trimmed -- consistent with 'all must be decreasing'.
    """
    n = len(tc)
    if n < smooth * 2:
        return n, "too short"
    onsets = []
    for c in range(tc.shape[1]):
        s = pd.Series(tc[:, c]).rolling(smooth, center=True, min_periods=1).mean().values
        runmax = np.maximum.accumulate(s)
        at_peak = np.where(s >= runmax - tol)[0]      # indices still touching the peak
        onsets.append(at_peak[-1] if len(at_peak) else n - 1)
    cut = int(max(onsets))                            # last channel to leave its peak
    if n - cut < min_len:
        return n, f"no cooldown (last channel peaks at {cut}/{n})"
    m = tc.mean(axis=1)
    drop = float(m[cut] - m[-1])
    if drop < min_drop:
        return n, f"decline only {drop:.2f}C < {min_drop}"
    return cut, f"all-decrease@{cut}s drop {drop:.2f}C over {n-1-cut}s"


def process_file(path, out_dir, do_cooldown=True, overrides=None):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    is92 = "abs92" in name.lower()
    m = START_RE.search(name)
    if not m:
        raise ValueError("no _NNNs_ start time in filename")
    start = int(m.group(1))

    df = load_clean(path)

    tc1_src = pick_source(df, TC1_SOURCES)
    out = pd.DataFrame()
    out["TC1"] = pd.to_numeric(df[tc1_src], errors="coerce")
    for c in [f"TC{i}" for i in range(2, 10)]:        # TC2..TC9
        out[c] = pd.to_numeric(df[c], errors="coerce")
    tc95_src = None
    if is92:
        tc95_src = pick_source(df, TC95_SOURCES)
        out["TC9.5"] = pd.to_numeric(df[tc95_src], errors="coerce")
    out["TC10"] = pd.to_numeric(df["TC10"], errors="coerce")

    time_orig = df["Time"].values
    out.insert(0, "_orig_time", time_orig)

    # step 5/6: drop pre-run rows, renumber time from 1
    kept = out[out["_orig_time"] >= start].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(f"start {start} beyond recording (max {time_orig.max():.0f})")
    kept = kept.drop(columns="_orig_time")
    kept.insert(0, "time", np.arange(1, len(kept) + 1))

    # step 7: cooldown trim (manual override wins over the auto-detector)
    tc_cols = [c for c in kept.columns if c != "time"]
    cut_time = None
    cut_info = "skipped"
    override = (overrides or {}).get(stem)
    if override is not None:
        cut_idx = int(override)                    # keep time 1..override inclusive
        if 0 < cut_idx < len(kept):
            cut_time = int(override)               # last second kept
            cut_info = f"manual keep<={override}"
            kept = kept.iloc[:cut_idx].reset_index(drop=True)
        else:
            cut_info = f"manual@{override} out of range (n={len(kept)})"
    elif do_cooldown:
        cut_idx, cut_info = find_cooldown_cut(kept[tc_cols].values)
        if cut_idx < len(kept):
            cut_time = int(kept["time"].iloc[cut_idx])
            kept = kept.iloc[:cut_idx].reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_dir / f"{stem}.csv", index=False)

    return {
        "file": name, "start": start, "is92": is92,
        "tc1_source": tc1_src, "tc95_source": tc95_src,
        "rows_out": len(kept), "cut_time": cut_time, "cut_info": cut_info,
        "n_channels": len(tc_cols),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of .xlsm/.xlsx files (or a single file)")
    ap.add_argument("-o", "--out", required=True, help="output folder for CSVs")
    ap.add_argument("--no-cooldown-trim", action="store_true", help="skip step 7")
    ap.add_argument("--overrides", help="CSV/TSV of 'run_stem, cut_time' manual cooldown cuts "
                    "(default: overrides.csv next to this script, if present)")
    args = ap.parse_args()

    ov_path = Path(args.overrides) if args.overrides else Path(__file__).with_name("overrides.csv")
    overrides = {}
    if ov_path.exists():
        odf = pd.read_csv(ov_path)
        odf.columns = [c.strip().lower() for c in odf.columns]
        key = "run_stem" if "run_stem" in odf.columns else odf.columns[0]
        val = "cut_time" if "cut_time" in odf.columns else odf.columns[1]
        for _, rr in odf.iterrows():
            stem = os.path.splitext(str(rr[key]).strip())[0]
            overrides[stem] = int(rr[val])
        print(f"loaded {len(overrides)} manual override(s) from {ov_path.name}")

    p = Path(args.input)
    files = sorted([f for f in p.glob("*.xls*")]) if p.is_dir() else [p]
    if not files:
        sys.exit("no .xls* files found")

    rows = []
    cut_lines = []
    for f in files:
        try:
            r = process_file(f, args.out, do_cooldown=not args.no_cooldown_trim, overrides=overrides)
            rows.append(r)
            ct = r["cut_time"] if r["cut_time"] is not None else "-"
            flag = "  [TC1_tip!]" if r["tc1_source"] == "TC1_tip" else ""
            print(f"  {r['file']:48s} start={r['start']:5d} rows={r['rows_out']:5d} "
                  f"chans={r['n_channels']} cut@{str(ct):>5}  ({r['cut_info']}){flag}")
            if r["cut_time"] is not None:
                cut_lines.append(f"{r['file']}\t{r['cut_time']}")
        except Exception as e:
            print(f"  !! {os.path.basename(str(f))}: {type(e).__name__}: {e}", file=sys.stderr)

    out_dir = Path(args.out)
    with open(out_dir / "cooldown_cuts.txt", "w") as fh:
        fh.write("filename\tcut_time_s\n")
        fh.write("\n".join(cut_lines) + ("\n" if cut_lines else ""))
    pd.DataFrame(rows).to_csv(out_dir / "process_summary.csv", index=False)
    print(f"\n{len(rows)} files -> {out_dir}")
    print(f"cooldown cuts logged: {len(cut_lines)} (see cooldown_cuts.txt)")
    tip = [r["file"] for r in rows if r["tc1_source"] == "TC1_tip"]
    if tip:
        print(f"TC1 fell back to TC1_tip in: {tip}")


if __name__ == "__main__":
    main()
