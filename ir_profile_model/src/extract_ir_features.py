#!/usr/bin/env python3
"""Extract per-second surface features from the top-down IR .h5 runs.

The camera is not fixed, so the molten-salt dish sits at different pixels each run.
For each run we detect the dish once (from a robust median of sampled frames), then
per frame summarise the salt SURFACE the IR sees -- the model's proxy for the
near-surface thermocouples (TC10 = surface, coldest & most variable) and the field
that carries the abs-regime information the interior model needs.

We avoid pixel-perfect rod masking: the thermocouple rake shows as a cold vertical
rod, but robust percentiles over the dish naturally push it into the low tail, so
percentile features are stable without segmenting it out.

Per-run output CSV (data/ir_features/<run>.csv), one row per IR second:
    time            IR-clock second (0-based, matches the .h5 'time')
    ir_center       salt surface near the rake (central ROI, rod-robust mean)
    ir_bulk         salt surface bulk (upper-half median of the dish)
    ir_p95, ir_p99  hot rim / peak
    ir_p25          cooler dish pixels
    ir_std          spatial spread across the dish
    ir_grad         radial gradient  (p90 - ir_center)  ~ rim-vs-centre
    ir_area         dish pixel count (per-run ~constant; a detection sanity check)

Usage:
    python extract_ir_features.py                 # all runs -> data/ir_features/
    python extract_ir_features.py --plot          # + a detection/feature review PNG
"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
IR_DIR = PROJ / "data" / "processed_IR"
OUT_DIR = PROJ / "data" / "ir_features"

CENTER_FRAC = 0.30      # central ROI radius as a fraction of dish radius
HOT_PCTL = 65           # frame percentile that separates dish from background


def read_frames(path, n_sample=None):
    """Return (frames, time) in Celsius. n_sample evenly subsamples for speed."""
    with h5py.File(path, "r") as h:
        sc = float(h.attrs.get("temp_scale", 0.02))
        off = float(h.attrs.get("temp_offset", -100.0))
        T = h["temperature"].shape[0]
        idx = np.arange(T) if n_sample is None else np.linspace(0, T - 1, min(n_sample, T)).astype(int)
        frames = h["temperature"][idx].astype(np.float32) * sc + off
        time = h["time"][idx].astype(np.float64) if "time" in h else idx.astype(np.float64)
    return frames, time


def detect_dish(med):
    """Locate the salt dish on a median frame -> (mask, cx, cy, R)."""
    thr = med > np.percentile(med, HOT_PCTL)
    lab, nl = ndi.label(thr)
    if nl == 0:
        raise ValueError("no hot region found")
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, nl + 1))
    dish = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    cy, cx = ndi.center_of_mass(dish)
    R = float(np.sqrt(dish.sum() / np.pi))
    return dish, float(cx), float(cy), R


def extract_run(path, n_geom=40):
    frames_g, _ = read_frames(path, n_sample=n_geom)
    med = np.median(frames_g, axis=0)
    dish, cx, cy, R = detect_dish(med)

    H, W = med.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    center_roi = r2 <= (CENTER_FRAC * R) ** 2

    frames, time = read_frames(path, n_sample=None)
    rows = []
    for fr, t in zip(frames, time):
        dv = fr[dish]
        cv = fr[center_roi]
        c_hi = cv[cv > np.median(cv)]                 # drop the cold rod pixels (low half)
        d_hi = dv[dv > np.median(dv)]                 # upper-half salt surface
        rows.append((
            float(t),
            float(c_hi.mean()) if c_hi.size else float(cv.mean()),   # ir_center
            float(np.median(d_hi)),                                   # ir_bulk
            float(np.percentile(dv, 95)),                             # ir_p95
            float(np.percentile(dv, 99)),                             # ir_p99
            float(np.percentile(dv, 25)),                             # ir_p25
            float(dv.std()),                                          # ir_std
            float(np.percentile(dv, 90) - (c_hi.mean() if c_hi.size else cv.mean())),  # ir_grad
            int(dish.sum()),                                          # ir_area
        ))
    cols = ["time", "ir_center", "ir_bulk", "ir_p95", "ir_p99", "ir_p25", "ir_std", "ir_grad", "ir_area"]
    df = pd.DataFrame(rows, columns=cols)
    return df, (med, dish, cx, cy, R)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ir-dir", default=str(IR_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--plot", action="store_true", help="write a detection + feature review PNG")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.ir_dir, "*.h5")))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    geoms = {}
    for i, f in enumerate(files, 1):
        stem = Path(f).stem
        try:
            df, geom = extract_run(f)
            df.to_csv(os.path.join(args.out, f"{stem}.csv"), index=False)
            geoms[stem] = geom
            print(f"[{i}/{len(files)}] {stem}: {len(df)}s  "
                  f"center={df['ir_center'].mean():.0f}C bulk={df['ir_bulk'].mean():.0f}C R={geom[4]:.0f}")
        except Exception as e:
            print(f"[{i}/{len(files)}] {stem}: FAILED {type(e).__name__}: {e}")

    if args.plot:
        _review_plot(files[:6], args.out, geoms)


def _review_plot(files, out, geoms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    for j, f in enumerate(files):
        stem = Path(f).stem
        med, dish, cx, cy, R = geoms[stem]
        ax = axes[0, j]
        ax.imshow(med, cmap="inferno")
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), "lime", lw=1)
        ax.plot(cx + CENTER_FRAC * R * np.cos(th), cy + CENTER_FRAC * R * np.sin(th), "cyan", lw=1)
        ax.set_title(stem.replace("h6_", ""), fontsize=7)
        ax.axis("off")
        df = pd.read_csv(os.path.join(out, f"{stem}.csv"))
        ax2 = axes[1, j]
        for c in ["ir_center", "ir_bulk", "ir_p99"]:
            ax2.plot(df["time"], df[c], lw=0.8, label=c)
        ax2.set_xlabel("IR time (s)"); ax2.tick_params(labelsize=7)
        if j == 0:
            ax2.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out, "_review.png")
    fig.savefig(p, dpi=90)
    print(f"review plot -> {p}")


if __name__ == "__main__":
    main()
