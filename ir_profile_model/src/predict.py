#!/usr/bin/env python3
"""Predict the vertical temperature profile from one IR snapshot + bottom TC.

This is the deployment interface: no time history, one frame in -> profile out.

    python predict.py --h5 data/processed_IR/h6_abs0_flux88_surf0_541s_run1.h5 \
                      --frame 300 --tc1 358.2 --flux 88 --abs 0 --surf 0

    # or from an already-extracted feature row (e.g. live camera pipeline):
    python predict.py --features 379.1 376.4 388.0 391.2 370.0 55.1 9.0 \
                      --tc1 358.2 --flux 88 --abs 0 --surf 0

Outputs the profile at the rake positions (TC2..TC10, incl. TC9.5) and, with
--dense, a 50-point continuous curve. Temperatures in deg C.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

# The saved model may have been pickled from a __main__ training script; make the
# wrapper class resolvable under both module paths before loading.
import __main__
from mlp_model import MLPModel
__main__.MLPModel = MLPModel

PROJ = Path(__file__).resolve().parent.parent
MODEL = PROJ / "models" / "profile_model.joblib"


def features_from_h5(path, frame_idx):
    """Extract the same surface features used in training, from one stored frame."""
    import h5py
    from extract_ir_features import detect_dish, CENTER_FRAC, read_frames
    frames_g, _ = read_frames(path, n_sample=40)
    med = np.median(frames_g, axis=0)
    dish, cx, cy, R = detect_dish(med)
    with h5py.File(path, "r") as h:
        sc = float(h.attrs.get("temp_scale", 0.02))
        off = float(h.attrs.get("temp_offset", -100.0))
        fr = h["temperature"][frame_idx].astype(np.float32) * sc + off
    H, W = fr.shape
    yy, xx = np.mgrid[0:H, 0:W]
    croi = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (CENTER_FRAC * R) ** 2
    dv, cv = fr[dish], fr[croi]
    c_hi = cv[cv > np.median(cv)]
    d_hi = dv[dv > np.median(dv)]
    center = float(c_hi.mean()) if c_hi.size else float(cv.mean())
    return [center, float(np.median(d_hi)), float(np.percentile(dv, 95)),
            float(np.percentile(dv, 99)), float(np.percentile(dv, 25)),
            float(dv.std()), float(np.percentile(dv, 90) - center)]


def predict_profile(payload, ir_feats, tc1, flux, absr, surf, zs):
    base = np.array(ir_feats + [tc1, flux, absr, surf], dtype=float)
    z01 = (np.asarray(zs, dtype=float) - 1.0) / 9.0
    X = np.column_stack([np.tile(base, (len(z01), 1)), z01])
    return payload["model"].predict(X) + tc1          # offset target -> absolute deg C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", help="IR .h5 run to take a snapshot from")
    src.add_argument("--features", nargs=7, type=float, metavar=("CENTER", "BULK", "P95", "P99", "P25", "STD", "GRAD"),
                     help="pre-extracted IR features")
    ap.add_argument("--frame", type=int, default=None, help="frame index for --h5 (default: middle)")
    ap.add_argument("--tc1", type=float, required=True, help="bottom thermocouple, deg C")
    ap.add_argument("--flux", type=float, required=True)
    ap.add_argument("--abs", dest="absr", type=float, required=True)
    ap.add_argument("--surf", type=int, required=True)
    ap.add_argument("--dense", action="store_true", help="also print a 50-point continuous profile")
    args = ap.parse_args()

    payload = joblib.load(MODEL)
    if args.h5:
        import h5py
        with h5py.File(args.h5, "r") as h:
            n = h["temperature"].shape[0]
        fi = args.frame if args.frame is not None else n // 2
        feats = features_from_h5(args.h5, fi)
        print(f"snapshot: {Path(args.h5).name} frame {fi}")
    else:
        feats = list(args.features)

    chans = payload["channels"]
    zs = list(chans.values())
    temps = predict_profile(payload, feats, args.tc1, args.flux, args.absr, args.surf, zs)
    print(f"\ninputs: TC1={args.tc1:.1f}C flux={args.flux:g} abs={args.absr:g} surf={args.surf}")
    print("predicted profile (deg C):")
    print(f"  {'TC1':6s} {args.tc1:7.1f}   (input, bottom)")
    for ch, T in zip(chans, temps):
        note = "(surface)" if ch == "TC10" else ""
        print(f"  {ch:6s} {T:7.1f}   {note}")

    if args.dense:
        zd = np.linspace(1, 10, 50)
        Td = predict_profile(payload, feats, args.tc1, args.flux, args.absr, args.surf, zd)
        print("\ndense profile (z, degC):")
        for z, T in zip(zd, Td):
            print(f"  {z:5.2f}  {T:7.2f}")


if __name__ == "__main__":
    main()
