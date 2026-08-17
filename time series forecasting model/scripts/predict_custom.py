"""Parametric prediction tool driven by an editable request file.

You fill in `predict_requests.csv` (one row per prediction you want) and run:

    python3 scripts/predict_custom.py
    python3 scripts/predict_custom.py --requests predict_requests.csv --out output/custom_predictions.csv

IMPORTANT — how the model works:
    The PG-LSTM is a *forecaster*. Its forward pass needs a 20-second window of recent
    bin temperatures as input — it cannot predict from (h, flux, abs, surf, time) alone.
    So each request still needs a 20 s temperature "seed". This tool gets that seed from
    a real experimental run:
        • seed_file given  -> uses that file's recent history.
        • seed_file blank  -> auto-picks the run in the dataset folder whose operating
                              params (h, flux, abs, surf) are closest to your request.
    The static parameters fed to the model are always YOUR requested values, so you can
    do "what-if" runs (real recent history + hypothetical operating conditions). When your
    requested params differ from the seed run's params, the row is flagged WHAT-IF and no
    ground-truth "actual" is reported.

Request file columns (predict_requests.csv):
    dataset          TC11 or TC10  (selects model family + bin count)
    h, flux, abs, surf   operating conditions (flux = q0 in W/m^2)
    horizon_s        forecast horizon in seconds; must be a trained horizon:
                       TC11 -> 60,180,300,480,600 ;  TC10 -> 60,180,300,480,600,900
    seed_file        (optional) experimental CSV to draw the 20 s history from
    seed_end_time_s  (optional) Time(s) at which the 20 s input window ends (default 120)

Output CSV columns:
    request echo + seed_file_used, seed_end_time_s, t_target_s, match,
    BIN1..n_pred, BIN1..n_actual, MAE_K
"""
import argparse
import csv
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for sibling _binutils
from new_model import PhysicsInformedLSTM  # noqa: E402
from _binutils import unbin_anchored        # noqa: E402

SEQ_LEN = 20
TIME_MEAN = 300.0
TIME_STD = 300.0

DATASETS = {
    'TC11': {
        'data_dir': 'data/output_with_TC11',
        'scaler_dir': 'models_TC11',
        'output_root': 'output',
        'experiment_name': 'new_theoretical_TC11',
        'num_sensors': 11,
        'num_outputs': 10,
        'horizons': [60, 180, 300, 480, 600],
    },
    'TC10': {
        'data_dir': 'data/processed_H6',
        'scaler_dir': 'models_TC10',
        'output_root': 'output',
        'experiment_name': 'new_theoretical_TC10',
        'num_sensors': 10,
        'num_outputs': 9,
        'horizons': [60, 180, 300, 480, 600, 900],
    },
}

PARAM_COLS = ['h', 'flux', 'abs', 'surf']


def find_tc_columns(df, num_sensors):
    cols = []
    for c in df.columns:
        if 'TC' in c.upper():
            m = re.findall(r'\d+', c)
            if m and 1 <= int(m[0]) <= num_sensors:
                cols.append((int(m[0]), c))
    cols.sort(key=lambda x: x[0])
    out = [c for _, c in cols]
    if len(out) != num_sensors:
        raise RuntimeError(f"expected {num_sensors} TC columns, found {out}")
    return out


def bin_average(arr):
    return 0.5 * (arr[..., :-1] + arr[..., 1:])


def load_model(cfg, H, device, _cache={}):
    key = (cfg['experiment_name'], H)
    if key in _cache:
        return _cache[key]
    ckpt = os.path.join(cfg['output_root'], f"{cfg['experiment_name']}_L{SEQ_LEN}_H{H}",
                        f"best_model_L{SEQ_LEN}_H{H}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt} (sweep didn't train this horizon?)")
    m = PhysicsInformedLSTM(num_sensors=cfg['num_outputs'], sequence_length=SEQ_LEN,
                            lstm_units=512, dropout_rate=0.2, residual_prediction=True,
                            baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.to(device).eval()
    _cache[key] = m
    return m


def list_dataset_files(cfg):
    import glob
    return sorted(glob.glob(os.path.join(cfg['data_dir'], '*.csv')))


def auto_pick_seed(cfg, req_params, param_scaler):
    """Pick the dataset run whose (h,flux,abs,surf) are closest to req_params (z-scored)."""
    best, best_d = None, np.inf
    req = np.array([req_params[c] for c in PARAM_COLS], dtype=float)
    scale = param_scaler.scale_
    for f in list_dataset_files(cfg):
        try:
            d = pd.read_csv(f, nrows=1)
            p = np.array([float(d[c].iloc[0]) for c in PARAM_COLS])
        except Exception:
            continue
        dist = np.sqrt(np.sum(((p - req) / scale) ** 2))
        if dist < best_d:
            best_d, best = dist, f
    return best, best_d


def seed_params(seed_path):
    d = pd.read_csv(seed_path, nrows=1)
    return {c: float(d[c].iloc[0]) for c in PARAM_COLS}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--requests', default='predict_requests.csv',
                    help="path to the request CSV (default: predict_requests.csv)")
    ap.add_argument('--out', default='output/custom_predictions.csv',
                    help="path to write predictions (default: output/custom_predictions.csv)")
    args = ap.parse_args()

    if not os.path.exists(args.requests):
        sys.exit(f"request file not found: {args.requests}")

    device = torch.device('cuda' if torch.cuda.is_available()
                          else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                                else 'cpu'))
    print(f"Using device: {device}")

    reqs = pd.read_csv(args.requests).to_dict('records')
    scalers = {}   # dataset -> (thermal, param)
    out_rows = []
    max_bins = 0

    for i, req in enumerate(reqs):
        ds = str(req['dataset']).strip()
        if ds not in DATASETS:
            print(f"[row {i}] [SKIP] unknown dataset {ds!r} (use TC11 or TC10)")
            continue
        cfg = DATASETS[ds]
        max_bins = max(max_bins, cfg['num_outputs'])

        H = int(req['horizon_s'])
        if H not in cfg['horizons']:
            print(f"[row {i}] [SKIP] {ds} H={H}s not trained. Available: {cfg['horizons']}")
            continue

        if ds not in scalers:
            scalers[ds] = (joblib.load(os.path.join(cfg['scaler_dir'], 'thermal_scaler.save')),
                           joblib.load(os.path.join(cfg['scaler_dir'], 'param_scaler.save')))
        thermal_scaler, param_scaler = scalers[ds]

        req_params = {c: float(req[c]) for c in PARAM_COLS}

        # --- resolve seed file (blank cells arrive as pandas NaN) ---
        seed_file = req.get('seed_file', '')
        seed_file = '' if (seed_file is None or (isinstance(seed_file, float) and pd.isna(seed_file))) else str(seed_file).strip()
        if seed_file:
            seed_path = os.path.join(cfg['data_dir'], seed_file)
            if not os.path.exists(seed_path):
                print(f"[row {i}] [SKIP] seed_file not found: {seed_path}")
                continue
            pick_dist = 0.0
        else:
            seed_path, pick_dist = auto_pick_seed(cfg, req_params, param_scaler)
            if seed_path is None:
                print(f"[row {i}] [SKIP] could not auto-pick a seed file in {cfg['data_dir']}")
                continue

        sdf = pd.read_csv(seed_path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        tc_cols = find_tc_columns(sdf, cfg['num_sensors'])

        # --- locate the 20-row window ending at seed_end_time_s ---
        seed_end = req.get('seed_end_time_s', '')
        seed_end = 120.0 if (seed_end == '' or pd.isna(seed_end)) else float(seed_end)
        # row whose Time is closest to seed_end
        seq_end_idx = int((sdf['Time'] - seed_end).abs().idxmin())
        seq_end_idx = max(SEQ_LEN - 1, min(seq_end_idx, len(sdf) - 1))  # need >=20 rows before
        win_lo = seq_end_idx - (SEQ_LEN - 1)
        window = sdf.iloc[win_lo:seq_end_idx + 1]
        seed_end_actual = float(sdf['Time'].iloc[seq_end_idx])

        # --- build model input ---
        tc_win = window[tc_cols].values.astype(np.float32)            # (20, num_sensors)
        bin_win = bin_average(tc_win)                                  # (20, num_outputs)
        bin_win_s = (bin_win - thermal_scaler.mean_) / thermal_scaler.scale_
        time_win = window['Time'].values.astype(np.float32).reshape(-1, 1)
        time_win_s = (time_win - TIME_MEAN) / TIME_STD
        ts_np = np.hstack([time_win_s, bin_win_s])                    # (20, 1+num_outputs)

        # static branch uses the REQUESTED params (enables what-if)
        static_s = param_scaler.transform([[req_params[c] for c in PARAM_COLS]])
        sp_t = torch.from_numpy(static_s).float().to(device)
        ts_t = torch.from_numpy(ts_np).unsqueeze(0).float().to(device)

        # --- predict ---
        model = load_model(cfg, H, device)
        with torch.no_grad():
            y = model([ts_t, sp_t]).cpu().numpy()[0]
        pred_bins = y * thermal_scaler.scale_ + thermal_scaler.mean_
        # Raw-TC prediction: un-bin anchored on last observed raw TCs (input-window end)
        tc_pred = unbin_anchored(pred_bins, tc_win[-1].astype(np.float64))

        t_target = seed_end_actual + H

        # --- ground truth, only if seed run matches request AND is long enough ---
        sp_run = seed_params(seed_path)
        is_match = all(abs(sp_run[c] - req_params[c]) < 1e-6 for c in PARAM_COLS)
        target_idx = seq_end_idx + H
        actual_bins = None
        tc_actual_arr = None
        if is_match and target_idx < len(sdf):
            tc_actual_arr = sdf[tc_cols].iloc[target_idx].values.astype(np.float64)
            actual_bins = bin_average(tc_actual_arr)

        row = {
            'dataset': ds,
            'h': req_params['h'], 'flux': req_params['flux'],
            'abs': req_params['abs'], 'surf': req_params['surf'],
            'horizon_s': H,
            'seed_file_used': os.path.basename(seed_path),
            'seed_auto_dist': round(pick_dist, 3),
            'seed_end_time_s': seed_end_actual,
            't_target_s': t_target,
            'match': 'MATCH' if is_match else 'WHAT-IF',
            'num_bins': cfg['num_outputs'],
        }
        for s in range(cfg['num_sensors']):
            row[f'TC{s+1}_pred'] = round(float(tc_pred[s]), 2)
        for s in range(cfg['num_sensors']):
            row[f'TC{s+1}_actual'] = (round(float(tc_actual_arr[s]), 2)
                                      if tc_actual_arr is not None else '')
        for b in range(cfg['num_outputs']):
            row[f'BIN{b+1}_pred'] = round(float(pred_bins[b]), 2)
        for b in range(cfg['num_outputs']):
            row[f'BIN{b+1}_actual'] = (round(float(actual_bins[b]), 2)
                                       if actual_bins is not None else '')
        row['MAE_K_tc'] = (round(float(np.mean(np.abs(tc_pred - tc_actual_arr))), 3)
                           if tc_actual_arr is not None else '')
        row['MAE_K_bin'] = (round(float(np.mean(np.abs(pred_bins - actual_bins))), 3)
                            if actual_bins is not None else '')
        out_rows.append(row)

        amae = f"  MAE_tc={row['MAE_K_tc']} K" if row['MAE_K_tc'] != '' else ""
        print(f"[row {i}] {ds} H={H:<4d} t_target={t_target:6.0f}s  "
              f"seed={os.path.basename(seed_path)[:34]:34} {row['match']:7}{amae}")

    if not out_rows:
        sys.exit("no predictions produced — check the request file.")

    # Build a stable union header (bins up to the largest dataset used)
    max_sensors = max_bins + 1
    meta = ['dataset', 'h', 'flux', 'abs', 'surf', 'horizon_s',
            'seed_file_used', 'seed_auto_dist', 'seed_end_time_s', 't_target_s',
            'match', 'num_bins']
    header = (meta
              + [f'TC{s+1}_pred' for s in range(max_sensors)]
              + [f'TC{s+1}_actual' for s in range(max_sensors)]
              + [f'BIN{b+1}_pred' for b in range(max_bins)]
              + [f'BIN{b+1}_actual' for b in range(max_bins)]
              + ['MAE_K_tc', 'MAE_K_bin'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {len(out_rows)} predictions -> {args.out}")


if __name__ == '__main__':
    main()
