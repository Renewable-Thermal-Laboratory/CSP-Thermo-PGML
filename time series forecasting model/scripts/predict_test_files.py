"""Predict held-out test files at multiple horizons from the first 20 s of each file.

For every test file we already use in the sweep:
  • Take rows 0..19 as the input window (first 20 seconds — dt=1 s).
  • For each H in [60, 180, 300, 480, 600, 900]:
      - load the best checkpoint trained for that H,
      - forecast the bin-temperature profile at the target time t = 19 + H,
      - compare to the actual bin temperatures derived from the raw TCs.

The trained models are bin models (paper-aligned, Config.bin_target=True):
  TC11 dataset → 10 bins (avg of TC_i and TC_{i+1} for i=1..10).
  TC10 dataset → 9 bins (avg of TC_i and TC_{i+1} for i=1..9).

Outputs:
  output/test_predictions_TC11.csv
  output/test_predictions_TC10.csv

Each row = one (file, H) pair, with columns:
  file, h, flux, abs, surf, t_input_start, t_input_end, H, t_target,
  BIN1_actual..BINn_actual,
  BIN1_pred..BINn_pred,
  TC1_actual..TC(n+1)_actual          ← raw TC values at the target time for reference
"""
import argparse
import csv
import os
import re
import sys
from glob import glob

import joblib
import numpy as np
import pandas as pd
import torch

# Make `src` importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for sibling _binutils

from new_model import PhysicsInformedLSTM  # noqa: E402
from _binutils import unbin_anchored        # noqa: E402


# ---------------------------------------------------------------------------
# Dataset configuration — mirrors run_all.py
# ---------------------------------------------------------------------------
DATASETS = {
    'TC11': {
        'data_dir': 'data/output_with_TC11',
        'scaler_dir': 'models_TC11',
        'output_root': 'output',                # output/new_theoretical_TC11_L20_H{H}
        'experiment_name': 'new_theoretical_TC11',
        'num_sensors': 11,                       # raw sensor count
        'num_outputs': 10,                       # bin count (always num_sensors-1)
        # The TC11 sweep in run_all.py does NOT train H=900 (only TC10 does).
        'horizons': [60, 180, 300, 480, 600],
        'test_files': [
            'h6_flux88_abs20_surf0_781s - Sheet2.csv',
            'h6_flux88_abs92_surf0_648s - Sheet3.csv',
            'h6_flux88_abs0_surf1_790s - Sheet1.csv',
            'h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv',
        ],
        'out_csv': 'output/test_predictions_TC11.csv',
    },
    'TC10': {
        'data_dir': 'data/processed_H6',
        'scaler_dir': 'models_TC10',
        'output_root': 'output',
        'experiment_name': 'new_theoretical_TC10',
        'num_sensors': 10,
        'num_outputs': 9,
        'horizons': [60, 180, 300, 480, 600, 900],
        'test_files': [
            'h6_flux88_abs0_surf1_790s - Sheet1.csv',
            'h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv',
            'h6_flux88_abs20_surf0_longRun_612s - Sheet2.csv',
        ],
        'out_csv': 'output/test_predictions_TC10.csv',
    },
}
SEQ_LEN = 20            # 20-second input window
TIME_MEAN = 300.0       # matches training pipeline
TIME_STD  = 300.0


# ---------------------------------------------------------------------------
def find_tc_columns(df: pd.DataFrame, num_sensors: int):
    """Return TC column names in sensor-number order (handles 'TC1_tip' etc.)."""
    cols = []
    for c in df.columns:
        if 'TC' in c.upper():
            m = re.findall(r'\d+', c)
            if m:
                n = int(m[0])
                if 1 <= n <= num_sensors:
                    cols.append((n, c))
    cols.sort(key=lambda x: x[0])
    out = [c for _, c in cols]
    if len(out) != num_sensors:
        raise RuntimeError(f"expected {num_sensors} TC columns, found {out}")
    return out


def bin_average(arr: np.ndarray) -> np.ndarray:
    """0.5 * (arr[i] + arr[i+1]) across the LAST axis. Works on (T, S) or (S,)."""
    return 0.5 * (arr[..., :-1] + arr[..., 1:])


def load_model_for_horizon(output_root: str, experiment_name: str,
                           H: int, num_outputs: int, device: torch.device):
    """Load the best checkpoint for a given horizon. Returns model in eval mode."""
    ckpt_dir = os.path.join(output_root, f"{experiment_name}_L{SEQ_LEN}_H{H}")
    ckpt_path = os.path.join(ckpt_dir, f"best_model_L{SEQ_LEN}_H{H}.pth")
    if not os.path.exists(ckpt_path):
        # DO NOT fall back to timestamped older checkpoints — those have a different
        # architecture (pre-bin-mode, smaller LSTM) and would load silently-wrong weights.
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path} "
            "(did the bin-mode sweep finish for this horizon?)"
        )
    model = PhysicsInformedLSTM(
        num_sensors=num_outputs,
        sequence_length=SEQ_LEN,
        lstm_units=512,
        dropout_rate=0.2,
        residual_prediction=True,
        baseline_mode='persistence',
        horizon_steps=H,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt_path


def predict_one(model, time_series: torch.Tensor, static_params: torch.Tensor,
                thermal_scaler) -> np.ndarray:
    """Run the model on a (1, T, num_outputs+1) input and return UNSCALED bin temps."""
    with torch.no_grad():
        y = model([time_series, static_params])           # (1, num_outputs) scaled
    y_np = y.cpu().numpy()[0]
    return y_np * thermal_scaler.scale_ + thermal_scaler.mean_


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', nargs='+', default=['TC11', 'TC10'],
                    choices=list(DATASETS.keys()),
                    help="which datasets to run (default: both)")
    ap.add_argument('--start-offset', type=int, default=0,
                    help="row index where the 20 s input window begins (default 0 = first 20 s). "
                         "Use a later offset (e.g. 100) to sample a mid-experiment window where the "
                         "model is in its strong regime, rather than the rare cold-start window.")
    args = ap.parse_args()
    start_offset = args.start_offset

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else 'cpu')
    )
    print(f"Using device: {device}")

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        print("\n" + "=" * 72)
        print(f"DATASET: {ds_name}  (raw TCs={cfg['num_sensors']}, bins={cfg['num_outputs']})")
        print("=" * 72)

        thermal_scaler = joblib.load(os.path.join(cfg['scaler_dir'], 'thermal_scaler.save'))
        param_scaler   = joblib.load(os.path.join(cfg['scaler_dir'], 'param_scaler.save'))
        if thermal_scaler.mean_.shape[0] != cfg['num_outputs']:
            sys.exit(
                f"[ABORT] {cfg['scaler_dir']}/thermal_scaler.save has shape "
                f"{thermal_scaler.mean_.shape}, expected ({cfg['num_outputs']},). "
                "Looks like the scaler is from a non-bin training. Re-train in bin mode."
            )

        # Load every requested horizon's model up front. Catch broad load errors so a
        # stale/incompatible checkpoint for one H doesn't kill predictions for the rest.
        models = {}
        for H in cfg['horizons']:
            try:
                m, p = load_model_for_horizon(cfg['output_root'], cfg['experiment_name'],
                                              H, cfg['num_outputs'], device)
                models[H] = m
                print(f"  loaded H={H:<4d} from {os.path.basename(p)}")
            except FileNotFoundError as e:
                print(f"  [SKIP] H={H}: {e}")
            except RuntimeError as e:
                # state_dict shape/key mismatch — almost certainly an old non-bin checkpoint
                print(f"  [SKIP] H={H}: incompatible checkpoint ({str(e)[:80]}...)")

        rows = []
        for fname in cfg['test_files']:
            path = os.path.join(cfg['data_dir'], fname)
            if not os.path.exists(path):
                print(f"  [WARN] test file missing: {path}")
                continue

            df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            if len(df) < start_offset + SEQ_LEN + 1:
                print(f"  [WARN] {fname}: too short ({len(df)} rows) for start_offset={start_offset}")
                continue

            tc_cols = find_tc_columns(df, cfg['num_sensors'])
            param_cols = ['h', 'flux', 'abs', 'surf']

            # --- Build the input window (rows start_offset .. start_offset+19) ---
            win_lo, win_hi = start_offset, start_offset + SEQ_LEN          # [lo, hi)
            window = df.iloc[win_lo:win_hi]
            tc_window = window[tc_cols].values.astype(np.float32)                  # (20, num_sensors)
            bin_window = bin_average(tc_window)                                    # (20, num_outputs)
            bin_window_scaled = (bin_window - thermal_scaler.mean_) / thermal_scaler.scale_
            time_window = window['Time'].values.astype(np.float32).reshape(-1, 1)  # (20, 1)
            time_window_scaled = (time_window - TIME_MEAN) / TIME_STD              # (20, 1)
            ts_np = np.hstack([time_window_scaled, bin_window_scaled])             # (20, 1+num_outputs)

            static_raw = window[param_cols].iloc[0].values.astype(np.float32).reshape(1, -1)
            static_scaled = param_scaler.transform(static_raw)                     # (1, 4)
            sp_t = torch.from_numpy(static_scaled).float().to(device)              # (1, 4)
            ts_t = torch.from_numpy(ts_np).unsqueeze(0).float().to(device)         # (1, 20, 1+num_outputs)

            seq_end_idx = win_hi - 1                                        # last row of the input window
            t_input_start = float(df['Time'].iloc[win_lo])
            t_input_end   = float(df['Time'].iloc[seq_end_idx])
            h, flux, ab, surf = (float(window[c].iloc[0]) for c in param_cols)

            for H in cfg['horizons']:
                if H not in models:
                    continue
                target_idx = seq_end_idx + H        # matches dataset's convention
                if target_idx >= len(df):
                    print(f"  [SKIP] {fname} H={H}: file too short (needs row {target_idx}, has {len(df)})")
                    continue

                pred_bins = predict_one(models[H], ts_t, sp_t, thermal_scaler)     # (num_outputs,)
                tc_actual = df[tc_cols].iloc[target_idx].values.astype(np.float64)  # (num_sensors,)
                bin_actual = bin_average(tc_actual)                                # (num_outputs,)
                # Raw-TC prediction: un-bin anchored on the last observed raw TCs (input window end)
                tc_pred = unbin_anchored(pred_bins, tc_window[-1].astype(np.float64))  # (num_sensors,)
                t_target = float(df['Time'].iloc[target_idx])

                row = {
                    'file': fname,
                    'h': h, 'flux': flux, 'abs': ab, 'surf': surf,
                    'start_offset': start_offset,
                    't_input_start': t_input_start,
                    't_input_end': t_input_end,
                    'H': H,
                    't_target': t_target,
                }
                for i in range(cfg['num_sensors']):
                    row[f'TC{i+1}_actual'] = float(tc_actual[i])
                for i in range(cfg['num_sensors']):
                    row[f'TC{i+1}_pred'] = float(tc_pred[i])
                for i in range(cfg['num_outputs']):
                    row[f'BIN{i+1}_actual'] = float(bin_actual[i])
                for i in range(cfg['num_outputs']):
                    row[f'BIN{i+1}_pred'] = float(pred_bins[i])
                row['MAE_K_tc']  = float(np.mean(np.abs(tc_actual - tc_pred)))     # raw-TC error
                row['MAE_K_bin'] = float(np.mean(np.abs(bin_actual - pred_bins)))  # bin error (model-native)
                row['RMSE_K_tc'] = float(np.sqrt(np.mean((tc_actual - tc_pred) ** 2)))
                rows.append(row)

                print(f"  {fname[:36]:36} H={H:<4d} t={t_target:6.0f}s  "
                      f"MAE_tc={row['MAE_K_tc']:6.3f} K  MAE_bin={row['MAE_K_bin']:6.3f} K")

        if not rows:
            print(f"  [WARN] no rows produced for {ds_name}")
            continue

        # Suffix the output filename with the offset so different windows don't overwrite.
        out_path = cfg['out_csv']
        if start_offset != 0:
            base, ext = os.path.splitext(out_path)
            out_path = f"{base}_offset{start_offset}{ext}"
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        # Stable column order: meta → BIN actual → BIN pred → TC actual → errors
        header = (['file', 'h', 'flux', 'abs', 'surf', 'start_offset',
                   't_input_start', 't_input_end', 'H', 't_target']
                  + [f'TC{i+1}_actual'  for i in range(cfg['num_sensors'])]
                  + [f'TC{i+1}_pred'    for i in range(cfg['num_sensors'])]
                  + [f'BIN{i+1}_actual' for i in range(cfg['num_outputs'])]
                  + [f'BIN{i+1}_pred'   for i in range(cfg['num_outputs'])]
                  + ['MAE_K_tc', 'MAE_K_bin', 'RMSE_K_tc'])
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n  wrote {len(rows)} rows → {out_path}")


if __name__ == '__main__':
    main()
