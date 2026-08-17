"""Predicted-vs-actual raw TCs for all test files, per horizon — using the MATCHED
per-experiment scaler saved next to each checkpoint (fixes the shared-scaler bug).

For each (dataset, horizon) it loads:
  • the checkpoint at output/<exp>_L20_H<H>/best_model_L20_H<H>.pth, and
  • the scaler saved in that SAME dir (output/<exp>_L20_H<H>/{thermal,param}_scaler.save).
If the per-experiment scaler isn't present (e.g. an experiment trained before the fix),
it falls back to re-fitting the scaler for that horizon via the dataset builder.

Usage:
    python3 scripts/predict_all_tc.py                       # both datasets, offset 100
    python3 scripts/predict_all_tc.py --datasets TC10       # TC10 only
    python3 scripts/predict_all_tc.py --start-offset 0
    python3 scripts/predict_all_tc.py --files "<file>.csv"  # specific file(s)

Output: output/predictions_all_TC_<DS>_offset<N>[<tag>].csv
"""
import argparse, contextlib, csv, glob, io, os, re, sys
import joblib, numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from new_model import PhysicsInformedLSTM            # noqa: E402
from new_dataset_builder import TempSequenceDataset  # noqa: E402
from _binutils import unbin_anchored                  # noqa: E402

SEQ_LEN, TM, TS = 20, 300.0, 300.0
PARAM_COLS = ['h', 'flux', 'abs', 'surf']

DATASETS = {
    'TC11': dict(data_dir='data/output_with_TC11', output_root='output',
                 experiment_name='new_theoretical_TC11', num_sensors=11, horizons=[60, 180, 300, 480, 600],
                 test_files=['h6_flux88_abs20_surf0_781s - Sheet2.csv', 'h6_flux88_abs92_surf0_648s - Sheet3.csv',
                             'h6_flux88_abs0_surf1_790s - Sheet1.csv', 'h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv']),
    'TC10': dict(data_dir='data/processed_H6', output_root='output',
                 experiment_name='new_theoretical_TC10', num_sensors=10, horizons=[60, 180, 300, 480, 600, 900],
                 test_files=['h6_flux88_abs0_surf1_790s - Sheet1.csv', 'h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv',
                             'h6_flux88_abs20_surf0_longRun_612s - Sheet2.csv']),
}


def tc_cols(df, ns):
    return sorted([c for c in df.columns if 'TC' in c.upper()], key=lambda c: int(re.findall(r'\d+', c)[0]))[:ns]


def get_scalers(cfg, H):
    """Matched per-experiment scaler if present, else re-fit for this horizon."""
    d = f"{cfg['output_root']}/{cfg['experiment_name']}_L{SEQ_LEN}_H{H}"
    tp, pp = os.path.join(d, 'thermal_scaler.save'), os.path.join(d, 'param_scaler.save')
    if os.path.exists(tp) and os.path.exists(pp):
        return joblib.load(tp), joblib.load(pp), 'matched'
    with contextlib.redirect_stdout(io.StringIO()):
        ds = TempSequenceDataset(data_dir=cfg['data_dir'], scaler_dir='/tmp/refit_pred', num_sensors=cfg['num_sensors'],
                                 zscore_threshold=1e9, prediction_horizon=H, target_test_files=cfg['test_files'],
                                 bin_target=False, split='train')
    return ds.thermal_scaler, ds.param_scaler, 're-fit'


def load_model(cfg, H, n_out, dev):
    ck = f"{cfg['output_root']}/{cfg['experiment_name']}_L{SEQ_LEN}_H{H}/best_model_L{SEQ_LEN}_H{H}.pth"
    st = torch.load(ck, map_location=dev)
    if st['output_dense.bias'].shape[0] != n_out:
        raise RuntimeError(f"checkpoint out-dim {st['output_dense.bias'].shape[0]} != scaler {n_out}")
    m = PhysicsInformedLSTM(num_sensors=n_out, sequence_length=SEQ_LEN, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(st); m.to(dev).eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['TC11', 'TC10'], choices=list(DATASETS))
    ap.add_argument('--start-offset', type=int, default=100)
    ap.add_argument('--files', nargs='+', default=None)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    off = args.start_offset
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device {dev}  offset {off}")

    for dsn in args.datasets:
        cfg = DATASETS[dsn]; ns = cfg['num_sensors']
        files = args.files if args.files else cfg['test_files']
        rows = []
        print(f"\n===== {dsn} =====")
        for H in cfg['horizons']:
            try:
                tsc, psc, src = get_scalers(cfg, H)
            except Exception as e:
                print(f"  [SKIP] H={H}: scaler error {e}"); continue
            n_out = tsc.mean_.shape[0]; is_raw = (n_out == ns)
            try:
                model = load_model(cfg, H, n_out, dev)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  [SKIP] H={H}: {str(e)[:60]}"); continue
            for fname in files:
                path = os.path.join(cfg['data_dir'], fname)
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                cols = tc_cols(df, ns)
                if len(cols) != ns or len(df) < off + SEQ_LEN + H + 1:
                    continue
                win = df.iloc[off:off + SEQ_LEN]
                tcw = win[cols].values.astype(np.float64)
                temp_in = tcw if is_raw else 0.5 * (tcw[:, :-1] + tcw[:, 1:])
                ts_np = np.hstack([((win['Time'].values - TM) / TS).reshape(-1, 1), (temp_in - tsc.mean_) / tsc.scale_])
                sp = psc.transform([[float(win[c].iloc[0]) for c in PARAM_COLS]])
                with torch.no_grad():
                    o = model([torch.from_numpy(ts_np).unsqueeze(0).float().to(dev),
                               torch.from_numpy(sp).float().to(dev)]).cpu().numpy()[0]
                o = o * tsc.scale_ + tsc.mean_
                tc_pred = o if is_raw else unbin_anchored(o, tcw[-1])
                seq_end = off + SEQ_LEN - 1
                tgt = seq_end + H
                tc_act = df[cols].iloc[tgt].values.astype(np.float64)
                row = dict(file=fname, H=H, scaler_src=src,
                           t_input_start=float(df['Time'].iloc[off]), t_input_end=float(df['Time'].iloc[seq_end]),
                           t_target=float(df['Time'].iloc[tgt]),
                           MAE_K=round(float(np.mean(np.abs(tc_pred - tc_act))), 3),
                           RMSE_K=round(float(np.sqrt(np.mean((tc_pred - tc_act) ** 2))), 3))
                for i in range(ns): row[f'TC{i+1}_actual'] = round(float(tc_act[i]), 2)
                for i in range(ns): row[f'TC{i+1}_pred'] = round(float(tc_pred[i]), 2)
                rows.append(row)
                print(f"  H={H:<4d} [{src:>5}] {fname[:34]:34} t={row['t_target']:6.0f}s  MAE={row['MAE_K']:6.3f} K")
        if not rows:
            print(f"  (no rows for {dsn})"); continue
        tag = f"_{args.tag}" if args.tag else ("_custom" if args.files else "")
        out = f"output/predictions_all_TC_{dsn}_offset{off}{tag}.csv"
        hdr = (['file', 'H', 'scaler_src', 't_input_start', 't_input_end', 't_target', 'MAE_K', 'RMSE_K']
               + [f'TC{i+1}_actual' for i in range(ns)] + [f'TC{i+1}_pred' for i in range(ns)])
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"  wrote {len(rows)} rows -> {out}")


if __name__ == '__main__':
    main()
