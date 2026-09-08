"""Rank files by how well the model predicts them ACROSS ALL horizons.

For each file, computes the mean MAE over ~40 evenly-spaced windows at every horizon
the file is long enough to support, then ranks files by their average MAE across all
supported horizons. "Best file for all horizons" = lowest average, full horizon coverage.

  TC11 dataset:  H = 60, 180, 300, 480, 600
  TC10 dataset:  H = 60, 180, 300, 480, 600, 900

Usage:  python3 scripts/best_files.py
Output: prints ranking + saves output/best_files_ranking.csv
"""
import csv, glob, os, re, sys
import joblib, numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from new_model import PhysicsInformedLSTM  # noqa: E402

SEQ_LEN, TIME_MEAN, TIME_STD = 20, 300.0, 300.0
N_WIN = 40   # windows sampled per (file, horizon)

JOBS = [
    dict(ds='TC11', data_dir='data/output_with_TC11', scaler_dir='models_TC11',
         exp='new_theoretical_TC11', num_sensors=11, horizons=[60, 180, 300, 480, 600]),
    dict(ds='TC10', data_dir='data/processed_H6', scaler_dir='models_TC10',
         exp='new_theoretical_TC10', num_sensors=10, horizons=[60, 180, 300, 480, 600, 900]),
]


def tc_cols(df, ns):
    return sorted([c for c in df.columns if 'TC' in c.upper()],
                  key=lambda c: int(re.findall(r'\d+', c)[0]))[:ns]


def load_model(exp, H, n_out, dev):
    ck = f"output/{exp}_L{SEQ_LEN}_H{H}/best_model_L{SEQ_LEN}_H{H}.pth"
    st = torch.load(ck, map_location=dev)
    m = PhysicsInformedLSTM(num_sensors=n_out, sequence_length=SEQ_LEN, lstm_units=512,
                            dropout_rate=0.2, residual_prediction=True,
                            baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(st); m.to(dev).eval()
    return m


def file_horizon_mae(model, df, cols, tsc, psc, is_raw, H, dev):
    """Mean MAE over up to N_WIN evenly-spaced windows for one file at horizon H."""
    last = len(df) - SEQ_LEN - H
    if last <= 0:
        return None
    offs = np.unique(np.linspace(0, last, min(N_WIN, last + 1), dtype=int))
    tcvals = df[cols].values.astype(np.float64)
    times = df['Time'].values.astype(np.float64)
    sp = psc.transform([[float(df[c].iloc[0]) for c in ['h', 'flux', 'abs', 'surf']]])
    batch = []
    targets = []
    for off in offs:
        tcw = tcvals[off:off + SEQ_LEN]
        temp_in = tcw if is_raw else 0.5 * (tcw[:, :-1] + tcw[:, 1:])
        ts_np = np.hstack([((times[off:off + SEQ_LEN] - TIME_MEAN) / TIME_STD).reshape(-1, 1),
                           (temp_in - tsc.mean_) / tsc.scale_])
        batch.append(ts_np)
        targets.append(tcvals[off + SEQ_LEN - 1 + H])
    ts_t = torch.from_numpy(np.stack(batch)).float().to(dev)
    sp_t = torch.from_numpy(np.repeat(sp, len(offs), axis=0)).float().to(dev)
    with torch.no_grad():
        out = model([ts_t, sp_t]).cpu().numpy()
    out = out * tsc.scale_ + tsc.mean_
    tgt = np.stack(targets)
    return float(np.mean(np.abs(out - tgt)))


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device: {dev}")
    for job in JOBS:
        ns = job['num_sensors']
        tsc = joblib.load(os.path.join(job['scaler_dir'], 'thermal_scaler.save'))
        psc = joblib.load(os.path.join(job['scaler_dir'], 'param_scaler.save'))
        n_out = tsc.mean_.shape[0]; is_raw = (n_out == ns)
        models = {H: load_model(job['exp'], H, n_out, dev) for H in job['horizons']}
        files = sorted(glob.glob(os.path.join(job['data_dir'], '*.csv')))

        rows = []
        for path in files:
            df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            cols = tc_cols(df, ns)
            if len(cols) != ns:
                continue
            maes = {}
            for H in job['horizons']:
                m = file_horizon_mae(models[H], df, cols, tsc, psc, is_raw, H, dev)
                if m is not None:
                    maes[H] = m
            if not maes:
                continue
            rows.append(dict(file=os.path.basename(path),
                             abs=float(df['abs'].iloc[0]), surf=float(df['surf'].iloc[0]),
                             flux=float(df['flux'].iloc[0]), rows=len(df),
                             maes=maes, cover=len(maes),
                             avg=float(np.mean(list(maes.values())))))

        full = [r for r in rows if r['cover'] == len(job['horizons'])]
        full.sort(key=lambda r: r['avg'])
        print('\n' + '=' * 100)
        print(f"{job['ds']} — files ranked by AVG MAE across ALL {len(job['horizons'])} horizons "
              f"(full coverage only)")
        print('=' * 100)
        hdr = ' '.join(f'H{H}' .rjust(7) for H in job['horizons'])
        print(f"{'rank':>4} {'avg':>6}  {hdr}   abs surf  file")
        for i, r in enumerate(full[:12], 1):
            per = ' '.join(f"{r['maes'][H]:7.2f}" for H in job['horizons'])
            print(f"{i:>4} {r['avg']:6.2f}  {per}  {r['abs']:4.0f} {r['surf']:4.2f}  {r['file'][:40]}")

        # save full ranking (all files, with whatever coverage)
        rows.sort(key=lambda r: r['avg'])
        out = f"output/best_files_ranking_{job['ds']}.csv"
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'file', 'abs', 'surf', 'flux', 'rows', 'horizons_covered', 'avg_MAE_K']
                       + [f'MAE_H{H}' for H in job['horizons']])
            for i, r in enumerate(rows, 1):
                w.writerow([i, r['file'], r['abs'], r['surf'], r['flux'], r['rows'], r['cover'],
                            round(r['avg'], 3)] + [round(r['maes'].get(H, float('nan')), 3) for H in job['horizons']])
        print(f"  saved -> {out}")


if __name__ == '__main__':
    main()
