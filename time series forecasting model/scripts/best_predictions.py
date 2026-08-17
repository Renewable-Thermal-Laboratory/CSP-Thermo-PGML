"""Find the BEST-case predicted-vs-actual window for abs0 and abs92 conditions.

For each (dataset, horizon, abs-group) it scans windows across ALL files of that
absorptivity and reports the single lowest-MAE window (file, time, actual vs predicted).

  TC11 dataset:  H = 60, 180, 300, 480, 600
  TC10 dataset:  H = 900
  abs groups: abs0 (column abs==3),  abs92 (column abs==100)

Usage:  python3 scripts/best_predictions.py
Output: prints best windows + saves output/best_predictions.csv
"""
import csv, os, re, sys
import joblib, numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from new_model import PhysicsInformedLSTM  # noqa: E402

SEQ_LEN, TIME_MEAN, TIME_STD = 20, 300.0, 300.0
MAX_WINDOWS = 120   # evenly-spaced windows scanned per file (bounds compute)

JOBS = [
    dict(ds='TC11', data_dir='data/output_with_TC11', scaler_dir='models_TC11',
         exp='new_theoretical_TC11', num_sensors=11, horizons=[60, 180, 300, 480, 600]),
    dict(ds='TC10', data_dir='data/processed_H6', scaler_dir='models_TC10',
         exp='new_theoretical_TC10', num_sensors=10, horizons=[900]),
]
ABS_GROUPS = {'abs0': 3.0, 'abs92': 100.0}


def tc_cols(df, ns):
    cols = sorted([c for c in df.columns if 'TC' in c.upper()],
                  key=lambda c: int(re.findall(r'\d+', c)[0]))[:ns]
    return cols


def load_model(exp, H, n_out, dev):
    ck = f"output/{exp}_L{SEQ_LEN}_H{H}/best_model_L{SEQ_LEN}_H{H}.pth"
    st = torch.load(ck, map_location=dev)
    assert st['output_dense.bias'].shape[0] == n_out, f"dim mismatch H={H}"
    m = PhysicsInformedLSTM(num_sensors=n_out, sequence_length=SEQ_LEN, lstm_units=512,
                            dropout_rate=0.2, residual_prediction=True,
                            baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(st); m.to(dev).eval()
    return m


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device: {dev}")
    results = []

    for job in JOBS:
        ns = job['num_sensors']
        tsc = joblib.load(os.path.join(job['scaler_dir'], 'thermal_scaler.save'))
        psc = joblib.load(os.path.join(job['scaler_dir'], 'param_scaler.save'))
        n_out = tsc.mean_.shape[0]
        is_raw = (n_out == ns)
        import glob
        all_files = sorted(glob.glob(os.path.join(job['data_dir'], '*.csv')))

        for H in job['horizons']:
            model = load_model(job['exp'], H, n_out, dev)
            for gname, gval in ABS_GROUPS.items():
                best = None
                for path in all_files:
                    df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    if 'abs' not in df.columns or abs(float(df['abs'].iloc[0]) - gval) > 1e-6:
                        continue
                    cols = tc_cols(df, ns)
                    if len(cols) != ns:
                        continue
                    last_start = len(df) - SEQ_LEN - H
                    if last_start <= 0:
                        continue
                    offs = np.linspace(0, last_start, min(MAX_WINDOWS, last_start + 1), dtype=int)
                    for off in offs:
                        win = df.iloc[off:off + SEQ_LEN]
                        tcw = win[cols].values.astype(np.float64)
                        temp_in = tcw if is_raw else 0.5 * (tcw[:, :-1] + tcw[:, 1:])
                        ts_np = np.hstack([((win['Time'].values - TIME_MEAN) / TIME_STD).reshape(-1, 1),
                                           (temp_in - tsc.mean_) / tsc.scale_])
                        sp = psc.transform([[float(win[c].iloc[0]) for c in ['h', 'flux', 'abs', 'surf']]])
                        with torch.no_grad():
                            out = model([torch.from_numpy(ts_np).unsqueeze(0).float().to(dev),
                                         torch.from_numpy(sp).float().to(dev)]).cpu().numpy()[0]
                        out = out * tsc.scale_ + tsc.mean_
                        tgt = off + SEQ_LEN - 1 + H
                        act = df[cols].iloc[tgt].values.astype(np.float64)
                        pred = out if is_raw else (act*0 + out[:ns])  # raw path only here
                        mae = float(np.mean(np.abs(pred - act)))
                        if best is None or mae < best['mae']:
                            best = dict(file=os.path.basename(path), off=int(off),
                                        t_target=float(df['Time'].iloc[tgt]), mae=mae,
                                        rmse=float(np.sqrt(np.mean((pred-act)**2))),
                                        act=act.copy(), pred=pred.copy())
                if best:
                    best.update(ds=job['ds'], H=H, group=gname, ns=ns)
                    results.append(best)
                    print(f"\n[{job['ds']} H={H} {gname}]  BEST: MAE={best['mae']:.3f} K  "
                          f"t={best['t_target']:.0f}s  {best['file'][:44]}")
                    print('   actual:', ' '.join(f'{x:6.1f}' for x in best['act']))
                    print('   pred  :', ' '.join(f'{x:6.1f}' for x in best['pred']))
                else:
                    print(f"\n[{job['ds']} H={H} {gname}]  no file long enough")

    # save
    with open('output/best_predictions.csv', 'w', newline='') as f:
        w = csv.writer(f)
        maxn = max(r['ns'] for r in results)
        w.writerow(['dataset','H','abs_group','file','window_start_row','t_target_s','MAE_K','RMSE_K']
                   + [f'TC{i+1}_actual' for i in range(maxn)] + [f'TC{i+1}_pred' for i in range(maxn)])
        for r in results:
            row = [r['ds'], r['H'], r['group'], r['file'], r['off'], r['t_target'], round(r['mae'],3), round(r['rmse'],3)]
            row += [round(x,2) for x in r['act']] + ['']*(maxn-r['ns'])
            row += [round(x,2) for x in r['pred']] + ['']*(maxn-r['ns'])
            w.writerow(row)
    print(f"\nsaved -> output/best_predictions.csv  ({len(results)} best windows)")


if __name__ == '__main__':
    main()
