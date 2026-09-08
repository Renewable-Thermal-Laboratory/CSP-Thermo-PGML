"""Run the TC10 (10-sensor) models on the TC11 test files, at H=600 and H=900.

TC11 only trained up to H=600; TC10 has H=900 — so to see H=900 on these specific
files you must use the TC10 model. Uses each TC10 horizon's MATCHED scaler.

Outputs: printed summary (aggregate MAE over all valid windows) + a CSV with one
representative window's predicted-vs-actual TC1..TC10 per (file, horizon).
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib, os, sys, csv
import numpy as np, torch, pandas as pd, joblib
import torch.utils.data as tud

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from new_dataset_builder import TempSequenceDataset, collate_fn  # noqa: E402
from new_model import PhysicsInformedLSTM                        # noqa: E402

DATA = os.path.join(ROOT, 'data', 'processed_H6')
NS, SEQ = 10, 20
HS = [600, 900]

# (label, filename, provenance note)  — the TC11 abs0/abs92 test files
FILES = [
    ('abs0_surf1_790s',  'h6_flux88_abs0_surf1_790s - Sheet1.csv',         'TC11 test & TC10 test (held-out)'),
    ('abs0_surf0_762s',  'h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv', 'TC11 test; TC10 TRAIN (in-sample!)'),
    ('abs92_surf0_648s', 'h6_flux88_abs92_surf0_648s - Sheet3.csv',        'TC11 test (short)'),
    ('abs92_surf0_618s', 'h6_flux88_abs92_surf0_longRun_618s - Sheet1.csv','TC10 test (held-out) - abs92 STAND-IN'),
]


def model_and_scaler(H):
    d = os.path.join(ROOT, 'output', f'new_theoretical_TC10_L20_H{H}')
    tsc = joblib.load(os.path.join(d, 'thermal_scaler.save'))
    m = PhysicsInformedLSTM(num_sensors=NS, sequence_length=SEQ, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(torch.load(os.path.join(d, f'best_model_L20_H{H}.pth'), map_location='cpu')); m.eval()
    return m, tsc, d


def main():
    rows = []
    print(f"\n{'H':>4} {'file':16} {'nwin':>5} {'aggMAE':>8} {'aggRMSE':>8}   note")
    print("-" * 78)
    for H in HS:
        m, tsc, mdir = model_and_scaler(H)
        sc = torch.tensor(tsc.scale_).float(); mn = torch.tensor(tsc.mean_).float()
        for tag, fn, note in FILES:
            path = os.path.join(DATA, fn)
            if not os.path.exists(path):
                print(f"{H:>4} {tag:16} {'-':>5}   MISSING FILE"); continue
            nrows = len(pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna())
            if nrows < SEQ + H + 1:
                print(f"{H:>4} {tag:16} {'--':>5}   TOO SHORT ({nrows} rows, need {SEQ+H+1})   {note}"); continue
            with contextlib.redirect_stdout(io.StringIO()):
                ds = TempSequenceDataset(data_dir=DATA, scaler_dir=mdir, split='test', prediction_horizon=H,
                                         num_sensors=NS, zscore_threshold=1e9, target_test_files=[fn], bin_target=False)
                ds.load_pretrained_scalers(mdir)
            ld = tud.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
            P, T = [], []
            with torch.no_grad():
                for ts_, sp_, tg_, _ in ld:
                    P.append(m([ts_, sp_]) * sc + mn); T.append(tg_ * sc + mn)
            P = torch.cat(P); T = torch.cat(T)
            mae = torch.abs(P - T).mean().item(); rmse = torch.sqrt(((P - T) ** 2).mean()).item(); n = len(P)
            print(f"{H:>4} {tag:16} {n:>5} {mae:>8.2f} {rmse:>8.2f}   {note}")
            # representative window = middle of file
            df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            si = ds.sample_indices; ri = len(si) // 2
            _, start = si[ri]; tgt_idx = start + SEQ - 1 + H
            ttime = float(df['Time'].iloc[tgt_idx]) if 'Time' in df.columns else -1.0
            pv = P[ri].numpy(); av = T[ri].numpy()
            row = dict(file=tag, condition_file=fn, H=H, provenance=note, n_windows=n,
                       agg_MAE_K=round(mae, 3), agg_RMSE_K=round(rmse, 3),
                       rep_target_time_s=ttime, rep_window_MAE_K=round(float(np.abs(pv - av).mean()), 3))
            for i in range(NS): row[f'TC{i+1}_actual'] = round(float(av[i]), 2)
            for i in range(NS): row[f'TC{i+1}_pred'] = round(float(pv[i]), 2)
            rows.append(row)
    out = os.path.join(ROOT, 'output', 'TC10models_on_TC11files_H600_H900.csv')
    if rows:
        hdr = (['file', 'condition_file', 'H', 'provenance', 'n_windows', 'agg_MAE_K', 'agg_RMSE_K',
                'rep_target_time_s', 'rep_window_MAE_K']
               + [f'TC{i+1}_actual' for i in range(NS)] + [f'TC{i+1}_pred' for i in range(NS)])
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {len(rows)} representative rows -> {out}")


if __name__ == '__main__':
    main()
