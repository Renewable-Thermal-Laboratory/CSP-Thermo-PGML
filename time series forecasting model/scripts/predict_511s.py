"""Predicted-vs-actual for one TC11 file across all horizons, using the CORRECT
per-horizon scaler (re-fit fresh, not the possibly-stale saved one).

For h6_flux73_abs0_surf0_511s, for each H in {60,180,300,480,600}, scans every input
window and reports the best (lowest-MAE) one: input time span, target time, actual vs
predicted TC1..TC11.
"""
import io, contextlib, os, re, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from new_dataset_builder import TempSequenceDataset  # noqa: E402
from new_model import PhysicsInformedLSTM            # noqa: E402

FILE = sys.argv[1] if len(sys.argv) > 1 else 'h6_flux73_abs0_surf0_511s - Sheet3.csv'
DATA_DIR = 'data/output_with_TC11'
SCALER_DIR = 'models_TC11'
NS, NOUT = 11, 11
HORIZONS = [60, 180, 300, 480, 600]
TC11_TEST = ['h6_flux88_abs20_surf0_781s - Sheet2.csv', 'h6_flux88_abs92_surf0_648s - Sheet3.csv',
             'h6_flux88_abs0_surf1_790s - Sheet1.csv', 'h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv']
SEQ_LEN, TM, TS = 20, 300.0, 300.0


def main():
    dev = torch.device('cpu')
    df = pd.read_csv(os.path.join(DATA_DIR, FILE)).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    cols = sorted([c for c in df.columns if 'TC' in c.upper()], key=lambda c: int(re.findall(r'\d+', c)[0]))[:NS]
    X = df[cols].values.astype(np.float64); T = df['Time'].values.astype(np.float64)

    print(f"File: {FILE}   rows={len(df)}")
    out_rows = []
    for H in HORIZONS:
        if len(df) - SEQ_LEN - H <= 0:
            print(f"\n=== H={H}: file too short ({len(df)} rows, need >{SEQ_LEN+H}) — SKIPPED ===")
            continue
        # Re-fit the scaler exactly as the H-experiment did (train split fits scalers).
        with contextlib.redirect_stdout(io.StringIO()):
            ds = TempSequenceDataset(data_dir=DATA_DIR, scaler_dir='/tmp/sc_tc11', num_sensors=NS,
                                     zscore_threshold=1e9, prediction_horizon=H,
                                     target_test_files=TC11_TEST, split='train')
        tsc, psc = ds.thermal_scaler, ds.param_scaler
        st = torch.load(f"output/new_theoretical_TC11_L20_H{H}/best_model_L20_H{H}.pth", map_location=dev)
        m = PhysicsInformedLSTM(num_sensors=NOUT, sequence_length=SEQ_LEN, lstm_units=512, dropout_rate=0.2,
                                residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
        m.load_state_dict(st); m.eval()
        sp = psc.transform([[float(df[c].iloc[0]) for c in ['h', 'flux', 'abs', 'surf']]])
        last = len(df) - SEQ_LEN - H
        best = None
        for off in range(0, last):
            tcw = X[off:off + SEQ_LEN]
            ts_np = np.hstack([((T[off:off + SEQ_LEN] - TM) / TS).reshape(-1, 1), (tcw - tsc.mean_) / tsc.scale_])
            with torch.no_grad():
                o = m([torch.from_numpy(ts_np).unsqueeze(0).float(), torch.from_numpy(sp).float()]).numpy()[0]
            o = o * tsc.scale_ + tsc.mean_
            tgt = off + SEQ_LEN - 1 + H
            act = X[tgt]
            mae = float(np.mean(np.abs(o - act)))
            if best is None or mae < best['mae']:
                best = dict(off=off, t_in0=T[off], t_in1=T[off + SEQ_LEN - 1], t_tgt=T[tgt],
                            mae=mae, rmse=float(np.sqrt(np.mean((o - act) ** 2))), act=act.copy(), pred=o.copy())
        print(f"\n=== H={H}  (input {best['t_in0']:.0f}-{best['t_in1']:.0f}s -> predict t={best['t_tgt']:.0f}s)  MAE={best['mae']:.3f} K ===")
        print('  actual:', ' '.join(f'{x:6.1f}' for x in best['act']))
        print('  pred  :', ' '.join(f'{x:6.1f}' for x in best['pred']))
        best['H'] = H; out_rows.append(best)

    import csv
    tag = re.sub(r'[^a-zA-Z0-9]+', '_', FILE.split(' - ')[0])
    out_csv = f'output/predict_{tag}_TC11.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['H', 't_input_start', 't_input_end', 't_target', 'MAE_K', 'RMSE_K']
                   + [f'TC{i+1}_actual' for i in range(NS)] + [f'TC{i+1}_pred' for i in range(NS)])
        for r in out_rows:
            w.writerow([r['H'], r['t_in0'], r['t_in1'], r['t_tgt'], round(r['mae'], 3), round(r['rmse'], 3)]
                       + [round(x, 2) for x in r['act']] + [round(x, 2) for x in r['pred']])
    print(f'\nsaved -> {out_csv}')


if __name__ == '__main__':
    main()
