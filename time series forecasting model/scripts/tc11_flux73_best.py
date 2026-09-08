"""Best TC11 predicted-vs-actual for flux73_abs0_surf0, per horizon (60..600).
Searches ALL windows across the 3 flux73_abs0_surf0 files and reports the lowest-MAE
(best-case) window per horizon. TC11 raw 11-sensor model; scaler re-fit per horizon with
the model's train/test split (these flux73 files are in TC11 training -> in-sample best case).
"""
import warnings; warnings.filterwarnings('ignore')
import contextlib, io, os, sys, glob, re, csv
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from new_model import PhysicsInformedLSTM             # noqa: E402
from new_dataset_builder import TempSequenceDataset   # noqa: E402

DATA = os.path.join(ROOT, 'data', 'output_with_TC11')
EXP_ROOT = os.path.join(ROOT, 'output'); EXP = 'new_theoretical_TC11'
NS, SEQ, TM, TS = 11, 20, 300.0, 300.0
PARAM_COLS = ['h', 'flux', 'abs', 'surf']
HORIZONS = [60, 180, 300, 480, 600]
TC11_TEST = ['h6_flux88_abs20_surf0_781s - Sheet2.csv', 'h6_flux88_abs92_surf0_648s - Sheet3.csv',
             'h6_flux88_abs0_surf1_790s - Sheet1.csv', 'h6_flux88_abs0_surf0_longRun_762s - Sheet1.csv']


def tc_cols(df):
    return sorted([c for c in df.columns if 'TC' in c.upper()], key=lambda c: int(re.findall(r'\d+', c)[0]))[:NS]


def get_scalers(H):
    with contextlib.redirect_stdout(io.StringIO()):
        ds = TempSequenceDataset(data_dir=DATA, scaler_dir='/tmp/refit_tc11', num_sensors=NS, zscore_threshold=1e9,
                                 prediction_horizon=H, target_test_files=TC11_TEST, bin_target=False, split='train')
    return ds.thermal_scaler, ds.param_scaler


def load_model(H):
    ck = os.path.join(EXP_ROOT, f'{EXP}_L{SEQ}_H{H}', f'best_model_L{SEQ}_H{H}.pth')
    m = PhysicsInformedLSTM(num_sensors=NS, sequence_length=SEQ, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(torch.load(ck, map_location='cpu')); m.eval(); return m


def main():
    files = sorted(glob.glob(os.path.join(DATA, '*flux73*abs0*surf0*.csv')))
    results = []
    for H in HORIZONS:
        tsc, psc = get_scalers(H); model = load_model(H)
        sc_t, mn_t = tsc.scale_, tsc.mean_
        best = None
        for path in files:
            df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            cols = tc_cols(df)
            if len(cols) != NS or len(df) < SEQ + H + 1:
                continue
            tcv = df[cols].values.astype(np.float64); tv = df['Time'].values.astype(np.float64)
            sp = psc.transform([[float(df[c].iloc[0]) for c in PARAM_COLS]])
            nwin = len(df) - SEQ - H + 1
            TSb, tgts, ttimes = [], [], []
            for off in range(nwin):
                wtc = tcv[off:off + SEQ]; wt = tv[off:off + SEQ]
                TSb.append(np.hstack([((wt - TM) / TS).reshape(-1, 1), (wtc - mn_t) / sc_t]))
                ti = off + SEQ - 1 + H; tgts.append(tcv[ti]); ttimes.append(tv[ti])
            TSb = torch.tensor(np.array(TSb)).float()
            SPb = torch.tensor(np.repeat(sp, nwin, axis=0)).float()
            with torch.no_grad():
                out = model([TSb, SPb]).numpy()
            pred = out * sc_t + mn_t; tgts = np.array(tgts)
            maes = np.abs(pred - tgts).mean(1); bi = int(np.argmin(maes))
            if best is None or maes[bi] < best['mae']:
                best = dict(H=H, mae=float(maes[bi]), rmse=float(np.sqrt(((pred[bi] - tgts[bi]) ** 2).mean())),
                            file=os.path.basename(path), t_target=float(ttimes[bi]),
                            actual=tgts[bi], pred=pred[bi])
        results.append(best)

    # print in order
    for r in results:
        print(f"\n=== H={r['H']}  best window  (MAE={r['mae']:.3f} K, RMSE={r['rmse']:.3f} K)  ===")
        print(f"    file: {r['file']}   forecast for t={r['t_target']:.0f}s")
        print('    TC#    ' + ''.join(f'{i+1:>7}' for i in range(NS)))
        print('    actual ' + ''.join(f'{a:>7.1f}' for a in r['actual']))
        print('    pred   ' + ''.join(f'{p:>7.1f}' for p in r['pred']))
        print('    |err|  ' + ''.join(f'{abs(a-p):>7.1f}' for a, p in zip(r['actual'], r['pred'])))
    # csv
    out = os.path.join(ROOT, 'output', 'tc11_flux73_abs0_surf0_best_pred_vs_actual.csv')
    hdr = ['H', 'file', 't_target', 'MAE_K', 'RMSE_K'] + [f'TC{i+1}_actual' for i in range(NS)] + [f'TC{i+1}_pred' for i in range(NS)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in results:
            w.writerow([r['H'], r['file'], round(r['t_target'], 1), round(r['mae'], 3), round(r['rmse'], 3)]
                       + [round(float(x), 2) for x in r['actual']] + [round(float(x), 2) for x in r['pred']])
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
