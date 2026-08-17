"""Best TC10 (TC1-TC10) pred-vs-actual on the NEW flux73 runs, input window at offset 600,
horizons 60/180/300/480/600/900. Best available model per horizon: canonical for 60-600,
flux73_h900_BEST (seed 123) for 900. Runs on CPU so it won't disturb a running MPS train.
NOTE: the new files have no 11th sensor, so this is TC10, not TC11.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, re, csv
import numpy as np, torch, pandas as pd, joblib

ROOT = "/Users/bhuvan/Desktop/research/ml_models/time series forecasting model"
sys.path.insert(0, os.path.join(ROOT, 'src'))
from new_model import PhysicsInformedLSTM  # noqa: E402

SEQ, TM, TS, NS, OFF = 20, 300.0, 300.0, 10, 600
PARAM_COLS = ['h', 'flux', 'abs', 'surf']
DATA = os.path.join(ROOT, 'data', 'processed_H6')
HS = [60, 180, 300, 480, 600, 900]
FILES = ["h6_flux73_abs0_surf0_longRun_1658s - Sheet1.csv",
         "h6_flux73_abs0_surf0_longRun2_1309s - Sheet1.csv",
         "h6_flux73_abs0_surf0_longRun3_1376s - Sheet1.csv",
         "h6_flux73_abs0_surf0_longRun4_1224s - Sheet1.csv"]


def model_dir(H):
    if H == 900:
        return os.path.join(ROOT, 'src', 'output', 'flux73_h900_BEST_L20_H900')
    return os.path.join(ROOT, 'output', f'new_theoretical_TC10_L20_H{H}')


def tc_cols(df):
    return sorted([c for c in df.columns if 'TC' in c.upper()], key=lambda c: int(re.findall(r'\d+', c)[0]))[:NS]


def main():
    dev = torch.device('cpu')
    rows = []
    for H in HS:
        d = model_dir(H)
        tsc = joblib.load(os.path.join(d, 'thermal_scaler.save'))
        psc = joblib.load(os.path.join(d, 'param_scaler.save'))
        m = PhysicsInformedLSTM(num_sensors=NS, sequence_length=SEQ, lstm_units=512, dropout_rate=0.2,
                                residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
        m.load_state_dict(torch.load(os.path.join(d, f'best_model_L20_H{H}.pth'), map_location='cpu')); m.to(dev).eval()
        for fn in FILES:
            df = pd.read_csv(os.path.join(DATA, fn)).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            cols = tc_cols(df)
            if len(df) < OFF + SEQ + H + 1:
                continue
            win = df.iloc[OFF:OFF + SEQ]
            tcw = win[cols].values.astype(np.float64)
            ts_np = np.hstack([((win['Time'].values - TM) / TS).reshape(-1, 1), (tcw - tsc.mean_) / tsc.scale_])
            sp = psc.transform([[float(win[c].iloc[0]) for c in PARAM_COLS]])
            with torch.no_grad():
                o = m([torch.from_numpy(ts_np).unsqueeze(0).float(), torch.from_numpy(sp).float()]).numpy()[0]
            pred = o * tsc.scale_ + tsc.mean_
            seq_end = OFF + SEQ - 1; tgt = seq_end + H
            act = df[cols].iloc[tgt].values.astype(np.float64)
            mae = float(np.mean(np.abs(pred - act))); rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
            row = dict(file=fn.replace('h6_flux73_abs0_surf0_', '').replace(' - Sheet1.csv', ''), H=H,
                       t_target=float(df['Time'].iloc[tgt]), MAE_K=round(mae, 3), RMSE_K=round(rmse, 3),
                       model='flux73_BEST(s123)' if H == 900 else 'canonical')
            for i in range(NS): row[f'TC{i+1}_actual'] = round(float(act[i]), 2)
            for i in range(NS): row[f'TC{i+1}_pred'] = round(float(pred[i]), 2)
            rows.append(row)

    out = os.path.join(ROOT, 'output', 'flux73_newruns_TC10_offset600_pred_vs_actual.csv')
    hdr = (['file', 'H', 't_target', 'MAE_K', 'RMSE_K', 'model']
           + [f'TC{i+1}_actual' for i in range(NS)] + [f'TC{i+1}_pred' for i in range(NS)])
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for r in rows: w.writerow(r)

    print("TC10 pred-vs-actual @ offset 600 (input rows 600-619), new flux73 runs\n")
    print(f"{'file':18} {'H':>4} {'t_tgt':>6} {'MAE':>7} {'RMSE':>7}  model")
    for r in rows:
        print(f"{r['file']:18} {r['H']:>4} {r['t_target']:>6.0f} {r['MAE_K']:>7.2f} {r['RMSE_K']:>7.2f}  {r['model']}")
    print(f"\nfull TC1-10 values -> {out}")
    # full values for run1 (the only file that reaches H=900)
    print("\n=== run1 (longRun_1658) — actual vs pred, all horizons ===")
    for r in rows:
        if not r['file'].startswith('longRun_1658'):
            continue
        a = [r[f'TC{i+1}_actual'] for i in range(NS)]; p = [r[f'TC{i+1}_pred'] for i in range(NS)]
        print(f"\n H={r['H']} (target t={r['t_target']:.0f}s, MAE={r['MAE_K']} K)")
        print('  actual ' + ''.join(f'{x:>7.1f}' for x in a))
        print('  pred   ' + ''.join(f'{x:>7.1f}' for x in p))


if __name__ == '__main__':
    main()
