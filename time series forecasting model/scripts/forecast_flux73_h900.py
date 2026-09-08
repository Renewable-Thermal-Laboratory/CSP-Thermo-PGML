"""H=900 FORECAST (no ground truth) for every flux73_abs0 file, using the TC10 H=900 model.
The runs end before t+900s, so there is nothing to score against — these are pure model
forecasts. Input = the last 20 rows of each file; target time = file_end + 900s.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, glob, csv, re
import numpy as np, torch, pandas as pd, joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from new_model import PhysicsInformedLSTM  # noqa: E402

SEQ, TM, TS, NS, H = 20, 300.0, 300.0, 10, 900
PARAM_COLS = ['h', 'flux', 'abs', 'surf']
DATA = os.path.join(ROOT, 'data', 'processed_H6')


def tc_cols(df):
    return sorted([c for c in df.columns if 'TC' in c.upper()], key=lambda c: int(re.findall(r'\d+', c)[0]))[:NS]


def main():
    d = os.path.join(ROOT, 'output', f'new_theoretical_TC10_L20_H{H}')
    tsc = joblib.load(os.path.join(d, 'thermal_scaler.save'))
    psc = joblib.load(os.path.join(d, 'param_scaler.save'))
    m = PhysicsInformedLSTM(num_sensors=NS, sequence_length=SEQ, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(torch.load(os.path.join(d, f'best_model_L20_H{H}.pth'), map_location='cpu')); m.eval()

    files = sorted(glob.glob(os.path.join(DATA, '*flux73*abs0*.csv')))
    rows = []
    for path in files:
        fn = os.path.basename(path)
        df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        cols = tc_cols(df)
        win = df.iloc[-SEQ:]
        tcw = win[cols].values.astype(np.float64)
        ts_np = np.hstack([((win['Time'].values - TM) / TS).reshape(-1, 1), (tcw - tsc.mean_) / tsc.scale_])
        sp = psc.transform([[float(win[c].iloc[0]) for c in PARAM_COLS]])
        with torch.no_grad():
            o = m([torch.from_numpy(ts_np).unsqueeze(0).float(), torch.from_numpy(sp).float()]).numpy()[0]
        pred = o * tsc.scale_ + tsc.mean_
        last = tcw[-1]
        t_end = float(win['Time'].iloc[-1]); t_tgt = t_end + H
        row = dict(file=fn, t_input_end_s=t_end, t_forecast_s=t_tgt)
        for i in range(NS): row[f'TC{i+1}_lastobs'] = round(float(last[i]), 2)
        for i in range(NS): row[f'TC{i+1}_forecast'] = round(float(pred[i]), 2)
        rows.append(row)
        print(f"\n=== {fn} ===")
        print(f"    input ends t={t_end:.0f}s  ->  forecast for t={t_tgt:.0f}s (+900s, beyond data end)")
        print('    TC#       ' + ''.join(f'{i+1:>8}' for i in range(NS)))
        print('    last obs  ' + ''.join(f'{v:>8.1f}' for v in last))
        print('    FORECAST  ' + ''.join(f'{v:>8.1f}' for v in pred))

    out = os.path.join(ROOT, 'output', 'forecast_flux73_abs0_H900.csv')
    hdr = ['file', 't_input_end_s', 't_forecast_s'] + [f'TC{i+1}_lastobs' for i in range(NS)] + [f'TC{i+1}_forecast' for i in range(NS)]
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {len(rows)} forecasts -> {out}")


if __name__ == '__main__':
    main()
