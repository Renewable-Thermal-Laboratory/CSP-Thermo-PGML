"""Best REAL flux73 results: TC10 models on all flux73 files across trained horizons
(60..600; H=900 impossible — no flux73 run reaches 15 min). Aggregate MAE over all
valid windows. flux73 files were all in TC10 training, so these are in-sample numbers.
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib, os, sys, glob
import numpy as np, torch, pandas as pd, joblib
import torch.utils.data as tud

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from new_dataset_builder import TempSequenceDataset, collate_fn  # noqa: E402
from new_model import PhysicsInformedLSTM                        # noqa: E402

DATA = os.path.join(ROOT, 'data', 'processed_H6')
NS, SEQ = 10, 20
HS = [60, 180, 300, 480, 600]


def model_and_scaler(H):
    d = os.path.join(ROOT, 'output', f'new_theoretical_TC10_L20_H{H}')
    tsc = joblib.load(os.path.join(d, 'thermal_scaler.save'))
    m = PhysicsInformedLSTM(num_sensors=NS, sequence_length=SEQ, lstm_units=512, dropout_rate=0.2,
                            residual_prediction=True, baseline_mode='persistence', horizon_steps=H)
    m.load_state_dict(torch.load(os.path.join(d, f'best_model_L20_H{H}.pth'), map_location='cpu')); m.eval()
    return m, tsc, d


def agg_mae(fn, H, m, sc, mn, mdir):
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
    return torch.abs(P - T).mean().item(), len(P)


def main():
    files = sorted(glob.glob(os.path.join(DATA, '*flux73*.csv')))
    short = {os.path.basename(f): len(pd.read_csv(f).apply(pd.to_numeric, errors='coerce').dropna()) for f in files}
    grid = {}  # (fn,H)->mae
    for H in HS:
        m, tsc, mdir = model_and_scaler(H)
        sc = torch.tensor(tsc.scale_).float(); mn = torch.tensor(tsc.mean_).float()
        for f in files:
            fn = os.path.basename(f)
            if short[fn] < SEQ + H + 1:
                continue
            grid[(fn, H)] = agg_mae(fn, H, m, sc, mn, mdir)

    short_name = lambda fn: fn.replace('h6_flux73_', '').replace(' - Sheet', ' S').replace('.csv', '')
    print(f"\nflux73 MAE (K) — TC10 models, in-sample. '—' = file too short for that horizon\n")
    print(f"{'file':28} " + ''.join(f'H{H:>5}' for H in HS))
    print('-' * 64)
    for f in files:
        fn = os.path.basename(f)
        cells = ''
        for H in HS:
            cells += f'{grid[(fn,H)][0]:>6.2f}' if (fn, H) in grid else f'{"—":>6}'
        print(f"{short_name(fn)[:28]:28} {cells}")
    # best
    best = min(grid.items(), key=lambda kv: kv[1][0])
    (bfn, bH), (bmae, bn) = best
    print(f"\nBEST flux73 result: {short_name(bfn)}  H={bH}  MAE={bmae:.2f} K  ({bn} windows)")
    print(f"Ceiling horizon = H=600 (data ends at ~14 min; H=900 needs 15.4 min -> impossible)")


if __name__ == '__main__':
    main()
