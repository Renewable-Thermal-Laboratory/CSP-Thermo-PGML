"""Out-of-fold predictions (leave-one-run-out), capped to ~400 pts/run for speed."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
SRC = Path("/Users/bhuvan/Desktop/research/ml_models/ir_profile_model/src")
sys.path.insert(0, str(SRC))
from train_final import CHANNELS, BASE, ZNORM, BAD_ALIGN
from mlp_model import MLPModel
SP = "/private/tmp/claude-501/-Users-bhuvan-Desktop-research-ml-models-time-series-forecasting-model/0087db2c-6d8b-477f-91b4-12c8cb8b30b6/scratchpad"

df = pd.read_csv(SRC.parent / "data" / "dataset.csv")
df = df[~df["run_key"].isin(BAD_ALIGN)].reset_index(drop=True)
# cap each run to <=400 evenly spaced seconds (adjacent seconds ~ duplicates)
CAP = 400
keep = []
for r, g in df.groupby("run_key"):
    idx = g.index.values
    keep.extend(idx if len(idx) <= CAP else idx[np.linspace(0, len(idx)-1, CAP).astype(int)])
df = df.loc[sorted(keep)].reset_index(drop=True)

def longfmt(d):
    Xb = d[BASE].values; run = d["run_key"].values; t = d["t"].values
    xs=ys=None; xs,ys,rs,cs,ts,zs = [],[],[],[],[],[]
    for ch, z in CHANNELS.items():
        m = d[ch].notna().values
        if not m.any(): continue
        xs.append(np.column_stack([Xb[m], np.full(m.sum(), ZNORM(z))]))
        ys.append(d.loc[m, ch].values - d.loc[m, "TC1"].values)
        rs.append(run[m]); cs.append(np.full(m.sum(), ch)); ts.append(t[m]); zs.append(np.full(m.sum(), z))
    return (np.vstack(xs), np.concatenate(ys), np.concatenate(rs),
            np.concatenate(cs), np.concatenate(ts), np.concatenate(zs))

X, y, run, chan, tt, zz = longfmt(df)
tc1 = df.set_index(["run_key","t"])["TC1"]
rows=[]
for i, r in enumerate(np.unique(run), 1):
    tr, te = run != r, run == r
    m = MLPModel().fit(X[tr], y[tr])
    pred = m.predict(X[te])
    rows.append(pd.DataFrame({"run":run[te],"t":tt[te],"chan":chan[te],"z":zz[te],
                              "y_true_off":y[te],"y_pred_off":pred}))
    print(f"[{i}/{len(np.unique(run))}] {r}  MAE={np.abs(pred-y[te]).mean():.2f}", flush=True)
oof = pd.concat(rows, ignore_index=True)
oof["TC1"] = [tc1.loc[(r,t)] for r,t in zip(oof["run"],oof["t"])]
oof["T_true"] = oof["y_true_off"]+oof["TC1"]; oof["T_pred"]=oof["y_pred_off"]+oof["TC1"]
oof.to_csv(f"{SP}/oof.csv", index=False)
print("SAVED", len(oof))
