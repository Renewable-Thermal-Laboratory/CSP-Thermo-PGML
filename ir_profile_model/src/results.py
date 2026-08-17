"""Journal-ready results + figures from out-of-fold predictions."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SP = "/private/tmp/claude-501/-Users-bhuvan-Desktop-research-ml-models-time-series-forecasting-model/0087db2c-6d8b-477f-91b4-12c8cb8b30b6/scratchpad"
OUTDIR = Path("/Users/bhuvan/Desktop/research/ml_models/ir_profile_model/results")
OUTDIR.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 120,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, RUST, GRAY = "#20567C", "#C1440E", "#6b6b6b"
ORDER = ["TC2","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC9.5","TC10"]
ZMAP = {"TC2":2,"TC3":3,"TC4":4,"TC5":5,"TC6":6,"TC7":7,"TC8":8,"TC9":9,"TC9.5":9.5,"TC10":10}

oof = pd.read_csv(f"{SP}/oof.csv")
oof["err"] = oof["T_pred"] - oof["T_true"]
oof["ae"] = oof["err"].abs()
oof["absr"] = oof["run"].str.extract(r"abs(\d+)").astype(int)
oof["cond"] = oof["run"].str.replace(r"_run\d+|_longrun\d+","",regex=True)

t, p = oof["T_true"].values, oof["T_pred"].values
r2 = 1 - np.sum((t-p)**2)/np.sum((t-t.mean())**2)
lines = []
def L(s=""): lines.append(s); print(s)

L("="*66)
L("JOURNAL RESULTS — IR->profile model (leave-one-run-out, out-of-fold)")
L("="*66)
L(f"\nDataset: {oof['run'].nunique()} runs, {len(oof):,} evaluated (channel,second) samples")
L("\n-- Overall predictive accuracy --")
L(f"  MAE   = {oof['ae'].mean():.2f} C")
L(f"  RMSE  = {np.sqrt((oof['err']**2).mean()):.2f} C")
L(f"  MedAE = {oof['ae'].median():.2f} C")
L(f"  Bias  = {oof['err'].mean():+.2f} C   (systematic offset)")
L(f"  R^2   = {r2:.4f}")
L(f"  Pearson r = {np.corrcoef(t,p)[0,1]:.4f}")
L(f"  |err|<=1C: {(oof['ae']<=1).mean()*100:.0f}%   <=2C: {(oof['ae']<=2).mean()*100:.0f}%   <=3C: {(oof['ae']<=3).mean()*100:.0f}%")

L("\n-- Per-channel (bottom -> surface) --")
L(f"  {'chan':6s}{'MAE':>7}{'RMSE':>7}{'bias':>8}{'p90|e|':>8}")
for ch in ORDER:
    g = oof[oof.chan==ch]
    if len(g)==0: continue
    L(f"  {ch:6s}{g.ae.mean():7.2f}{np.sqrt((g.err**2).mean()):7.2f}{g.err.mean():+8.2f}{g.ae.quantile(.9):8.2f}")

L("\n-- Per-absorber regime --")
for a in [0,20,92]:
    g = oof[oof.absr==a]
    L(f"  abs{a:<3d}: MAE {g.ae.mean():.2f} C   RMSE {np.sqrt((g.err**2).mean()):.2f}   ({g['run'].nunique()} runs)")

L("\n-- Easiest / hardest runs --")
rm = oof.groupby("run")["ae"].mean().sort_values()
for r,v in list(rm.items())[:3]: L(f"  best  {r.replace('h6_',''):40s}{v:.2f} C")
for r,v in list(rm.items())[-3:]: L(f"  worst {r.replace('h6_',''):40s}{v:.2f} C")

(OUTDIR/"results_summary.txt").write_text("\n".join(lines))

# LaTeX per-channel table
with open(OUTDIR/"table_perchannel.tex","w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\hline\nChannel & MAE & RMSE & Bias & P90 \\\\\n\\hline\n")
    for ch in ORDER:
        g=oof[oof.chan==ch]
        if len(g)==0: continue
        f.write(f"{ch} & {g.ae.mean():.2f} & {np.sqrt((g.err**2).mean()):.2f} & {g.err.mean():+.2f} & {g.ae.quantile(.9):.2f} \\\\\n")
    f.write(f"\\hline\n\\textbf{{Overall}} & {oof.ae.mean():.2f} & {np.sqrt((oof.err**2).mean()):.2f} & {oof.err.mean():+.2f} & {oof.ae.quantile(.9):.2f} \\\\\n\\hline\n\\end{{tabular}}\n")

# ========== FIG 1: predicted vs actual profiles (9 conditions) ==========
reps = [oof[oof.cond==c]["run"].iloc[0] for c in sorted(oof["cond"].unique())]
fig, axes = plt.subplots(3,3, figsize=(12,12))
for ax, run in zip(axes.flat, reps):
    g = oof[oof.run==run]; ts=sorted(g["t"].unique()); tmid=ts[len(ts)//2]; gm=g[g.t==tmid].sort_values("z")
    tc1=gm["TC1"].iloc[0]
    zs=[1]+list(gm["z"]); tr=[tc1]+list(gm["T_true"]); pr=[tc1]+list(gm["T_pred"])
    ax.plot(tr, zs, "o-", color=BLUE, label="measured", lw=1.8, ms=6)
    ax.plot(pr, zs, "s--", color=RUST, label="predicted", lw=1.8, ms=5, mfc="none")
    ax.set_title(run.replace("h6_",""), fontsize=9.5)   # TC1 bottom, TC10 surface at top
    ax.set_xlabel("Temperature (\u00b0C)"); ax.set_ylabel("Rake position (TC1 bottom \u2192 TC10 surface)")
    ax.text(.04,.04,f"t={tmid}s  MAE={g[g.t==tmid].ae.mean():.1f}\u00b0C",
            transform=ax.transAxes, fontsize=8.5, color=GRAY,
            bbox=dict(fc="white",ec="none",alpha=.7,pad=1.5))
axes.flat[0].legend(fontsize=9, loc="upper left")
fig.suptitle("Predicted vs. measured vertical temperature profiles (held-out runs)", fontsize=13, y=1.005)
fig.tight_layout(); fig.savefig(OUTDIR/"fig1_profiles.png"); plt.close(fig)

# ========== FIG 2: calibration + error-by-depth ==========
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.5))
s=oof.sample(min(9000,len(oof)),random_state=0)
sc=a1.scatter(s.T_true,s.T_pred,s=5,c=s.z,cmap="viridis",alpha=.5,ec="none")
lim=[min(t.min(),p.min())-3,max(t.max(),p.max())+3]
a1.plot(lim,lim,"k--",lw=1.2,zorder=5); a1.set_xlim(lim); a1.set_ylim(lim)
a1.set_xlabel("Measured T (\u00b0C)"); a1.set_ylabel("Predicted T (\u00b0C)")
a1.set_title(f"(a) Calibration:  $R^2$={r2:.3f},  MAE={oof.ae.mean():.2f}\u00b0C")
plt.colorbar(sc,ax=a1,label="rake position z")
maes=[oof[oof.chan==ch].ae.mean() for ch in ORDER]
rmses=[np.sqrt((oof[oof.chan==ch].err**2).mean()) for ch in ORDER]
x=np.arange(len(ORDER))
a2.bar(x-0.2,maes,0.4,color=RUST,label="MAE")
a2.bar(x+0.2,rmses,0.4,color=BLUE,label="RMSE",alpha=.8)
a2.axhline(2,color=GRAY,ls=":",lw=1.5,label="camera \u00b12\u00b0C")
a2.set_xticks(x); a2.set_xticklabels(ORDER,rotation=45); a2.set_ylabel("Error (\u00b0C)")
a2.set_title("(b) Error by depth (bottom \u2192 surface)"); a2.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUTDIR/"fig2_calibration.png"); plt.close(fig)

# ========== FIG 3: generalization (LORO vs LOCO) ==========
loco = {"abs0 f78 s1":1.20,"abs0 f73 s1":1.69,"abs0 f73 s0":1.86,"abs0 f78 s0":3.29,
        "abs0 f88 s1":3.51,"abs0 f88 s0":7.27,"abs20 f88 s1":7.50,"abs20 f88 s0":13.87,
        "abs92 f88 s0":48.89}
fig,ax=plt.subplots(figsize=(11,5))
names=list(loco); vals=list(loco.values())
cols=[RUST if v>20 else (BLUE if v<4 else "#E08A3C") for v in vals]
b=ax.bar(names,vals,color=cols)
ax.axhline(oof.ae.mean(),color="green",ls="--",lw=1.5,label=f"leave-run-out (interp.) = {oof.ae.mean():.2f}\u00b0C")
ax.set_ylabel("MAE (\u00b0C)"); ax.set_yscale("log")
ax.set_title("Generalization: leave-one-condition-out (log scale)")
ax.set_xticklabels(names,rotation=35,ha="right")
for rect,v in zip(b,vals): ax.text(rect.get_x()+rect.get_width()/2,v*1.05,f"{v:.1f}",ha="center",fontsize=8)
ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR/"fig3_generalization.png"); plt.close(fig)

print("\nfigures + tables ->", OUTDIR)
