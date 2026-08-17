"""Profile figures in the lab forecasting-paper style (ported from graphs.py):
metrology error bars, 20 C-binned shared temperature axis, Experimental/Predicted
styling, exact 11-slot depth array (TC9.5 at -0.0157 m)."""
import os, re, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SP = "/private/tmp/claude-501/-Users-bhuvan-Desktop-research-ml-models-time-series-forecasting-model/0087db2c-6d8b-477f-91b4-12c8cb8b30b6/scratchpad"
OUT = "/Users/bhuvan/Desktop/research/ml_models/ir_profile_model/results"
os.makedirs(f"{OUT}/profiles", exist_ok=True)

fonts = {f.name for f in fm.fontManager.ttflist}
font_name = next((f for f in ["Arial","Helvetica","DejaVu Sans"] if f in fonts), "DejaVu Sans")
plt.rcParams.update({"font.family":font_name,"font.weight":"bold","axes.labelweight":"bold",
    "axes.titleweight":"bold","axes.linewidth":2.0,"xtick.major.width":2.0,"ytick.major.width":2.0,
    "xtick.major.size":6,"ytick.major.size":6})

# exact channel -> depth (m) from graphs.py 11-slot array
CH_DEPTH = {"TC1":-0.1575,"TC2":-0.1418,"TC3":-0.1260,"TC4":-0.1102,"TC5":-0.0945,
            "TC6":-0.0787,"TC7":-0.0630,"TC8":-0.0472,"TC9":-0.0315,"TC9.5":-0.0157,"TC10":0.0}

def actual_error(temps, ir_index=-1):
    T = np.asarray(temps, float)
    u_tc = np.maximum(2.2, 0.0075*T); u_daq = 0.5 + 0.002*T
    var = (u_tc/np.sqrt(3.))**2 + (u_daq/np.sqrt(3.))**2
    if ir_index is not None:
        u_ir = np.maximum(2.0, 0.02*T)
        for i in np.atleast_1d(ir_index): var[i] = (u_ir[i]/np.sqrt(3.))**2
    return np.sqrt(var)

def style_axes(ax):
    ax.tick_params(axis="both",which="major",labelsize=20,width=2.0,length=6,direction="in",top=True,right=True)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight("bold")
    for s in ax.spines.values(): s.set_linewidth(2.0); s.set_color("black")
    ax.grid(True,color="#b0b0b0",alpha=0.15,linewidth=0.6); ax.set_axisbelow(True)

def make_profile_plot(depth, actual, predicted, title, out, xlim, xticks, actual_err, pred_label="Predicted"):
    fig, ax = plt.subplots(figsize=(6.8,7.8))
    ax.errorbar(actual, depth, xerr=actual_err, fmt="o", color="#ff7f0e", markersize=7,
        markerfacecolor="#ff7f0e", markeredgecolor="#ff7f0e", linewidth=2.2, elinewidth=2.0,
        capsize=6, capthick=2.0, ecolor="black", label="Experimental", zorder=6)
    ax.plot(predicted, depth, color="#1f77b4", linestyle=(0,(4,4)), marker="s", markersize=6,
        markerfacecolor="white", markeredgecolor="#1f77b4", markeredgewidth=1.6, linewidth=2.3,
        label=pred_label, zorder=4)
    ax.set_xlabel(r"Temperature ($^\circ$C)", fontsize=20, fontweight="bold", labelpad=7)
    ax.set_ylabel("Depth (m)", fontsize=20, fontweight="bold", labelpad=7)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.set_xlim(xlim); ax.set_xticks(xticks)
    if len(xticks) > 8: ax.set_xticklabels([str(t) if t%40==0 else "" for t in xticks])
    ax.set_ylim(-0.168, 0.01); style_axes(ax)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.14), ncol=3, fontsize=20,
        frameon=False, handlelength=2.0, handletextpad=0.5, columnspacing=1.2)
    for txt in leg.get_texts(): txt.set_fontweight("bold")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.20, top=0.90)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

def xaxis_20(vals, step=20):
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    t0 = int(np.floor(vmin/step)*step); t1 = int(np.ceil(vmax/step)*step)
    return (t0, t1), list(range(t0, t1+1, step))

# ---- build panels from OOF ----
oof = pd.read_csv(f"{SP}/oof.csv")
oof["cond"] = oof["run"].str.replace(r"_run\d+|_longrun\d+","",regex=True)
reps = [oof[oof.cond==c]["run"].iloc[0] for c in sorted(oof["cond"].unique())]

def panel_data(run):
    g = oof[oof.run==run]; ts=sorted(g["t"].unique()); tmid=ts[len(ts)//2]
    gm = g[g.t==tmid].sort_values("z"); tc1=gm["TC1"].iloc[0]
    chans=["TC1"]+list(gm["chan"]); actual=[tc1]+list(gm["T_true"]); pred=[tc1]+list(gm["T_pred"])
    dep=[CH_DEPTH[c] for c in chans]
    return dep, actual, pred, tmid, (g[g.t==tmid].T_pred-g[g.t==tmid].T_true).abs().mean()

# exemplar
dep,actual,pred,tmid,_ = panel_data("h6_abs0_flux78_surf0_run1")
xlim,xt = xaxis_20(actual+pred)
make_profile_plot(dep,actual,pred,f"Time = {tmid} s",f"{OUT}/fig1b_profile_single.png",
                  xlim,xt,actual_error(np.array(actual),ir_index=-1))

# all 9 single-panel figures, forecasting-paper style
for run in reps:
    dep,actual,pred,tmid,_ = panel_data(run)
    xlim,xt = xaxis_20(actual+pred)
    make_profile_plot(dep,actual,pred, run.replace("h6_",""),
        f"{OUT}/profiles/{run.replace('h6_','')}.png", xlim, xt,
        actual_error(np.array(actual),ir_index=-1))
print(f"9 single-panel figures -> {OUT}/profiles/")

# matching 3x3 grid (same style)
fig, axes = plt.subplots(3,3, figsize=(16.5,17))
for ax, run in zip(axes.flat, reps):
    dep,actual,pred,tmid,mae = panel_data(run)
    xlim,xt = xaxis_20(actual+pred)
    err = actual_error(np.array(actual), ir_index=-1)
    ax.errorbar(actual, dep, xerr=err, fmt="o", color="#ff7f0e", ms=7, mfc="#ff7f0e",
        mec="#ff7f0e", elinewidth=1.8, capsize=5, capthick=1.8, ecolor="black", lw=2.2, label="Experimental", zorder=6)
    ax.plot(pred, dep, color="#1f77b4", linestyle=(0,(4,4)), marker="s", ms=6, mfc="white",
        mec="#1f77b4", mew=1.6, lw=2.3, label="Predicted", zorder=4)
    ax.set_xlim(xlim); ax.set_xticks(xt); ax.set_ylim(-0.168,0.01)
    ax.set_title(run.replace("h6_",""), fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Temperature ($^\circ$C)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=14, fontweight="bold")
    style_axes(ax)
    for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontsize(12)
    ax.text(.03,.03,f"t={tmid}s  MAE={mae:.1f}°C", transform=ax.transAxes, fontsize=11,
            fontweight="bold", bbox=dict(fc="white",ec="none",alpha=.7,pad=2))
h,l = axes.flat[0].get_legend_handles_labels()
fig.legend(h,l, loc="lower center", ncol=2, fontsize=16, frameon=False, bbox_to_anchor=(0.5,-0.005),
           prop={"weight":"bold"})
fig.tight_layout(rect=[0,0.03,1,1])
fig.savefig(f"{OUT}/fig1_profiles.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("grid -> fig1_profiles.png")
