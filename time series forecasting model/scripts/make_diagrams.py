"""Recreate the three architecture/pipeline diagrams: uniform box size, Arial, tight spacing."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

# --- Arial font ---
for p in ['/Library/Fonts/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf',
          '/Library/Fonts/Arial Bold.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
    if os.path.exists(p):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
plt.rcParams['font.family'] = 'Arial'

OUT = "output/diagrams"
os.makedirs(OUT, exist_ok=True)

BW, BH = 3.9, 1.18          # uniform box size everywhere
GREY = '#d6d6d6'
LW = 2.0


def box(ax, cx, cy, title, sub, fc='white'):
    ax.add_patch(FancyBboxPatch((cx - BW / 2, cy - BH / 2), BW, BH,
                 boxstyle="round,pad=0.0,rounding_size=0.16",
                 linewidth=LW, edgecolor='black', facecolor=fc, mutation_aspect=1.0))
    ax.text(cx, cy + 0.20, title, ha='center', va='center', fontsize=12.5, fontweight='bold')
    if sub:
        ax.text(cx, cy - 0.21, sub, ha='center', va='center', fontsize=10.3)


def arrow(ax, p1, p2):
    ax.annotate('', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle='-|>', mutation_scale=20, lw=LW,
                                color='black', shrinkA=0, shrinkB=0))


def conn(ax, pts, label=None, lpos=None, lha='center'):
    for i in range(len(pts) - 2):
        a, b = pts[i], pts[i + 1]
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=LW, solid_capstyle='round')
    arrow(ax, pts[-2], pts[-1])
    if label:
        ax.text(lpos[0], lpos[1], label, fontsize=10.3, ha=lha, va='center')


def finish(ax, xlim, ylim, name):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')
    ax.figure.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(ax.figure)
    print("wrote", os.path.join(OUT, name))


# ===================== FIG 1 — Overview =====================
fig, ax = plt.subplots(figsize=(11, 5.6))
box(ax, 3.2, 6.0, 'Time-series input', '20 steps: time + TC1-TC11')
box(ax, 7.0, 6.0, 'Operating parameters', 'h, q0, kappa, epsilon')
box(ax, 5.1, 4.1, 'Physics-Guided LSTM', '2-layer LSTM + static branch', GREY)
box(ax, 5.1, 2.2, 'Forecast profile', 'TC1-TC11 at t + H', GREY)
box(ax, 10.3, 4.1, 'Energy constraint', 'stored energy <= input energy', GREY)
box(ax, 10.3, 2.2, 'Training loss', 'MSE + physics penalty')
arrow(ax, (3.2, 5.41), (3.2, 4.69))
arrow(ax, (7.0, 5.41), (7.0, 4.69))
arrow(ax, (5.1, 3.51), (5.1, 2.79))
conn(ax, [(7.0, 4.1), (8.0, 4.1), (8.0, 2.2), (8.41, 2.2)], 'Predictions', (7.55, 4.35), 'left')
conn(ax, [(10.3, 3.51), (10.3, 2.79)], 'Physics penalty', (10.55, 3.15), 'left')
finish(ax, (1.05, 12.4), (1.45, 6.85), 'fig1_overview.png')

# ===================== FIG 2 — Architecture =====================
fig, ax = plt.subplots(figsize=(12.5, 7.6))
box(ax, 2.7, 7.0, 'Sequence input', '20 x 12: time + TC1-TC11')
box(ax, 9.3, 7.0, 'Static input', 'h, q0, kappa, epsilon')
box(ax, 2.7, 5.4, 'LSTM layer 1', '64 units + LayerNorm', GREY)
box(ax, 9.3, 5.4, 'Dense encoder', '4 -> 32 + LayerNorm', GREY)
box(ax, 2.7, 3.8, 'LSTM layer 2', '64 units + Dropout', GREY)
box(ax, 6.0, 2.2, 'Fusion layer', 'Concat(64, 32) -> Dense 32', GREY)
box(ax, 6.0, 0.6, 'Output head', 'Forecast TC1-TC11', GREY)
box(ax, 10.9, 0.6, 'Physics-guided loss', 'energy constraint, unit-safe')
arrow(ax, (2.7, 6.41), (2.7, 5.99))
arrow(ax, (2.7, 4.81), (2.7, 4.39))
arrow(ax, (9.3, 6.41), (9.3, 5.99))
conn(ax, [(2.7, 3.21), (2.7, 2.2), (4.06, 2.2)])
conn(ax, [(9.3, 4.81), (9.3, 2.2), (7.95, 2.2)])
arrow(ax, (6.0, 1.61), (6.0, 1.19))
conn(ax, [(7.95, 0.6), (8.96, 0.6)], 'Predictions', (8.2, 0.86), 'left')
finish(ax, (0.5, 13.0), (-0.15, 7.75), 'fig2_architecture.png')

# ===================== FIG 3 — Pipeline =====================
fig, ax = plt.subplots(figsize=(10, 10.2))
ys = [10.5, 9.0, 7.5, 6.0, 4.5, 3.0, 1.5, 0.0]
box(ax, 4.0, ys[0], 'Raw experimental runs', 'CSV: TC profiles + parameters')
box(ax, 4.0, ys[1], 'Scaling and normalization', 'thermal + parameter scalers')
box(ax, 4.0, ys[2], 'Windowing', '20-step input -> horizon target')
box(ax, 4.0, ys[3], 'Split by run and time', 'train / validation / test')
box(ax, 4.0, ys[4], 'Batch loader', 'sequence + parameters + target')
box(ax, 4.0, ys[5], 'Training loop', 'AdamW: MSE + physics penalty', GREY)
box(ax, 4.0, ys[6], 'Checkpointing', 'early stopping')
box(ax, 4.0, ys[7], 'Evaluation', 'RMSE, MAE, R2, horizon curves')
box(ax, 8.7, 4.5, 'Energy constraint', 'stored <= incident energy', GREY)
for i in range(len(ys) - 1):
    arrow(ax, (4.0, ys[i] - 0.59), (4.0, ys[i + 1] + 0.59))
arrow(ax, (5.95, 4.5), (6.75, 4.5))                                   # batch -> energy
conn(ax, [(8.7, 3.91), (8.7, 3.0), (5.95, 3.0)], 'Physics penalty', (7.0, 3.25), 'center')  # energy -> training
finish(ax, (1.6, 10.9), (-0.75, 11.2), 'fig3_pipeline.png')

print("done")
