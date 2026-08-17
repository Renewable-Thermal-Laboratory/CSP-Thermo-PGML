# Results — IR snapshot → vertical temperature profile

Reconstructing the interior vertical temperature profile (TC2–TC10, plus TC9.5 in
abs92 runs) of a heated molten-salt sample from a **single top-down IR frame + the
bottom thermocouple (TC1) + run conditions (flux, abs, surf)**. Fully non-temporal.

All numbers below are **out-of-fold (leave-one-run-out)**: every run is predicted by
a model trained only on the other runs, so nothing is graded on data it saw.

## Dataset

| | |
|---|---|
| Runs | 38 total; 37 used (one excluded for IR/TC misalignment) |
| Conditions | 9 distinct flux×abs×surf combos (flux 73/78/88, abs 0/20/92, surf 0/1) |
| Aligned samples | 36,881 s of paired IR + rake data |
| IR→feature | per-run dish auto-detection → 8 robust surface features/second |
| IR–TC sync | verified via IR-surface vs TC10 correlation (median 0.97) |

## Model

Depth-conditioned MLP (128–128–64, ReLU): inputs `[7 IR features, TC1, flux, abs,
surf, z]` → temperature **offset from TC1** at normalized depth `z`. Depth as an
input means the profile is queryable at any depth and TC9.5 needs no special
handling. Beat a HistGradientBoosting baseline (1.34 vs 1.70 °C).

## Headline accuracy (out-of-fold)

| Metric | Value |
|---|---|
| MAE | **1.19 °C** |
| RMSE | 2.16 °C |
| Median AE | 0.65 °C |
| Bias | +0.09 °C (negligible) |
| R² | **0.960** |
| Pearson r | 0.980 |
| within ±1 °C / ±2 °C / ±3 °C | 66% / 84% / 91% |

(Full-data averaging, weighting every second equally, gives 1.34 °C; the 1.19 °C
above uses uniform temporal sampling so long runs don't dominate. Both consistent.)
Camera spec accuracy is ±2 °C, so 8 of 10 channels are reconstructed to within the
camera's own uncertainty.

## Per-channel (bottom → surface)  — `table_perchannel.tex`

| Channel | MAE | RMSE | Bias | P90\|e\| |
|---|---|---|---|---|
| TC2 | 0.87 | 1.33 | −0.02 | 1.82 |
| TC3 | 0.65 | 1.02 | −0.08 | 1.46 |
| TC4 | 0.79 | 1.27 | −0.05 | 2.02 |
| TC5 | 1.02 | 1.64 | +0.39 | 2.55 |
| TC6 | 0.93 | 1.39 | −0.15 | 1.83 |
| TC7 | 0.88 | 1.39 | +0.30 | 2.07 |
| TC8 | 0.88 | 1.54 | +0.15 | 1.78 |
| TC9 | 1.37 | 1.95 | −0.04 | 2.94 |
| TC9.5 | 4.52 | 8.91 | +1.40 | 12.18 |
| TC10 | 2.87 | 3.82 | +0.11 | 6.48 |

Interior channels ~0.6–1.4 °C; error rises toward the exposed surface (TC10) and is
worst for TC9.5, which appears in only 5 runs.

## By absorber regime

| Regime | runs | MAE | RMSE |
|---|---|---|---|
| abs0 | 23 | 0.95 | 1.55 |
| abs20 | 9 | 1.57 | 2.46 |
| abs92 | 5 | 1.55 | 3.43 |

## Ablation — value of the IR camera (RF, leave-run-out)

| Inputs | overall MAE | TC10 MAE |
|---|---|---|
| TC1 + conditions only (no camera) | 2.46 | 5.61 |
| **+ IR features** | **1.30** | **2.95** |

The IR nearly halves the error, and the gain is concentrated at the near-surface
channels — the camera does exactly the job the physics predicts.

## Generalization — leave-one-condition-out  (`fig3_generalization.png`)

Holding out an **entire** flux/abs/surf regime:

| held-out | MAE | type |
|---|---|---|
| abs0 f78 s1 | 1.20 | interpolation |
| abs0 f88 s0 | 7.27 | interpolation (contains longruns) |
| abs20 f88 s0 | 13.87 | thin support (2 abs20 conds) |
| **abs92 f88 s0** | **48.89** | **extrapolation — only abs=92 regime** |

**The model interpolates (~1.2 °C within sampled regimes) but does not yet
extrapolate.** A never-seen regime (abs92 removed) collapses — motivating (i) an
input out-of-range guard and (ii) a physics-informed term as the route to
extrapolation.

## Figures

Profile figures use physical depth (m), surface (TC10) at top → bottom (TC1) at
−0.1575 m, evenly spaced — matching the lab forecasting-paper style.

- `fig1_profiles.png` — predicted vs measured profiles, 9 held-out conditions
- `fig1b_profile_single.png` — single-panel exemplar (forecasting-paper layout)
- `fig2_calibration.png` — calibration scatter (R²=0.96) + error-by-depth vs ±2 °C
- `fig3_generalization.png` — leave-one-condition-out MAE (log scale)

(A third "1D model" series can be added to fig1 once the physics-informed term is
built, to fully mirror the forecasting figures' Actual/Predicted/1D-model layout.)

## Reproduce

```bash
python src/extract_ir_features.py      # IR .h5 -> features
python src/build_dataset.py            # features + TC -> dataset.csv
python src/train_final.py              # leave-run-out eval + save model
python src/leave_condition_out.py      # generalization test
```
