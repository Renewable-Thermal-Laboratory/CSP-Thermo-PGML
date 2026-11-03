# IR Profile Reconstruction

Predict the vertical temperature profile (TC2–TC10, incl. TC9.5 on abs92 runs) in a
heated molten-salt sample from a **single top-down IR snapshot + the bottom
thermocouple (TC1) + run conditions (flux, abs, surf)**. Non-temporal by design:
the deployment target is an industrial setting with only a camera and one probe.

**Picking this up? Start with `HANDOFF.md`** (current status, next steps, gotchas).
Full narrative: `IR_Profile_Model_Explained.docx`.
Results, figures and the per-channel table: `results/RESULTS.md`.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

All scripts resolve paths relative to the project, so run them from anywhere.

## Layout

```
src/
  extract_ir_features.py   IR .h5 -> 8 surface features/second (auto dish detection)
  build_dataset.py         join IR features + TC rake -> data/dataset.csv (sync-verified)
  train_final.py           depth-conditioned model, leave-one-run-out eval, saves model
  leave_condition_out.py   generalization test: hold out a whole flux/abs/surf regime
  mlp_model.py             shared MLP wrapper (import path for the pickle)
  predict.py               deployment CLI: snapshot + TC1 + conditions -> profile
  train.py                 earlier RF baseline + no-IR ablation
  seq_to_h5.py             FLIR .seq -> compact .h5 converter (run where the raw data lives)
  process_tc.py            raw .xlsm/.xlsx thermocouple workbooks -> processed_TC CSVs
  overrides.csv            manual cooldown-trim overrides for process_tc.py
  oof.py / results.py / graph_style.py   out-of-fold eval + figures
results/
  RESULTS.md               metrics, per-channel table, generalization, figure index
  fig1_profiles.png, fig1b_profile_single.png, fig2_calibration.png, fig3_generalization.png
  profiles/                one profile figure per condition
data/
  processed_IR/            38 .h5 IR runs (NOT in git: >100MB each; regenerate via seq_to_h5.py)
  processed_TC/            38 cleaned thermocouple CSVs + cooldown_cuts.txt + process_summary.csv
  ir_features/             extracted per-second IR features per run
  raw_TC/                  original thermocouple workbooks
  dataset.csv              final aligned training table (36,881 s across 38 runs)
models/
  profile_model.joblib     trained depth-conditioned MLP (offset-from-TC1 target)
```

## Reproduce the pipeline

```bash
# 1. raw FLIR .seq  -> compact .h5           (only where the raw data lives)
python src/seq_to_h5.py <raw_seq_dir> -o data/processed_IR
# 2. raw TC workbooks -> cleaned CSVs
python src/process_tc.py <raw_workbooks_dir> -o data/processed_TC
# 3. IR .h5 -> per-second surface features
python src/extract_ir_features.py
# 4. join IR features + TC -> data/dataset.csv
python src/build_dataset.py
# 5. train + leave-one-run-out eval, save models/profile_model.joblib
python src/train_final.py
# 6. generalization test, out-of-fold predictions, figures
python src/leave_condition_out.py
python src/oof.py && python src/results.py && python src/graph_style.py
```

`dataset.csv`, the per-run feature/TC CSVs, and the trained model are committed, so
steps 5-6 and `predict.py` work without the raw .seq / .h5 (steps 1-4 need the raw
data, which lives on the lab acquisition PC).

## Headline result

Leave-one-run-out MAE **1.34 °C** overall (abs0 1.12 / abs20 1.61 / abs92 1.84);
interior channels ~1 °C, surface (TC10) 3.3 °C, TC9.5 4.6 °C (only 5 runs carry it).
Camera spec accuracy is ±2 °C. Excluded run: `h6_abs0_flux73_surf0_run4`
(IR/TC alignment failure, corr −0.91).

This 1.34 °C is the **interpolation** number: predicting an unseen *run* of a
condition the model has other examples of.

## Generalization (leave-one-condition-out)

`src/leave_condition_out.py` holds out an entire flux/abs/surf regime and predicts
it from the other eight — the honest "unseen regime" test.

| held-out condition | runs | MAE (°C) | note |
|---|---|---|---|
| abs0 flux78 surf1 | 3 | 1.20 | well-surrounded |
| abs0 flux73 surf1 | 3 | 1.69 | |
| abs0 flux73 surf0 | 3 | 1.86 | |
| abs0 flux78 surf0 | 4 | 3.29 | |
| abs0 flux88 surf1 | 4 | 3.51 | |
| abs0 flux88 surf0 | 6 | 7.27 | contains both longruns (~390 °C) |
| abs20 flux88 surf1 | 3 | 7.50 | only 2 abs20 conditions exist |
| abs20 flux88 surf0 | 6 | 13.87 | only 2 abs20 conditions exist |
| **abs92 flux88 surf0** | 5 | **48.89** | **pure extrapolation — only abs=92 regime** |

**Takeaway: the model interpolates well but does not yet extrapolate.** Within a
sampled regime it is ~1.3 °C; a nearby unseen condition degrades to ~1–14 °C; a
genuinely new regime (abs92 removed entirely) collapses to ~49 °C. The model must
not be trusted outside the conditions it has training data for — motivating (a) an
input out-of-range guard before deployment and (b) a physics-informed (1D
conduction) term, which is the intended route to real extrapolation.

## Predict

```bash
python src/predict.py --h5 data/processed_IR/<run>.h5 --frame 300 \
       --tc1 358.2 --flux 88 --abs 0 --surf 0 [--dense]
# or with pre-extracted features (live pipeline):
python src/predict.py --features C B P95 P99 P25 STD GRAD --tc1 ... --flux ... --abs ... --surf ...
```

Known limits: only trust conditions seen in training (flux 73/78/88, abs 0/20/92,
surf 0/1); surface-emissivity changes (coatings, surfactant) shift IR readings —
`surf` input covers the studied case only.
