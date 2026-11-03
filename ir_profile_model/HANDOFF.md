# Handoff: IR profile reconstruction

If you are picking this up, read this first, then `README.md` (how to run) and
`IR_Profile_Model_Explained.docx` (the full plain-language narrative).

## What this project does

Predict the vertical temperature profile (TC2 to TC10, plus TC9.5 on abs92 runs)
inside a heated molten-salt sample from a **single top-down IR snapshot + the bottom
thermocouple (TC1) + run conditions (flux, abs, surf)**. It is non-temporal on
purpose: the deployment target is an industrial setting with only a camera and one
probe, where no time history is available.

## Current status (works today)

- End-to-end pipeline is built and reproducible: raw FLIR `.seq` to compact `.h5`,
  IR surface-feature extraction, join with thermocouples, train, evaluate, predict.
- Trained model is committed (`models/profile_model.joblib`) and `predict.py` runs
  from it with no raw data needed.
- Headline accuracy (leave-one-run-out, out-of-fold): **MAE ~1.2 to 1.3 C, R^2 0.96**.
  Interior channels about 1 C, surface (TC10) about 3 C, TC9.5 about 4.5 C.
- Honest generalization is measured (`leave_condition_out.py`) and documented in
  `results/RESULTS.md`.
- `dataset.csv`, the per-run feature/TC CSVs, the model, and `results/oof.csv` are
  all committed, so training, prediction, and figure regeneration work without the
  raw `.seq` / `.h5`.

## The model, in one paragraph

A small depth-conditioned MLP. Inputs are `[7 IR surface features, TC1, flux, abs,
surf, z]` where `z` is the normalized rake depth; the target is the temperature
**offset from TC1** at depth `z`. Two design choices matter: feeding depth as an
input means the profile is queryable at any depth and TC9.5 is just `z = 9.5` (no
special casing), and predicting the offset from TC1 lets the model learn profile
shape while the measured bottom thermocouple anchors the absolute level. It beat a
gradient-boosting baseline. See `train_final.py` and `mlp_model.py`.

## How timing / sync works (important, non-obvious)

The IR camera's internal clock was never set (every frame reports year 1999), so
absolute camera time is useless. Sync is instead achieved by trimming **both** the
IR `.h5` and the processed TC CSV to the same run window, so they align
second-for-second at index 0. This was verified: the IR surface feature `ir_bulk`
tracks the surface thermocouple TC10 with a median correlation of 0.97 across runs
(`build_dataset.py` prints this check).

## Next steps (roughly prioritized)

1. **Recover or confirm the one excluded run** `h6_abs0_flux73_surf0_run4`. It is
   dropped because `corr(ir_bulk, TC10) = -0.91` (anti-correlated), which points to a
   bad IR/TC alignment or a dish-detection glitch for that run. Either fix it or
   document why it is genuinely bad. It is excluded via `BAD_ALIGN` in
   `train_final.py`.
2. **Physics-informed term (biggest lever).** The model interpolates well but does
   not extrapolate: holding out an entire unseen regime degrades to 1 to 14 C, and
   removing abs92 entirely collapses to ~49 C (see `RESULTS.md`). Adding a 1D
   steady-state conduction residual as a loss, and/or pretraining on synthetic
   profiles from the lab's 1D model, is the intended route to real extrapolation.
   Bonus: it would let us add the green "1D model" series to the profile figures to
   fully match the forecasting-paper layout (`graph_style.py` is already structured
   for a third series via `make_profile_plot`'s `model_1d` slot).
3. **Out-of-range input guard before any deployment.** The model will output a
   confident, wrong profile for conditions it never saw (new flux, new salt). Add a
   check that warns when inputs fall outside the training envelope
   (flux 73/78/88, abs 0/20/92, surf 0/1). This is the main silent-failure risk.
4. **More abs20 and abs92 data, especially runs carrying TC9.5** (only 5 runs have
   it, which is why TC9.5 error is high). This is a data-collection task, not a
   modeling one.
5. **Live-camera feature pipeline.** `predict.py` reads a stored `.h5` or
   pre-extracted features; real deployment needs the dish detection + feature
   extraction running on a live IR feed.
6. **Review the 4 length-mismatch runs** flagged by `build_dataset.py`
   (`abs0_flux78_surf1_run2`, `abs0_flux88_surf0_longrun1`,
   `abs20_flux88_surf0_longrun2`, `abs92_flux88_surf0_run2`): IR and TC lengths differ
   by more than the tolerance; they still align at the start (high corr) so they are
   truncated to the shorter length, but the trim windows should be checked.

## Known issues and gotchas

- **Emissivity drift.** surfactant (surf=1) runs read about 130 C lower in the IR
  than the contact thermocouple, because surfactant changes surface emissivity. The
  model handles it only because `surf` is an input. Any surface change in the field
  (coatings, oxidation) will shift the IR readings the same way and is the number-one
  deployment risk to monitor.
- **scikit-learn is pinned to 1.9** in `requirements.txt` so the saved model unpickles
  without version warnings. If you retrain, you can relax this.
- **Raw data location.** The raw `.seq` (about 600 GB) and the converted `.h5`
  (about 18 GB) are not in git. They live on the lab acquisition PC. Pipeline steps 1
  to 4 (raw to processed) need them; everything downstream is reproducible from the
  committed files.
- **Rake geometry.** TC1 is the bottom (groove), TC10 is the exposed surface (coldest
  and most variable, which is what the IR camera sees). Depths are evenly spaced from
  0 (surface, TC10) to -0.1575 m (bottom, TC1); TC9.5 sits at -0.0157 m. See
  `graph_style.py` `CH_DEPTH`.

## File map (what to touch for what)

| You want to | Look at |
|---|---|
| Change the model | `train_final.py`, `mlp_model.py` |
| Change IR features | `extract_ir_features.py` (dish detection + 8 features) |
| Change how IR and TC are joined / synced | `build_dataset.py` |
| Run a prediction | `predict.py` |
| Re-measure generalization | `leave_condition_out.py` |
| Rebuild figures / stats | `oof.py` then `results.py` and `graph_style.py` |
| Re-process raw data | `seq_to_h5.py` (IR), `process_tc.py` + `overrides.csv` (TC) |
