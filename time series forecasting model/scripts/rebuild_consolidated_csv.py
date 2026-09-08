"""Rebuild the consolidated sweep CSV from a `run_all.py` log.

Some TC10 rows in the consolidated CSV ended up tagged `evaluation_failed` because
the profile-plot step crashed AFTER test evaluation succeeded — the test numbers
are in the log but never made it into the CSV. This script re-extracts every test
result printed by `evaluate_unscaled` and writes a clean consolidated CSV.

Usage (from project root):
    python3 scripts/rebuild_consolidated_csv.py [log_path] [out_csv]

Defaults:
    log_path = run_all_bins.log
    out_csv  = output/sweeps_experiments/consolidated_horizon_sweep_results.csv
"""
import csv
import os
import re
import sys
from collections import defaultdict


def parse_log(log_path):
    """Walk the log and return one record per successful test evaluation.

    Each record dict has the same fields the consolidated CSV header expects.
    """
    with open(log_path, 'r') as f:
        text = f.read()
    lines = text.splitlines()

    experiment = None         # "new_theoretical_TC10" / "_TC11"
    seq_len = 20
    current_h = None          # current horizon being trained
    best_epoch = {}           # H -> best_epoch
    records = []
    cur = None                # in-progress record dict during a test eval

    # Trackers for the POWER BALANCE ANALYSIS block (printed AFTER test eval)
    in_power_block = False
    power_block = {}

    def flush():
        nonlocal cur, in_power_block
        if cur is not None and cur.get('mae_K') is not None:
            records.append(cur)
        cur = None
        in_power_block = False

    for ln in lines:
        # --- which sweep are we in --- (also flushes any pending record)
        m = re.search(r'STARTING SWEEP:\s*(TC\d+)', ln)
        if m:
            flush()
            experiment = f"new_theoretical_{m.group(1)}"
            continue

        # --- start of an experiment --- (also flushes any pending record)
        m = re.search(r'RUNNING SINGLE EXPERIMENT WITH PROFILES:\s*L=(\d+),\s*H=(\d+)', ln)
        if m:
            flush()
            seq_len = int(m.group(1))
            current_h = int(m.group(2))
            continue

        # --- captured best epoch (printed at evaluate time) ---
        m = re.search(r'Evaluating best model for L=\d+,\s*H=(\d+)\s*\(best epoch=(\d+)\)', ln)
        if m:
            best_epoch[int(m.group(1))] = int(m.group(2))
            continue

        # --- start of a TEST SET EVALUATION block ---
        m = re.search(r'TEST SET EVALUATION \(UNSCALED - HORIZON = (\d+) step', ln)
        if m:
            cur = {
                'experiment': experiment,
                'seq_len': seq_len,
                'horizon': int(m.group(1)),
                'status': 'success',
                'best_epoch': best_epoch.get(int(m.group(1))),
                'mae_K': None, 'rmse_K': None, 'r2': None,
                'physics_loss': None, 'power_balance_loss': None,
                'mean_actual_power_W': None,
                'mean_predicted_power_W': None,
                'mean_incoming_power_W': None,
                'conservation_violations_pct': None,
            }
            continue

        if cur is not None:
            m = re.search(r'^\s*MAE:\s*([0-9.eE+-]+)\s*K', ln)
            if m: cur['mae_K'] = float(m.group(1)); continue
            m = re.search(r'^\s*RMSE:\s*([0-9.eE+-]+)\s*K', ln)
            if m: cur['rmse_K'] = float(m.group(1)); continue
            m = re.search(r'^\s*R²:\s*([0-9.eE+-]+)', ln)
            if m: cur['r2'] = float(m.group(1)); continue
            m = re.search(r'^\s*Physics Loss:\s*([0-9.eE+-]+)', ln)
            if m: cur['physics_loss'] = float(m.group(1)); continue
            m = re.search(r'^\s*Power Balance Loss:\s*([0-9.eE+-]+)', ln)
            if m: cur['power_balance_loss'] = float(m.group(1)); continue
            # Power-balance analysis block tops the rest of the per-experiment data
            if 'POWER BALANCE ANALYSIS' in ln:
                in_power_block = True
                power_block = {}
                continue

        if in_power_block:
            # The mean rows look like "Mean: 217.91 W" inside labeled subblocks
            m = re.search(r'INCOMING POWER STATISTICS', ln)
            if m: power_block['_subblock'] = 'incoming'; continue
            m = re.search(r'TOTAL ACTUAL POWER', ln)
            if m: power_block['_subblock'] = 'actual'; continue
            m = re.search(r'TOTAL PREDICTED POWER', ln)
            if m: power_block['_subblock'] = 'predicted'; continue
            m = re.search(r'^\s*Mean:\s*([0-9.eE+-]+)\s*W', ln)
            if m and power_block.get('_subblock'):
                key = {
                    'incoming':  'mean_incoming_power_W',
                    'actual':    'mean_actual_power_W',
                    'predicted': 'mean_predicted_power_W',
                }[power_block['_subblock']]
                if cur is not None:
                    cur[key] = float(m.group(1))
                continue
            # Violation rate line: "Predicted power > incoming: X/Y (Z.Z%)"
            m = re.search(r'Predicted power > incoming:\s*\d+/\d+\s*\(([0-9.]+)%\)', ln)
            if m and cur is not None:
                cur['conservation_violations_pct'] = float(m.group(1))
                continue
            # End of power block on profile-plot line; flush will happen on next sweep/exp marker.
            if 'Profile plots for horizon' in ln:
                in_power_block = False
                continue

    # End of file: flush the last open record
    flush()

    return records


def write_csv(records, out_path):
    """Write records to the consolidated CSV with the standard header."""
    header = [
        'experiment', 'seq_len', 'horizon', 'status', 'best_epoch',
        'mae_K', 'rmse_K', 'r2',
        'physics_loss', 'power_balance_loss',
        'mean_actual_power_W', 'mean_predicted_power_W', 'mean_incoming_power_W',
        'conservation_violations_pct',
    ]
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in records:
            w.writerow({k: (r.get(k) if r.get(k) is not None else '') for k in header})


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'run_all_bins.log'
    out_csv = (sys.argv[2] if len(sys.argv) > 2
               else 'output/sweeps_experiments/consolidated_horizon_sweep_results.csv')

    if not os.path.exists(log_path):
        sys.exit(f"log not found: {log_path}")

    records = parse_log(log_path)

    # Stable de-dupe: keep the LAST successful evaluation per (experiment, horizon).
    # Defensive — if the log was re-run and contains multiple eval blocks for the same
    # experiment we want the most recent.
    keyed = {}
    for r in records:
        keyed[(r['experiment'], r['horizon'])] = r
    records = sorted(keyed.values(), key=lambda r: (r['experiment'], r['horizon']))

    # Backup the existing CSV before overwriting
    if os.path.exists(out_csv):
        bak = out_csv + '.bak'
        os.replace(out_csv, bak)
        print(f"backed up existing CSV → {bak}")

    write_csv(records, out_csv)

    print(f"wrote {len(records)} rows to {out_csv}")
    print()
    print(f"{'experiment':25} {'H':>4}  {'MAE_K':>7}  {'RMSE_K':>7}  {'R²':>9}  {'best_ep':>7}")
    for r in records:
        print(f"{r['experiment']:25} {r['horizon']:>4}  "
              f"{r['mae_K']:>7.2f}  {r['rmse_K']:>7.2f}  {r['r2']:>9.4f}  "
              f"{(r['best_epoch'] if r['best_epoch'] is not None else '?'):>7}")


if __name__ == '__main__':
    main()
