"""Bin <-> raw-TC conversion helpers.

The PG-LSTM predicts (N-1) bin temperatures, where bin_i = (TC_i + TC_{i+1})/2.
Recovering N raw thermocouples from N-1 bins is underdetermined: the unrecoverable
component is the "zigzag" mode v = (1,-1,1,-1,...). For an ODD sensor count (e.g. 11)
that mode carries a DC component, so naive reconstruction loses the absolute level
(empirically ~29 K offset on TC11). Anchoring on the known last-observed raw TCs and
reconstructing only the (small, smooth) change fixes this — measured added error
~0.5-0.7 K mean for both TC10 and TC11.
"""
import numpy as np


def averaging_matrix(n_out):
    """(n_out x (n_out+1)) matrix A with bin_i = 0.5*(TC_i + TC_{i+1})."""
    N = n_out + 1
    A = np.zeros((n_out, N))
    for i in range(n_out):
        A[i, i] = 0.5
        A[i, i + 1] = 0.5
    return A


def unbin_anchored(pred_bins, last_obs_tcs):
    """Reconstruct raw TC profile from predicted bins, anchored on last-observed raw TCs.

        TC_pred = last_obs_TC + pinv(A) @ (pred_bins - A @ last_obs_TC)

    Fixes the absolute level using the known input-window TCs; only the change over the
    horizon is reconstructed, so the lost zigzag/DC ambiguity is negligible.
    """
    pred_bins = np.asarray(pred_bins, dtype=float)
    last = np.asarray(last_obs_tcs, dtype=float)
    A = averaging_matrix(len(pred_bins))
    return last + np.linalg.pinv(A) @ (pred_bins - A @ last)
