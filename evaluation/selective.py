"""
selective.py — Selective Prediction Curve
==========================================
Key experiment: "Does filtering out high-uncertainty predictions improve MAE?"

Procedure:
  1. Sort all predictions by uncertainty (ascending = most confident first).
  2. For each retention fraction r ∈ [0.1, 0.2, ..., 1.0]:
     - Keep only the r * N most confident predictions.
     - Compute MAE on the retained subset.
  3. Plot MAE vs. % retained.

Interpretation:
  - A steep downward curve → uncertainty is informative (well-calibrated).
  - A flat curve → uncertainty is uninformative (random / uncalibrated).
  - The area under this curve (AUC) summarises performance in one number.

This is the core experiment for your report — it directly answers:
"Can the model's own uncertainty flag unreliable predictions?"
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SelectivePredictionResult:
    retention_fractions: np.ndarray   # x-axis, e.g. [0.1, 0.2, ..., 1.0]
    mae_at_fraction:     np.ndarray   # y-axis, MAE at each retention level
    n_retained:          np.ndarray   # number of samples kept at each level
    auc:                 float        # area under curve (lower = better)
    mae_full:            float        # baseline MAE at 100% retention


def compute_selective_prediction(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    fractions: np.ndarray | None = None,
) -> SelectivePredictionResult:
    """
    Compute the selective prediction curve.

    Args:
        errors        : (N,) angular errors in degrees
        uncertainties : (N,) uncertainty scores — HIGHER = less confident
        fractions     : retention fractions to evaluate, default 0.1→1.0 in 0.1 steps
    Returns:
        SelectivePredictionResult
    """
    errors        = np.array(errors,        dtype=float)
    uncertainties = np.array(uncertainties, dtype=float)
    N             = len(errors)

    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 10)
    fractions = np.array(fractions)

    # Sort by uncertainty ascending → most confident first
    sorted_idx = np.argsort(uncertainties)

    mae_at_fraction = np.zeros(len(fractions))
    n_retained      = np.zeros(len(fractions), dtype=int)

    for i, frac in enumerate(fractions):
        k = max(1, int(np.ceil(frac * N)))
        retained_idx = sorted_idx[:k]                  # top-k most confident
        mae_at_fraction[i] = errors[retained_idx].mean()
        n_retained[i]      = k

    # AUC via trapezoidal integration (normalised by x-range)
    auc = float(np.trapz(mae_at_fraction, fractions) / (fractions[-1] - fractions[0]))

    return SelectivePredictionResult(
        retention_fractions = fractions,
        mae_at_fraction     = mae_at_fraction,
        n_retained          = n_retained,
        auc                 = auc,
        mae_full            = float(errors.mean()),
    )


def compute_random_baseline(
    errors: np.ndarray,
    fractions: np.ndarray | None = None,
    n_trials: int = 20,
) -> np.ndarray:
    """
    Random baseline: filter predictions randomly (not by uncertainty).
    Returns mean MAE across trials for each retention fraction.

    Useful to show in the plot that your model's curve beats random selection.

    Args:
        errors    : (N,) angular errors
        fractions : same fractions used in selective prediction
        n_trials  : number of random shuffles to average
    Returns:
        (len(fractions),) mean MAE under random selection
    """
    errors = np.array(errors, dtype=float)
    N      = len(errors)

    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 10)

    rng = np.random.default_rng(42)
    all_maes = np.zeros((n_trials, len(fractions)))

    for t in range(n_trials):
        perm = rng.permutation(N)
        for i, frac in enumerate(fractions):
            k = max(1, int(np.ceil(frac * N)))
            all_maes[t, i] = errors[perm[:k]].mean()

    return all_maes.mean(axis=0)


def print_selective_summary(result: SelectivePredictionResult):
    improvement = result.mae_full - result.mae_at_fraction[0]
    print(f"\nSelective Prediction Summary")
    print(f"  Full-data MAE (100% retained): {result.mae_full:.2f}°")
    print(f"  MAE at 10% retained:           {result.mae_at_fraction[0]:.2f}°")
    print(f"  MAE improvement (10% vs 100%): {improvement:.2f}°")
    print(f"  AUC (lower = better):          {result.auc:.4f}")
    print(f"\n  {'Fraction':>10} {'MAE (°)':>10} {'N retained':>12}")
    print("  " + "-" * 35)
    for frac, mae, n in zip(result.retention_fractions, result.mae_at_fraction, result.n_retained):
        print(f"  {frac:>10.1%} {mae:>10.2f} {n:>12}")


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N   = 500

    # Simulate well-calibrated model: uncertainty ≈ error magnitude
    uncertainties = rng.uniform(0, 1, N)
    errors = uncertainties * 15 + rng.normal(0, 1, N)
    errors = np.clip(errors, 0, None)

    result = compute_selective_prediction(errors, uncertainties)
    random_baseline = compute_random_baseline(errors, result.retention_fractions)

    print_selective_summary(result)

    print("\nComparison at each fraction:")
    print(f"  {'Fraction':>10} {'Model MAE':>12} {'Random MAE':>12}")
    for frac, m_mae, r_mae in zip(result.retention_fractions, result.mae_at_fraction, random_baseline):
        print(f"  {frac:>10.1%} {m_mae:>12.2f} {r_mae:>12.2f}")
