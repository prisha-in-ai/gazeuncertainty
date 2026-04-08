"""
calibration.py — Calibration Metrics for Regression Uncertainty
================================================================
For regression (not classification), calibration means:
  "When the model says it's uncertain, is it actually more likely to be wrong?"

We measure this with two tools:

1. Expected Calibration Error (ECE) for regression:
   Bin predictions by predicted uncertainty. Within each bin, compare
   the mean predicted uncertainty to the mean observed error.
   A perfectly calibrated model has ECE = 0.

2. Reliability Diagram Data:
   Returns (bin_uncertainty, bin_error) pairs for plotting.
   In a well-calibrated model these points lie on the diagonal y = x.

Reference: Kuleshov et al., "Accurate Uncertainties for Deep Learning
Using Calibrated Regression" (ICML 2018) — adapted for angular error.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    ece: float
    bin_mean_uncertainty: np.ndarray   # x-axis of reliability diagram
    bin_mean_error: np.ndarray         # y-axis of reliability diagram
    bin_counts: np.ndarray             # number of samples per bin
    n_bins: int


def compute_ece(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
    normalize: bool = True,
) -> CalibrationResult:
    """
    Expected Calibration Error for regression uncertainty.

    Bins samples by predicted uncertainty, then measures whether
    the mean error in each bin matches the mean uncertainty.

    Args:
        errors        : (N,) angular errors in degrees
        uncertainties : (N,) predicted uncertainty scores (e.g. mean alpha)
        n_bins        : number of equal-width bins over uncertainty range
        normalize     : if True, normalise errors and uncertainties to [0,1]
                        before binning so ECE is scale-independent.
    Returns:
        CalibrationResult with ECE and bin arrays for plotting.
    """
    errors        = np.array(errors,        dtype=float)
    uncertainties = np.array(uncertainties, dtype=float)

    if normalize:
        # Scale to [0, 1] so ECE is interpretable across different runs
        unc_min, unc_max = uncertainties.min(), uncertainties.max()
        err_min, err_max = errors.min(), errors.max()

        if unc_max > unc_min:
            uncertainties = (uncertainties - unc_min) / (unc_max - unc_min)
        if err_max > err_min:
            errors = (errors - err_min) / (err_max - err_min)

    # Build bins over uncertainty range
    bin_edges = np.linspace(0.0, 1.0 if normalize else uncertainties.max(), n_bins + 1)
    bin_indices = np.digitize(uncertainties, bin_edges[1:-1])  # 0-indexed bins

    bin_mean_unc  = np.zeros(n_bins)
    bin_mean_err  = np.zeros(n_bins)
    bin_counts    = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_mean_unc[b] = uncertainties[mask].mean()
        bin_mean_err[b] = errors[mask].mean()
        bin_counts[b]   = mask.sum()

    # ECE = weighted mean absolute difference between uncertainty and error
    total = bin_counts.sum()
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] == 0:
            continue
        weight = bin_counts[b] / total
        ece   += weight * abs(bin_mean_unc[b] - bin_mean_err[b])

    return CalibrationResult(
        ece                  = float(ece),
        bin_mean_uncertainty = bin_mean_unc,
        bin_mean_error       = bin_mean_err,
        bin_counts           = bin_counts,
        n_bins               = n_bins,
    )


def reliability_diagram_arrays(result: CalibrationResult):
    """
    Returns (x, y, counts) arrays suitable for plotting.
    Filter out empty bins.

    Usage with matplotlib:
        x, y, counts = reliability_diagram_arrays(result)
        plt.plot([0,1],[0,1], 'k--', label='Perfect calibration')
        plt.scatter(x, y, s=counts/counts.max()*200, label='Model')
    """
    mask = result.bin_counts > 0
    return (
        result.bin_mean_uncertainty[mask],
        result.bin_mean_error[mask],
        result.bin_counts[mask],
    )


def print_calibration_summary(result: CalibrationResult):
    print(f"\nCalibration Summary ({result.n_bins} bins)")
    print(f"  ECE = {result.ece:.4f}  (lower is better; 0 = perfect)")
    print(f"\n  {'Bin':>4} {'Mean Uncertainty':>18} {'Mean Error':>12} {'Count':>7}")
    print("  " + "-" * 45)
    for b in range(result.n_bins):
        if result.bin_counts[b] == 0:
            continue
        print(
            f"  {b:>4} {result.bin_mean_uncertainty[b]:>18.4f} "
            f"{result.bin_mean_error[b]:>12.4f} {result.bin_counts[b]:>7}"
        )


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N   = 1000

    # Well-calibrated model: errors ≈ uncertainties
    uncertainties = rng.uniform(0, 1, N)
    errors_good   = uncertainties + rng.normal(0, 0.05, N)
    errors_good   = np.clip(errors_good, 0, None)

    # Overconfident model: small uncertainty, high error
    errors_bad = rng.uniform(0.5, 1.0, N)

    result_good = compute_ece(errors_good, uncertainties)
    result_bad  = compute_ece(errors_bad,  uncertainties)

    print(f"Well-calibrated ECE:  {result_good.ece:.4f}  (should be low)")
    print(f"Overconfident ECE:    {result_bad.ece:.4f}  (should be higher)")
    print_calibration_summary(result_good)
