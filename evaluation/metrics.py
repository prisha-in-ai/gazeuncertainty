"""
metrics.py — Evaluation Metrics
================================
Covers:
  1. MAE per subject  (leave-one-out evaluation)
  2. Spearman ρ       (error–uncertainty correlation)
  3. Summary table    (collects results across subjects)

Usage pattern:
    collector = MetricsCollector()
    for batch in dataloader:
        # ... run model ...
        collector.update(subject_ids, errors_deg, uncertainty_scores)
    results = collector.compute()
"""

import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubjectResult:
    """Holds per-subject aggregated results."""
    subject_id: str
    mae: float
    spearman_rho: float
    spearman_pval: float
    n_samples: int


def compute_spearman(
    errors: np.ndarray,
    uncertainties: np.ndarray,
) -> tuple[float, float]:
    """
    Spearman rank correlation between angular errors and uncertainty scores.

    A well-calibrated model should show ρ > 0:
    higher uncertainty → higher error.

    Args:
        errors        : (N,) angular errors in degrees
        uncertainties : (N,) uncertainty scores (e.g. mean alpha)
    Returns:
        (rho, p_value)
    """
    if len(errors) < 3:
        return float("nan"), float("nan")

    rho, pval = spearmanr(errors, uncertainties)
    return float(rho), float(pval)


class MetricsCollector:
    """
    Accumulates predictions over an entire evaluation run, then computes
    per-subject MAE and Spearman ρ in one go.

    Example:
        collector = MetricsCollector()
        for subject_id, mu, alpha, target in eval_loop:
            errors = angular_error(mu, target).cpu().numpy()
            uncert = alpha.mean(dim=1).cpu().numpy()
            collector.update(
                subject_ids=[subject_id] * len(errors),
                errors=errors,
                uncertainties=uncert,
            )
        results = collector.compute()
        collector.print_summary(results)
    """

    def __init__(self):
        # keyed by subject_id string, values are lists
        self._errors:        defaultdict = defaultdict(list)
        self._uncertainties: defaultdict = defaultdict(list)

    def update(
        self,
        subject_ids: list[str],
        errors: np.ndarray,
        uncertainties: np.ndarray,
    ):
        """
        Accumulate a batch of results.

        Args:
            subject_ids   : list of length B — e.g. ['p00', 'p00', ...]
            errors        : (B,) angular errors in degrees
            uncertainties : (B,) uncertainty scores
        """
        for sid, err, unc in zip(subject_ids, errors, uncertainties):
            self._errors[sid].append(float(err))
            self._uncertainties[sid].append(float(unc))

    def compute(self) -> list[SubjectResult]:
        """
        Compute per-subject MAE and Spearman ρ.

        Returns:
            List of SubjectResult, one per subject, sorted by subject_id.
        """
        results = []
        for sid in sorted(self._errors.keys()):
            errs  = np.array(self._errors[sid])
            uncs  = np.array(self._uncertainties[sid])
            rho, pval = compute_spearman(errs, uncs)

            results.append(SubjectResult(
                subject_id   = sid,
                mae          = float(errs.mean()),
                spearman_rho = rho,
                spearman_pval= pval,
                n_samples    = len(errs),
            ))
        return results

    def compute_overall(self) -> dict:
        """
        Aggregate across all subjects.

        Returns dict with keys: mae, spearman_rho, spearman_pval, n_samples.
        """
        all_errors = []
        all_uncs   = []
        for sid in self._errors:
            all_errors.extend(self._errors[sid])
            all_uncs.extend(self._uncertainties[sid])

        all_errors = np.array(all_errors)
        all_uncs   = np.array(all_uncs)
        rho, pval  = compute_spearman(all_errors, all_uncs)

        return {
            "mae":           float(all_errors.mean()),
            "spearman_rho":  rho,
            "spearman_pval": pval,
            "n_samples":     len(all_errors),
        }

    @staticmethod
    def print_summary(results: list[SubjectResult], overall: Optional[dict] = None):
        """Pretty-print per-subject results."""
        header = f"{'Subject':<10} {'MAE (°)':>8} {'Spearman ρ':>12} {'p-value':>10} {'N':>6}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.subject_id:<10} {r.mae:>8.2f} {r.spearman_rho:>12.4f} "
                f"{r.spearman_pval:>10.4f} {r.n_samples:>6}"
            )
        if overall:
            print("-" * len(header))
            print(
                f"{'OVERALL':<10} {overall['mae']:>8.2f} {overall['spearman_rho']:>12.4f} "
                f"{overall['spearman_pval']:>10.4f} {overall['n_samples']:>6}"
            )


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    collector = MetricsCollector()

    # Simulate two subjects with correlated error/uncertainty
    for subject in ["p00", "p01"]:
        uncertainties = rng.uniform(0.1, 1.0, 200)
        # errors correlated with uncertainty + noise (ρ should be > 0)
        errors = uncertainties * 10 + rng.normal(0, 2, 200)
        errors = np.clip(errors, 0, None)
        collector.update(
            subject_ids=[subject] * 200,
            errors=errors,
            uncertainties=uncertainties,
        )

    results = collector.compute()
    overall = collector.compute_overall()
    MetricsCollector.print_summary(results, overall)
