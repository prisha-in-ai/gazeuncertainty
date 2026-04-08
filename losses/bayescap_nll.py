"""
bayescap_nll.py — BayesCap NLL Loss + Combined Training Loss
=============================================================
BayesCap models the predictive distribution as a Cauchy distribution
parameterised by (mu, alpha, beta).  The negative log-likelihood of a
Cauchy distribution is:

    NLL = log(alpha) + log(1 + ((y - mu) / alpha)^beta)

Intuitively:
  - If alpha is small (confident) but residual is large → NLL spikes.
  - The model is penalised for being confidently wrong.
  - This is what teaches calibration.

Combined loss used during training:
    L = L1(mu, target) + lambda_nll * NLL(mu, alpha, beta, target)

The L1 term drives angular accuracy.
The NLL term drives calibration of the uncertainty estimate.

Reference: Upadhyay et al., BayesCap (ECCV 2022), Section 3.
"""

import torch
import torch.nn as nn


def bayescap_nll(
    mu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    BayesCap Negative Log-Likelihood loss (mean over batch and output dims).

    NLL_i = log(alpha_i) + log(1 + |residual_i / alpha_i|^beta_i)

    Args:
        mu     : (B, 2) predicted mean (pitch, yaw)
        alpha  : (B, 2) spread parameter, strictly > 0
        beta   : (B, 2) shape  parameter, strictly > 0
        target : (B, 2) ground-truth gaze (pitch, yaw)
    Returns:
        scalar mean NLL
    """
    residual = (target - mu).abs()                          # (B, 2)
    ratio    = (residual / alpha).clamp(min=1e-8)           # (B, 2)
    # Use log1p for numerical stability: log(1 + x) = log1p(x)
    nll = torch.log(alpha) + torch.log1p(ratio.pow(beta))   # (B, 2)
    return nll.mean()


class CombinedGazeLoss(nn.Module):
    """
    Combined L1 + BayesCap NLL loss for training the full model.

        L = L1(mu, target) + lambda_nll * NLL(mu, alpha, beta, target)

    Args:
        lambda_nll (float): Weight on the NLL term.
                            Start at 0.5. Increase → more calibration focus.
                            Decrease → more accuracy focus.
                            If uncertainty collapses or diverges, tune this first.
    """

    def __init__(self, lambda_nll: float = 0.5):
        super().__init__()
        self.lambda_nll = lambda_nll
        self.l1 = nn.L1Loss()

    def forward(
        self,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            mu     : (B, 2)
            alpha  : (B, 2)
            beta   : (B, 2)
            target : (B, 2)
        Returns:
            total_loss : scalar
            components : dict with 'l1', 'nll', 'total' for logging
        """
        l1_loss  = self.l1(mu, target)
        nll_loss = bayescap_nll(mu, alpha, beta, target)
        total    = l1_loss + self.lambda_nll * nll_loss

        components = {
            "l1":    l1_loss.item(),
            "nll":   nll_loss.item(),
            "total": total.item(),
        }
        return total, components


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B = 8
    mu     = torch.randn(B, 2)
    alpha  = torch.ones(B, 2) * 0.5
    beta   = torch.ones(B, 2) * 2.0
    target = torch.randn(B, 2)

    nll = bayescap_nll(mu, alpha, beta, target)
    print(f"NLL loss: {nll.item():.4f}")

    criterion = CombinedGazeLoss(lambda_nll=0.5)
    total, comps = criterion(mu, alpha, beta, target)
    print(f"L1:  {comps['l1']:.4f}")
    print(f"NLL: {comps['nll']:.4f}")
    print(f"Total: {comps['total']:.4f}")

    # A confident-but-wrong model should have higher loss
    alpha_small = torch.ones(B, 2) * 0.05   # very confident
    nll_overconfident = bayescap_nll(mu, alpha_small, beta, target)
    print(f"\nOverconfident NLL (should be >> {nll.item():.2f}): {nll_overconfident.item():.4f}")
