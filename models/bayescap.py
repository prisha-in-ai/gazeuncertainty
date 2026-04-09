"""
bayescap.py — BayesCap Head for Uncertainty-Aware Gaze Estimation
==================================================================
Sits on top of the frozen CNN-Transformer backbone.
Takes a feature vector (B, feat_dim) and outputs:
  mu    (B, 2) — predicted pitch and yaw
  alpha (B, 2) — spread parameter (> 0), learned via Softplus
  beta  (B, 2) — shape parameter  (> 0), learned via Softplus

The (mu, alpha, beta) triplet parameterises a Cauchy-like predictive
distribution. alpha encodes uncertainty: larger alpha → less confident.

Reference: Upadhyay et al., BayesCap (ECCV 2022)
"""

import torch
import torch.nn as nn


class BayesCapHead(nn.Module):
    """
    BayesCap head that predicts mean + uncertainty parameters.

    Args:
        feat_dim (int): Dimensionality of the incoming feature vector.
                        Must match the CLS-token output dim of the backbone.
                        Default 256 — agree this with M1 before wiring.
        hidden_dim (int): Hidden layer width inside each branch.
    """

    def __init__(self, feat_dim: int = 256, hidden_dim: int = 128):
        super().__init__()

        # ── Shared projection ──────────────────────────────────────────────
        # Small shared MLP so all three branches start from the same
        # intermediate representation, reducing parameter count.
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
        )

        # ── Mean branch ───────────────────────────────────────────────────
        # Predicts (pitch, yaw) in radians — no activation, unbounded.
        self.mu_head = nn.Linear(hidden_dim, 2)

        # ── Alpha branch (spread) ─────────────────────────────────────────
        # Softplus ensures strictly positive output.
        # eps clamp prevents collapse to zero.
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softplus(),
        )

        # ── Beta branch (shape) ───────────────────────────────────────────
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for stable early training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (B, feat_dim) — frozen backbone output.

        Returns:
            mu    : (B, 2) gaze direction [pitch, yaw] in radians
            alpha : (B, 2) spread, strictly > 0
            beta  : (B, 2) shape,  strictly > 0
        """
        eps = 1e-6
        h = self.shared(features)

        mu    = self.mu_head(h)
        alpha = self.alpha_head(h) + eps
        beta  = self.beta_head(h)  + eps

        return mu, alpha, beta

    def uncertainty_score(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Collapse (B, 2) alpha into a scalar per-sample uncertainty score.
        Uses the mean across pitch and yaw.

        Args:
            alpha: (B, 2)
        Returns:
            scores: (B,)  — higher means less confident
        """
        return alpha.mean(dim=1)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, feat_dim = 8, 256
    dummy = torch.randn(B, feat_dim)

    head = BayesCapHead(feat_dim=feat_dim)
    mu, alpha, beta = head(dummy)

    print(f"mu    shape: {mu.shape}")     # (8, 2)
    print(f"alpha shape: {alpha.shape}")  # (8, 2)
    print(f"beta  shape: {beta.shape}")   # (8, 2)
    print(f"alpha min (should be > 0): {alpha.min().item():.6f}")
    print(f"Uncertainty scores: {head.uncertainty_score(alpha)}")
