"""
angular.py — Angular Error Metric and Loss
==========================================
Converts (pitch, yaw) predictions and targets into 3-D unit vectors,
then computes the angle between them in degrees.

Why not just use MSE on pitch/yaw directly?
  MSE on raw angles is non-linear in angular distance — e.g. yaw errors
  near ±90° get inflated relative to pitch errors. Converting to 3-D
  vectors and using arccos gives the true great-circle distance.

Reference: Zhang et al., MPIIGaze (TPAMI 2017) — standard evaluation metric.
"""

import torch
import torch.nn as nn


def pitchyaw_to_vector(pitchyaw: torch.Tensor) -> torch.Tensor:
    """
    Convert (pitch, yaw) angles (radians) to unit 3-D gaze vectors.

    Coordinate convention (same as MPIIFaceGaze):
        x =  cos(pitch) * sin(yaw)
        y = -sin(pitch)
        z = -cos(pitch) * cos(yaw)

    Args:
        pitchyaw: (B, 2) tensor — column 0 is pitch, column 1 is yaw.
    Returns:
        vectors:  (B, 3) unit vectors.
    """
    pitch = pitchyaw[:, 0:1]   # (B, 1)
    yaw   = pitchyaw[:, 1:2]   # (B, 1)

    x =  torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)

    vectors = torch.cat([x, y, z], dim=1)   # (B, 3)
    # Normalise just in case of floating-point drift
    vectors = nn.functional.normalize(vectors, p=2, dim=1)
    return vectors


def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-sample angular error in degrees.

    Args:
        pred   : (B, 2) predicted  (pitch, yaw) in radians
        target : (B, 2) ground-truth (pitch, yaw) in radians
    Returns:
        errors : (B,)  angular error per sample in degrees
    """
    v_pred   = pitchyaw_to_vector(pred)    # (B, 3)
    v_target = pitchyaw_to_vector(target)  # (B, 3)

    # Dot product clamped to [-1, 1] to avoid NaN from arccos
    dot = (v_pred * v_target).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    errors_rad = torch.acos(dot)           # (B,)
    return torch.rad2deg(errors_rad)       # (B,)  in degrees


def mean_angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Angular Error (MAE) over a batch — the standard reporting metric.

    Args:
        pred   : (B, 2)
        target : (B, 2)
    Returns:
        scalar MAE in degrees
    """
    return angular_error(pred, target).mean()


class AngularLoss(nn.Module):
    """
    Differentiable angular loss for training.
    Equivalent to mean_angular_error but wrapped as an nn.Module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mean_angular_error(pred, target)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B = 16
    pred   = torch.randn(B, 2) * 0.3   # plausible gaze range
    target = torch.randn(B, 2) * 0.3

    errors = angular_error(pred, target)
    print(f"Per-sample errors (deg): {errors[:5].tolist()}")
    print(f"MAE: {errors.mean().item():.2f}°")

    # Edge case: identical prediction should give ~0°
    zero_err = angular_error(target, target)
    print(f"Self-error (should be ~0): {zero_err.mean().item():.6f}°")
