import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


def angular_error(pred, target):
    """
    Computes mean angular error in degrees between predicted and target gaze.
    pred, target: (B, 2) tensors of (pitch, yaw) in radians.
    """
    # Convert (pitch, yaw) to unit 3D gaze vectors
    def to_vec(p):
        pitch, yaw = p[:, 0], p[:, 1]
        x = torch.cos(pitch) * torch.sin(yaw)
        y = torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        return torch.stack([x, y, z], dim=1)  # (B, 3)

    v1 = to_vec(pred)
    v2 = to_vec(target)

    # Clamp dot product to [-1, 1] for numerical stability
    cos_sim = (v1 * v2).sum(dim=1).clamp(-1.0, 1.0)
    err = torch.acos(cos_sim) * (180.0 / torch.pi)  # radians → degrees
    return err.mean().item()


def train_one_epoch(model, loader, optimizer, device, clip_grad=1.0):
    """
    Runs one training epoch.
    Returns average loss and average angular error for the epoch.
    """
    model.train()
    criterion = nn.MSELoss()
    total_loss, total_err, n_batches = 0.0, 0.0, 0

    for imgs, gazes in loader:
        imgs  = imgs.to(device)
        gazes = gazes.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, gazes)
        loss.backward()

        # Gradient clipping — prevents exploding gradients with transformer
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        total_loss += loss.item()
        total_err  += angular_error(preds.detach(), gazes.detach())
        n_batches  += 1

    return total_loss / n_batches, total_err / n_batches


def evaluate(model, loader, device):
    """
    Evaluates the model on a dataloader.
    Returns average loss and average angular error.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss, total_err, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for imgs, gazes in loader:
            imgs  = imgs.to(device)
            gazes = gazes.to(device)
            preds = model(imgs)
            loss  = criterion(preds, gazes)
            total_loss += loss.item()
            total_err  += angular_error(preds, gazes)
            n_batches  += 1

    return total_loss / n_batches, total_err / n_batches


def evaluate_per_subject(model, subject_loaders, device):
    """
    Evaluates angular error per subject.
    Args:
        subject_loaders: dict of {subject_id: DataLoader}
    Returns:
        dict of {subject_id: angular_error_degrees}
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for subj, loader in subject_loaders.items():
            errors = []
            for imgs, gazes in loader:
                imgs  = imgs.to(device)
                gazes = gazes.to(device)
                preds = model(imgs)

                # Per-sample angular error
                def to_vec(p):
                    pitch, yaw = p[:, 0], p[:, 1]
                    x = torch.cos(pitch) * torch.sin(yaw)
                    y = torch.sin(pitch)
                    z = torch.cos(pitch) * torch.cos(yaw)
                    return torch.stack([x, y, z], dim=1)

                v1 = to_vec(preds)
                v2 = to_vec(gazes)
                cos_sim = (v1 * v2).sum(dim=1).clamp(-1.0, 1.0)
                err = torch.acos(cos_sim) * (180.0 / torch.pi)
                errors.extend(err.cpu().numpy().tolist())

            results[subj] = np.mean(errors)
            print(f"  {subj}: {results[subj]:.2f}°")

    return results