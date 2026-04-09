import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch
from torch.utils.data import DataLoader

from data.mpiigaze import MPIIGaze, split_subjects
from data.transforms import get_train_transform, get_val_transform
from models.backbone import GazeBackbone
from training.trainer import train_one_epoch, evaluate, evaluate_per_subject
from training.scheduler import get_optimizer, get_scheduler

# ── Load config ───────────────────────────────────────────────────────────────
cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

OUT_ROOT      = cfg["out_root"]
BATCH_SIZE    = cfg["batch_size"]
NUM_EPOCHS    = cfg["num_epochs"]
LR            = cfg["lr"]
FEAT_DIM      = cfg["feat_dim"]
TEST_SUBJECT  = cfg["test_subject"]
VAL_SUBJECT   = cfg["val_subject"]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH     = cfg["checkpoint_path"]

# ── Data ──────────────────────────────────────────────────────────────────────
train_subjects, val_subjects, test_subjects = split_subjects(
    OUT_ROOT, test_subject=TEST_SUBJECT, val_subject=VAL_SUBJECT
)
print(f"Train: {train_subjects}")
print(f"Val:   {val_subjects}")
print(f"Test:  {test_subjects}")

train_dataset = MPIIGaze(OUT_ROOT, train_subjects, transform=get_train_transform())
val_dataset   = MPIIGaze(OUT_ROOT, val_subjects,   transform=get_val_transform())
test_dataset  = MPIIGaze(OUT_ROOT, test_subjects,  transform=get_val_transform())

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = GazeBackbone(feat_dim=FEAT_DIM).to(DEVICE)
optimizer = get_optimizer(model, lr=LR)
scheduler = get_scheduler(optimizer, scheduler_type="cosine", num_epochs=NUM_EPOCHS)

# ── Checkpoint dir ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ── Sanity check ─────────────────────────────────────────────────────────────
sample_imgs, sample_labels = next(iter(train_loader))
print(f"Image shape:  {sample_imgs.shape}")
print(f"Label shape:  {sample_labels.shape}")
print(f"Label range — pitch: [{sample_labels[:,0].min():.2f}, {sample_labels[:,0].max():.2f}]  "
      f"yaw: [{sample_labels[:,1].min():.2f}, {sample_labels[:,1].max():.2f}]")

dummy = torch.randn(8, 3, 224, 224).to(DEVICE)
assert model.extract_features(dummy).shape == (8, FEAT_DIM), "extract_features() shape mismatch!"
print(f"extract_features() shape: OK → (8, {FEAT_DIM})")

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_err = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_err = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss,   val_err   = evaluate(model, val_loader, DEVICE)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f}  Angular Err: {train_err:.2f}° | "
          f"Val Loss: {val_loss:.4f}  Angular Err: {val_err:.2f}°")

    if val_err < best_val_err:
        best_val_err = val_err
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model (val err: {best_val_err:.2f}°)")

# ── Final evaluation ──────────────────────────────────────────────────────────
print("\n── Test Set Evaluation ──")
model.load_state_dict(torch.load(SAVE_PATH))
test_loss, test_err = evaluate(model, test_loader, DEVICE)
print(f"Test Angular Error: {test_err:.2f}°")

print("\n── Per-Subject Error (Val + Test) ──")
subject_loaders = {
    s: DataLoader(MPIIGaze(OUT_ROOT, [s], transform=get_val_transform()),
                  batch_size=BATCH_SIZE, shuffle=False)
    for s in val_subjects + test_subjects
}
evaluate_per_subject(model, subject_loaders, DEVICE)