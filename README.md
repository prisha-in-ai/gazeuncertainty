# gazeuncertainty

Uncertainty-aware gaze estimation on MPIIGaze using a hybrid CNN-Transformer backbone (ResNet-18 + Transformer Encoder) with a BayesCap head for per-prediction uncertainty.

---

## Setup

### 1. Create and activate the conda environment

```bash
conda create -n bayeshaze python=3.10
conda activate bayeshaze
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Do not use `pip install numpy>=1.24.0` directly in zsh — the `>` character must be quoted. Always install via the requirements file, or use `pip install "numpy>=1.24.0"`.

> **macOS pip path issue:** If `which pip` points to system Python instead of the conda env, use the full path:
> ```bash
> /opt/miniconda3/envs/bayeshaze/bin/pip install -r requirements.txt
> ```

---

## Dataset

Uses the **MPIIGaze** dataset. Download from the official source (requires registration):
> https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild

Place the downloaded dataset at:
```
data/archive/MPIIGaze/
```

The expected structure under that path:
```
MPIIGaze/
  Data/
    Normalized/
      p00/  day01.mat  day02.mat  ...
      p01/  ...
      ...
      p14/
  Evaluation Subset/
    sample list for eye image/
      p00.txt  p01.txt  ...
```

> **No preprocessing needed.** The `Data/Normalized/` directory contains pre-normalized eye images (36×60 grayscale), 3D gaze vectors, and head pose. The `preprocessing/` scripts are only needed if you want to reprocess the raw images and require the full dataset including `Data/Original/pXX/Calibration/Camera.mat`, which is not included in all downloads.

### Update config

In `config.yaml`, set `out_root` to the Normalized directory:
```yaml
out_root: "/path/to/data/archive/MPIIGaze/Data/Normalized"
```

---

## Project Structure

```
gazeuncertainty/
├── config.yaml                  # Training hyperparameters and paths
├── requirements.txt
│
├── data/
│   ├── mpiigaze.py              # MPIIGaze Dataset class + split_subjects()
│   └── transforms.py            # get_train_transform / get_val_transform
│
├── models/
│   ├── backbone.py              # GazeBackbone — CNN-Transformer feature extractor
│   └── bayescap.py              # BayesCapHead — uncertainty prediction head
│
├── losses/
│   ├── angular.py               # Angular loss
│   └── bayescap_nll.py          # BayesCap NLL loss
│
├── training/
│   ├── trainer.py               # train_one_epoch, evaluate, evaluate_per_subject
│   └── scheduler.py             # get_optimizer, get_scheduler
│
├── evaluation/
│   ├── metrics.py               # Angular error metrics
│   ├── selective.py             # Selective prediction (uncertainty thresholding)
│   └── calibration.py           # Uncertainty calibration
│
├── experiments/
│   └── train_baseline.py        # End-to-end baseline training script
│
└── preprocessing/               # Optional — only needed for raw data reprocessing
    ├── data_processing_mpii.py
    └── data_processing_core.py
```

---

## Architecture

### GazeBackbone (`models/backbone.py`)

Hybrid CNN-Transformer for feature extraction:

```
Input (B, 3, 224, 224)
  → ResNet-18 (pretrained ImageNet, layers 1–4 only)
  → Feature map (B, 512, 7, 7) → flattened to 49 patch tokens
  → Linear projection: 512 → 256
  → Prepend CLS token → sequence (B, 50, 256)
  → Transformer Encoder (2 layers, 8 heads, Pre-LN)
  → CLS token output (B, 256)
  → LayerNorm + Linear → (B, 2)  [pitch, yaw in radians]
```

Two modes:
- `forward(x)` → `(B, 2)` — used during baseline training
- `extract_features(x)` → `(B, 256)` — CLS embedding, used as input to BayesCap

### BayesCapHead (`models/bayescap.py`)

Sits on top of the **frozen** backbone. Takes `(B, 256)` features and outputs a Cauchy-like predictive distribution:

```
features (B, 256)
  → Shared MLP: Linear(256→128) + GELU
  → mu_head:    Linear(128→2)              — predicted (pitch, yaw)
  → alpha_head: Linear(128→2) + Softplus   — spread / uncertainty (> 0)
  → beta_head:  Linear(128→2) + Softplus   — shape parameter (> 0)
```

Larger `alpha` = less confident prediction.

---

## Training

### Baseline (backbone only)

```bash
cd /path/to/gazeuncertainty
/opt/miniconda3/envs/bayeshaze/bin/python experiments/train_baseline.py
```

Subject split is leave-one-out by default (`config.yaml`):
- Test: `p14`, Val: `p13`, Train: remaining 13 subjects

Saves best checkpoint to `checkpoints/baseline_p14.pt`.

---

## Config reference (`config.yaml`)

| Key | Description |
|-----|-------------|
| `out_root` | Path to `Data/Normalized/` |
| `test_subject` | Subject held out for test |
| `val_subject` | Subject held out for validation |
| `feat_dim` | CLS token output dim (must match BayesCapHead input) |
| `batch_size` | Training batch size |
| `num_epochs` | Number of training epochs |
| `lr` | Learning rate |
| `weight_decay` | AdamW weight decay |
| `checkpoint_path` | Where to save the best model |
