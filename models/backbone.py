import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GazeBackbone(nn.Module):
    """
    Hybrid CNN-Transformer backbone for gaze estimation.

    Architecture:
        Input (3 x 224 x 224)
            → ResNet-18 (pretrained, feature extractor)
            → Feature map flattened to sequence of patches
            → Transformer Encoder with CLS token
            → CLS token projected to 256-dim
            → Regression head → (pitch, yaw)

    Two modes:
        forward(x)           → (B, 2)   pitch + yaw, used during baseline training
        extract_features(x)  → (B, 256) CLS embedding, used for BayesCap
    """

    def __init__(self, feat_dim=256, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        # ── 1. ResNet-18 CNN feature extractor ──────────────────────────────
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        # Output for 224x224 input: (B, 512, 7, 7) → 49 patches

        # ── 2. Projection from 512 to feat_dim ──────────────────────────────
        self.proj = nn.Linear(512, feat_dim)

        # ── 3. CLS token ─────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 4. Transformer Encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── 5. Gaze regression head ──────────────────────────────────────────
        self.gaze_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 2),  # (pitch, yaw)
        )

    def extract_features(self, x):
        """
        Returns 256-dim CLS token embedding.
        Input:  (B, 3, 224, 224)
        Output: (B, 256)
        """
        B = x.size(0)

        # CNN spatial features
        feat = self.cnn(x)                          # (B, 512, 7, 7)
        feat = feat.flatten(2).permute(0, 2, 1)     # (B, 49, 512)

        # Project to feat_dim
        feat = self.proj(feat)                       # (B, 49, 256)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, 256)
        seq = torch.cat([cls, feat], dim=1)          # (B, 50, 256)

        # Transformer encoder
        seq = self.transformer(seq)                  # (B, 50, 256)

        return seq[:, 0, :]                          # (B, 256) — CLS token only

    def forward(self, x):
        """
        Full forward pass for baseline training.
        Input:  (B, 3, 224, 224)
        Output: (B, 2) — pitch and yaw in radians
        """
        features = self.extract_features(x)          # (B, 256)
        return self.gaze_head(features)              # (B, 2)