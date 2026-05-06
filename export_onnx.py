#!/usr/bin/env python3
"""
Export RFNet backbone + per-device support centroids into a single ONNX model.

The wrapper model takes raw CSI features (batch, 2, N_subcarriers) and outputs
class similarity scores (batch, num_devices) that can be treated as logits.

Usage:
    python export_onnx.py

Produces:
    rfnet_classifier.onnx   — ready to load in the CSI Monitor app
"""

import os
import glob
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from timm.layers import DropPath


# =============================================================================
# 1. Model architecture (copied from train.py to be self-contained)
# =============================================================================

class Block(nn.Module):
    def __init__(self, dim, expansion, kernel_size, stride=1, padding=0, drop_path=0.0):
        super().__init__()
        inner_dim = dim * expansion
        self.f_in = nn.Conv1d(dim, inner_dim, kernel_size=1, bias=False)
        self.gate = nn.Conv1d(dim, inner_dim, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=inner_dim, bias=False,
        )
        self.project = nn.Conv1d(inner_dim, dim, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.depthwise_conv(self.f_in(x)) * F.silu(self.gate(x))
        x = self.drop_path(self.project(x)) + residual
        return x


class Reduction(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=2, stride=2, bias=False)
        self.ln = nn.LayerNorm(dim_in, eps=1e-6)

    def forward(self, x):
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv(x)
        return x


class RFNet(nn.Module):
    def __init__(self, num_channel=2, layers=[3, 3, 9, 3], base_dim=48,
                 expansion=2, dim_embedding=128):
        super().__init__()
        self.stem = nn.Conv1d(num_channel, base_dim, kernel_size=7, stride=1,
                              padding=3, bias=False)
        dims = [base_dim * (2 ** i) for i in range(len(layers))]
        self.blocks = nn.ModuleList()
        for i in range(len(layers)):
            for _ in range(layers[i]):
                self.blocks.append(Block(dims[i], expansion, kernel_size=5, padding=2))
            if i < len(layers) - 1:
                self.blocks.append(Reduction(dims[i], dims[i + 1]))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        final_spatial_dim = 117 // (2 ** (len(layers) - 1))
        self.classifier = nn.Linear(dims[-1] * final_spatial_dim, dim_embedding, bias=False)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.classifier(x.flatten(1))
        return F.normalize(x, p=2, dim=1)


# =============================================================================
# 2. Wrapper: backbone + StandardScaler + centroids → logits
# =============================================================================

class RFNetClassifier(nn.Module):
    """
    End-to-end wrapper:
        raw (batch, 2, 117) → StandardScaler → RFNet → cosine sim → logits
    """

    def __init__(self, backbone: RFNet, centroids: torch.Tensor,
                 scaler_mean: torch.Tensor, scaler_scale: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        # Store centroids and scaler params as buffers (saved in state_dict)
        self.register_buffer("centroids", centroids)       # (num_classes, 128)
        self.register_buffer("scaler_mean", scaler_mean)   # (234,)
        self.register_buffer("scaler_scale", scaler_scale) # (234,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 2, 117)  — channel 0 = amplitude, channel 1 = phase
        returns: (batch, num_classes)  — cosine similarity scores (usable as logits)
        """
        b = x.shape[0]
        # Flatten to (batch, 234) for scaling, then reshape back
        flat = x.reshape(b, -1)
        flat = (flat - self.scaler_mean) / self.scaler_scale
        x = flat.reshape(b, 2, 117)

        emb = self.backbone(x)                              # (batch, 128)
        logits = torch.matmul(emb, self.centroids.T)        # (batch, num_classes)
        return logits


# =============================================================================
# 3. Load CSI data from the local csi/ directory
# =============================================================================

def read_csi_csv(file_path: str) -> np.ndarray:
    """Read one CSV and return complex CSI matrix (N, 117)."""
    df = pd.read_csv(file_path)
    sub_indices = sorted({
        int(m.group(1))
        for col in df.columns
        if (m := re.search(r"Sub_(\d+)_Real", col))
    })
    n = len(df)
    csi = np.zeros((n, len(sub_indices)), dtype=complex)
    for j, idx in enumerate(sub_indices):
        csi[:, j] = df[f"Sub_{idx}_Real"].values + 1j * df[f"Sub_{idx}_Imag"].values
    return csi


def load_device_data(device_dir: str) -> np.ndarray:
    """Load all CSVs in a device folder → complex matrix."""
    files = sorted(glob.glob(os.path.join(device_dir, "*.csv")))
    matrices = [read_csi_csv(f) for f in files]
    return np.concatenate(matrices, axis=0)


# =============================================================================
# 4. Main export routine
# =============================================================================

def main():
    WEIGHTS   = "best_rfnet_model.pth"
    CSI_ROOT  = "csi"
    OUTPUT    = "rfnet_classifier.onnx"
    DEVICE_DIRS = sorted([
        d for d in os.listdir(CSI_ROOT)
        if os.path.isdir(os.path.join(CSI_ROOT, d))
    ])
    LABEL_MAP = {i: name for i, name in enumerate(DEVICE_DIRS)}

    print(f"Devices found: {DEVICE_DIRS}")
    print(f"Label map: {LABEL_MAP}")

    # -- Load backbone --
    backbone = RFNet(dim_embedding=128)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state)
    backbone.eval()
    print(f"Loaded weights from {WEIGHTS}")

    # -- Load CSI data per device --
    all_features = []   # list of (N_i, 2, 117) arrays
    all_labels   = []

    for label_idx, dname in enumerate(DEVICE_DIRS):
        csi_complex = load_device_data(os.path.join(CSI_ROOT, dname))
        amp   = np.abs(csi_complex)
        phase = np.unwrap(np.angle(csi_complex), axis=1)
        feat  = np.stack([amp, phase], axis=1).astype(np.float32)  # (N, 2, 117)
        all_features.append(feat)
        all_labels.append(np.full(feat.shape[0], label_idx, dtype=int))
        print(f"  {dname}: {feat.shape[0]} samples")

    X = np.concatenate(all_features, axis=0)   # (N_total, 2, 117)
    y = np.concatenate(all_labels, axis=0)

    # -- Fit StandardScaler --
    N = X.shape[0]
    X_flat = X.reshape(N, -1)                  # (N, 234)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(N, 2, 117)

    scaler_mean  = torch.tensor(scaler.mean_,  dtype=torch.float32)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    print(f"Scaler fitted on {N} samples (mean shape: {scaler_mean.shape})")

    # -- Compute centroids --
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        embeddings = backbone(X_tensor)  # (N, 128)

    num_classes = len(DEVICE_DIRS)
    centroids = torch.zeros(num_classes, 128)
    for c in range(num_classes):
        mask = (y == c)
        centroids[c] = F.normalize(embeddings[mask].mean(dim=0), p=2, dim=0)
    print(f"Centroids computed: {centroids.shape}")

    # -- Build wrapper model --
    model = RFNetClassifier(backbone, centroids, scaler_mean, scaler_scale)
    model.eval()

    # Quick sanity check
    dummy = torch.randn(1, 2, 117)
    with torch.no_grad():
        out = model(dummy)
    print(f"Sanity check — input: {dummy.shape} → output: {out.shape}")

    # -- Export to ONNX --
    torch.onnx.export(
        model,
        dummy,
        OUTPUT,
        input_names=["csi_input"],
        output_names=["logits"],
        dynamic_axes={
            "csi_input": {0: "batch_size"},
            "logits":    {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"\nONNX model exported → {OUTPUT}")
    print(f"  Input:  csi_input  shape=(batch, 2, 117)")
    print(f"  Output: logits     shape=(batch, {num_classes})")
    print(f"  Label map: {LABEL_MAP}")

    # -- Save label map as JSON for convenience --
    import json
    with open("label_map.json", "w") as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, indent=2)
    print(f"  Label map saved → label_map.json")


if __name__ == "__main__":
    main()
