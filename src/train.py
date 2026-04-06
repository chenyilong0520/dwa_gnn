#!/usr/bin/env python3
"""
train.py — Train a GNN to predict robot-level CV residual offset y=[dx,dy]

Data format (per sequence):
- You previously saved one NPZ per folder:
    gnn_dataset_<folder>.npz
  with keys:
    xs (object array of [V,5] float32)
    edge_indices (object array of [2,E] int64)
    edge_attrs (object array of [E,7] float32)
    ys (float32 array [N,2])
    metas (object array)

This script:
- Automatically discovers datasets under BASE_DIR by finding output4.xml folders
  OR loads pre-saved gnn_dataset_*.npz from an output directory.
- Builds a PyTorch Geometric Dataset and DataLoaders
- Trains a simple edge-aware message-passing GNN
- Uses per-graph regression loss (MSE) on y
- Reports train/val metrics

Requirements:
  pip install torch torch_geometric
"""

from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

import matplotlib.pyplot as plt
from gnn import *
from utils import *

# ----------------------------
# Train / Eval
# ----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)                  # [B,2]
        y = batch.y.view(-1, 2)              # [B,2]
        mse = F.mse_loss(pred, y, reduction="sum").item()
        mae = F.l1_loss(pred, y, reduction="sum").item()
        mse_sum += mse
        mae_sum += mae
        n += y.shape[0]
    return mse_sum / max(n, 1), mae_sum / max(n, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        y = batch.y.view(-1, 2)
        loss = F.mse_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item() * y.shape[0]
        n += y.shape[0]
    return total / max(n, 1)


def format_float_for_filename(value: float) -> str:
    """
    Convert a float into a compact, filesystem-friendly token.
    Example: 1e-4 -> 1e-04, 0.0003 -> 3e-04.
    """
    return f"{value:.0e}"


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, default="./data/data_processed", help="Directory containing gnn_dataset_*.npz files")
    ap.add_argument(
        "--augmented_npz_dir",
        type=str,
        default="./data/data_augmented",
        help="Directory containing augmented gnn_dataset_*.npz files",
    )
    ap.add_argument(
        "--include_augmented_train",
        action="store_true",
        default=True,
        help="Append augmented NPZ files to the training split only.",
    )
    ap.add_argument(
        "--no_include_augmented_train",
        dest="include_augmented_train",
        action="store_false",
        help="Disable augmented NPZ files for the training split.",
    )
    ap.add_argument(
        "--include_augmented_val",
        action="store_true",
        default=True,
        help="Append augmented NPZ files for the held-out validation sequences.",
    )
    ap.add_argument(
        "--no_include_augmented_val",
        dest="include_augmented_val",
        action="store_false",
        help="Disable augmented NPZ files for the validation split.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of sequences used for validation")
    ap.add_argument("--save_path", type=str, default="gnn_model.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_files = discover_npz_files(args.npz_dir)
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No gnn_dataset_*.npz found in {args.npz_dir}")

    # Sequence-level split (recommended): hold out whole sequences for val
    num_seqs = len(npz_files)
    num_val = max(1, int(round(num_seqs * args.val_ratio)))
    val_files = npz_files[-num_val:]
    train_files = npz_files[:-num_val]

    augmented_train_npz_files = []
    augmented_val_npz_files = []
    all_augmented_npz_files = []
    if args.include_augmented_train or args.include_augmented_val:
        all_augmented_npz_files = discover_npz_files(args.augmented_npz_dir)

    if args.include_augmented_train:
        train_augmented_names = {
            f"{os.path.splitext(os.path.basename(path))[0]}_flip_x.npz"
            for path in train_files
        }
        augmented_train_npz_files = [
            path for path in all_augmented_npz_files
            if os.path.basename(path) in train_augmented_names
        ]
        if len(augmented_train_npz_files) == 0:
            print(
                f"No train-matched augmented NPZ files found in {args.augmented_npz_dir}; "
                "training will use only original data."
            )
    if args.include_augmented_val:
        val_augmented_names = {
            f"{os.path.splitext(os.path.basename(path))[0]}_flip_x.npz"
            for path in val_files
        }
        augmented_val_npz_files = [
            path for path in all_augmented_npz_files
            if os.path.basename(path) in val_augmented_names
        ]
        if len(augmented_val_npz_files) == 0:
            print(
                f"No val-matched augmented NPZ files found in {args.augmented_npz_dir}; "
                "validation will use only original data."
            )

    print("Found NPZ sequences:", num_seqs)
    print("Train sequences:", len(train_files))
    for f in train_files:
        print("  -", os.path.basename(f))
    print("Val sequences:", len(val_files))
    for f in val_files:
        print("  -", os.path.basename(f))
    if augmented_train_npz_files:
        print("Augmented train sequences:", len(augmented_train_npz_files))
        for f in augmented_train_npz_files:
            print("  -", os.path.basename(f))
    if augmented_val_npz_files:
        print("Augmented val sequences:", len(augmented_val_npz_files))
        for f in augmented_val_npz_files:
            print("  -", os.path.basename(f))

    train_data: List[Data] = []
    val_data: List[Data] = []

    for f in train_files:
        train_data.extend(load_samples_from_npz(f))
    for f in augmented_train_npz_files:
        train_data.extend(load_samples_from_npz(f))
    for f in val_files:
        val_data.extend(load_samples_from_npz(f))
    for f in augmented_val_npz_files:
        val_data.extend(load_samples_from_npz(f))

    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = SocialForceGNN(
        node_dim=5,
        edge_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device)
        val_mse, val_mae = evaluate(model, val_loader, device)
        
        train_losses.append(tr_loss)
        val_losses.append(val_mse)

        print(
            f"Epoch {epoch:03d} | "
            f"train_mse={tr_loss:.6f} | val_mse={val_mse:.6f} | val_mae={val_mae:.6f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "val_mse": best_val,
                },
                args.save_path,
            )
            print(f"  ✓ saved best model to {args.save_path} (val_mse={best_val:.6f})")

    
    # ---- Plot losses vs epochs ----
    loss_plot_name = (
        f"loss_lr{format_float_for_filename(args.lr)}_"
        f"hd{args.hidden_dim}_nl{args.num_layers}_bs{args.batch_size}.png"
    )
    plot_loss_history(train_losses, val_losses, loss_plot_name)
    print(f"Saved loss curve to {loss_plot_name}")

    print("Done.")


if __name__ == "__main__":
    main()
