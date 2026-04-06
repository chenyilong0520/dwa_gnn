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
from typing import List, Tuple, Dict, Any

import numpy as np
import torch

# PyTorch Geometric
from torch_geometric.data import Data

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from gnn import *



# =========================
# Utils
# =========================

def natural_key(s: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", s)]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_loss_history(train_losses, val_losses, save_path):
    """
    Plots the loss history
    """
    plt.figure()
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train_mse")
    plt.plot(epochs, val_losses, label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

# =========================
# Loading NPZ datasets
# =========================

def discover_npz_files(npz_dir: str, prefix: str = "gnn_dataset_", suffix: str = ".npz") -> List[str]:
    if not os.path.isdir(npz_dir):
        return []
    files = []
    for name in os.listdir(npz_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            files.append(os.path.join(npz_dir, name))
    files.sort(key=natural_key)
    return files


def load_samples_from_npz(npz_path: str) -> List[Data]:
    z = np.load(npz_path, allow_pickle=True)
    xs = z["xs"]
    edge_indices = z["edge_indices"]
    edge_attrs = z["edge_attrs"]
    ys = z["ys"]  # [N,2] float32
    # metas optional
    metas = z["metas"] if "metas" in z.files else None

    data_list: List[Data] = []
    n = len(xs)
    for i in range(n):
        #x = torch.tensor(xs[i], dtype=np.float32)  # [V,5]
        x = torch.from_numpy(np.asarray(xs[i], dtype=np.float32))
        #edge_index = torch.tensor(edge_indices[i], dtype=np.int64)  # [2,E]
        edge_index = torch.from_numpy(np.asarray(edge_indices[i], dtype=np.int64))
        #edge_attr = torch.tensor(edge_attrs[i], dtype=np.float32)  # [E,7]
        edge_attr = torch.from_numpy(np.asarray(edge_attrs[i], dtype=np.float32))  # [E,7]
        #y = torch.tensor(ys[i], dtype=np.float32).view(1, 2)  # [1,2] for graph-level
        y = torch.from_numpy(np.asarray(ys[i], dtype=np.float32)).view(1, 2)  # [1,2] for graph-level

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        if metas is not None:
            meta = dict(metas[i])
            meta.setdefault("augmented", False)
            meta.setdefault("augmentation", "none")
            d.meta = meta
        data_list.append(d)
    return data_list


def flip_graph_sample_x(
    x: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    meta: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Mirror one graph sample across the y-axis in robot frame.

    Flipped terms:
      - node feature x_rel   -> -x_rel
      - node feature vx_rel  -> -vx_rel
      - label dx             -> -dx

    Edge features are recomputed from the mirrored node features so the
    direction-dependent terms stay self-consistent.
    """
    x_aug = np.asarray(x, dtype=np.float32).copy()
    edge_index_aug = np.asarray(edge_index, dtype=np.int64).copy()
    y_aug = np.asarray(y, dtype=np.float32).copy()

    x_aug[:, 0] *= -1.0
    x_aug[:, 2] *= -1.0
    y_aug[0] *= -1.0
    edge_attr_aug = compute_edge_attr(x_aug, edge_index_aug)

    meta_aug: Dict[str, Any] = dict(meta) if meta is not None else {}
    meta_aug["augmented"] = True
    meta_aug["augmentation"] = "flip_x"

    return x_aug, edge_index_aug, edge_attr_aug, y_aug, meta_aug


def save_graph_samples_npz(npz_path: str, samples) -> None:
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        xs=np.array([s.x for s in samples], dtype=object),
        edge_indices=np.array([s.edge_index for s in samples], dtype=object),
        edge_attrs=np.array([s.edge_attr for s in samples], dtype=object),
        ys=np.array([s.y for s in samples], dtype=np.float32),
        metas=np.array([s.meta for s in samples], dtype=object),
    )

# =========================
# Loading gnn models
# =========================

def load_model(checkpoint_path:str)->SocialForceGNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device,weights_only=True)
    print(checkpoint_path+" loaded.")
    model = SocialForceGNN(
        node_dim=5,
        edge_dim=7,
        hidden_dim=ckpt["args"].get("hidden_dim", 128),
        num_layers=ckpt["args"].get("num_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


# =========================
# XML parsing utilities
# =========================

def infer_num_nodes(attrib: Dict[str, str]) -> int:
    """
    Infer V from keys like x_0, x_1, ...
    """
    max_i = -1
    for k in attrib.keys():
        m = re.match(r"^x_(\d+)$", k)
        if m:
            max_i = max(max_i, int(m.group(1)))
    if max_i < 0:
        raise ValueError("No node features found (missing keys like x_0, x_1, ...).")
    return max_i + 1


def parse_frame_to_node_features(
    frame: ET.Element,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse one frame into:
      - x_node: [V,5] node features = [x_rel, y_rel, vx_rel, vy_rel, is_robot]
      - p_robot_global: [2] global robot position (x_0, y_0 from XML)
      - v_robot_global: [2] global robot velocity (vx_0, vy_0 from XML)

    Inputs:
      - x_i,y_i are global positions
      - vx_i,vy_i are global velocities
    """
    attrib = frame.attrib
    V = infer_num_nodes(attrib)

    # Read robot-frame positions and GLOBAL velocities first
    pos_glb = np.zeros((V, 2), dtype=np.float32)
    vel_glb = np.zeros((V, 2), dtype=np.float32)

    for i in range(V):
        pos_glb[i, 0] = float(attrib.get(f"x_{i}", 0.0))
        pos_glb[i, 1] = float(attrib.get(f"y_{i}", 0.0))
        vel_glb[i, 0] = float(attrib.get(f"vx_{i}", 0.0))
        vel_glb[i, 1] = float(attrib.get(f"vy_{i}", 0.0))

    p_robot_global = pos_glb[0].copy()
    v_robot_global = vel_glb[0].copy()

    # Convert GLOBAL velocities -> robot-relative velocities for node features
    pos_rel = pos_glb - p_robot_global
    vel_rel = vel_glb - v_robot_global  # broadcasting: [V,2] - [2]

    # Assemble node feature matrix
    x = np.zeros((V, 5), dtype=np.float32)
    x[:, 0:2] = pos_rel
    x[:, 2:4] = vel_rel

    # is_robot feature (robot is always node 0)
    x[:, 4] = 0.0
    x[0, 4] = 1.0

    return x, p_robot_global ,v_robot_global

def preprocess_frame_to_node_features(frame: np.ndarray) -> np.ndarray:
    """
    Input:  [N,5] = [x,y,vx_global,vy_global,rbt]
    Output: [N,5] = [x_rel,y_rel,vx_rel,vy_rel,is_robot]
    """
    pos_glb = frame[:, 0:2]
    vel_glb = frame[:, 2:4]

    p_robot_glob = pos_glb[0].copy()
    v_robot_glb = vel_glb[0].copy()
    pos_rel = pos_glb - p_robot_glob
    vel_rel = vel_glb - v_robot_glb

    is_robot = np.zeros((frame.shape[0], 1), dtype=np.float32)
    is_robot[0, 0] = 1.0

    return np.concatenate([pos_rel, vel_rel, is_robot], axis=1).astype(np.float32)

# =========================
# Filtering: nearest pedestrian distance
# =========================

def nearest_ped_distance(x: np.ndarray) -> float:
    """
    Nearest distance from robot(node0) to any pedestrian(node1..).
    Uses robot-frame positions in x[:,0:2].
    """
    V = x.shape[0]
    if V <= 1:
        return float("inf")
    dx = x[1:, 0] - x[0, 0]
    dy = x[1:, 1] - x[0, 1]
    d = np.sqrt(dx * dx + dy * dy)
    return float(np.min(d))


# =========================
# Label: robot-level CV residual (uses GLOBAL robot velocity)
# =========================

def robot_cv_residual_label(
    p_curr_global: np.ndarray,
    p_prev_global: np.ndarray,
    v_prev_global: np.ndarray,
    dt: float,
    k: int,
) -> np.ndarray:
    """
    Robot-level CV residual:
      y = p_curr - (p_prev + v_prev_global * (k*dt))
    where p_* are global-frame positions,
    and v_prev_global is GLOBAL (vx_0,vy_0) from XML for the prev frame.
    """
    p_pred = p_prev_global + v_prev_global * (k * dt)
    return (p_curr_global - p_pred).astype(np.float32)


# =========================
# Graph construction
# =========================

def build_bidirectional_star(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robot (node 0) <-> each pedestrian.
    If V = N+1, E = 2N directed edges.
    """
    V = x.shape[0]
    if V <= 1:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 7), dtype=np.float32)
        return edge_index, edge_attr

    src, dst = [], []
    for i in range(1, V):
        src.extend([0, i])
        dst.extend([i, 0])

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = compute_edge_attr(x, edge_index)
    return edge_index, edge_attr

def build_directional_star(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robot (node 0) <-> each pedestrian.
    If V = N+1, E = N directed edges.
    """
    V = x.shape[0]
    if V <= 1:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 7), dtype=np.float32)
        return edge_index, edge_attr

    src, dst = [], []
    src = list(range(1,V))
    dst = [0] * (V-1)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = compute_edge_attr(x, edge_index)
    return edge_index, edge_attr


def compute_edge_attr(x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """
    edge_attr = [dx, dy, dvx, dvy, d, sin(theta), cos(theta)]
    computed from node features:
      - dx,dy from robot-frame positions (x_rel,y_rel)
      - dvx,dvy from relative velocities (vx_rel,vy_rel)
    Uses (dst - src) for all terms.
    """
    if edge_index.shape[1] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    src = edge_index[0]
    dst = edge_index[1]

    dx  = x[dst, 0] - x[src, 0]
    dy  = x[dst, 1] - x[src, 1]
    dvx = x[dst, 2] - x[src, 2]
    dvy = x[dst, 3] - x[src, 3]

    d = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    return np.stack(
        [dx, dy, dvx, dvy, d, np.sin(theta), np.cos(theta)],
        axis=1
    ).astype(np.float32)
