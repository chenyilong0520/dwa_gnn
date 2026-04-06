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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing



class EdgeMLPConv(MessagePassing):
    """
    MessagePassing with an explicit edge embedding.
    message = MLP([h_src, h_dst, e])
    aggregate = mean
    update = MLP([h_dst, aggr])
    """
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, h=h, edge_emb=edge_emb)

    def message(self, h_j: torch.Tensor, h_i: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.msg_mlp(torch.cat([h_j, h_i, edge_emb], dim=-1))

    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.upd_mlp(torch.cat([h, aggr_out], dim=-1))


class SocialForceGNN(nn.Module):
    """
    Graph -> graph-level y (2D).
    Uses:
      node encoder (5 -> H)
      edge encoder (7 -> H)
      K message passing layers (residual)
      Readout: take robot node embedding (node 0) OR pooled embedding
      Head: H -> 2
    """
    def __init__(self, node_dim: int = 5, edge_dim: int = 7, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([EdgeMLPConv(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)

        for conv in self.convs:
            h = h + conv(h, edge_index, e)  # residual connection

        # Robot node is node 0 for each graph.
        # In a batched Data object, nodes are concatenated. We need per-graph robot indices.
        # robot index for each graph is the first node in that graph.
        # PyG provides `data.ptr` for batch graphs (start offsets).
        if hasattr(data, "ptr") and data.ptr is not None:
            robot_indices = data.ptr[:-1]  # [num_graphs]
            robot_h = h[robot_indices]     # [B,H]
        else:
            # single graph
            robot_h = h[0].unsqueeze(0)

        y_hat = self.head(robot_h)  # [B,2]
        return y_hat