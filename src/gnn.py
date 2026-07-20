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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter, softmax



class EdgeMLPConv(MessagePassing):
    """
    MessagePassing with an explicit edge embedding.
    message = MLP([h_src, h_dst, e])
    aggregate = mean
    update = MLP([h_dst, aggr])
    """
    def __init__(self, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__(aggr="add")  # "mean" or "add" or "max"
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, h=h, edge_emb=edge_emb)

    def message(self, h_j: torch.Tensor, h_i: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.msg_mlp(torch.cat([h_j, h_i, edge_emb], dim=-1))

    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.upd_mlp(torch.cat([h, aggr_out], dim=-1))


class AttentionEdgeConv(MessagePassing):
    """
    Edge-aware message passing with learned attention over incoming edges.
    message = MLP([h_src, h_dst, e])
    attention = softmax(score([h_src, h_dst, e])) over incoming neighbors
    update = MLP([h_dst, weighted_sum(messages)])
    """
    def __init__(self, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_dropout = nn.Dropout(p=dropout_p)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, h=h, edge_emb=edge_emb)

    def message(
        self,
        h_j: torch.Tensor,
        h_i: torch.Tensor,
        edge_emb: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        attn_input = torch.cat([h_j, h_i, edge_emb], dim=-1)
        msg = self.msg_mlp(attn_input)
        attn_logits = self.attn_mlp(attn_input).squeeze(-1)
        attn = softmax(attn_logits, index, ptr=ptr, num_nodes=size_i)
        attn = self.attn_dropout(attn)
        return msg * attn.unsqueeze(-1)

    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.upd_mlp(torch.cat([h, aggr_out], dim=-1))


class SimpleAttentionEdgeConv(MessagePassing):
    """
    Lightweight edge-aware attention.
    Messages still depend on [h_src, h_dst, e], but attention weights are
    predicted from edge features only.
    """
    def __init__(self, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_dropout = nn.Dropout(p=dropout_p)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, h=h, edge_emb=edge_emb)

    def message(
        self,
        h_j: torch.Tensor,
        h_i: torch.Tensor,
        edge_emb: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        msg = self.msg_mlp(torch.cat([h_j, h_i, edge_emb], dim=-1))
        attn_logits = self.attn_mlp(edge_emb).squeeze(-1)
        attn = softmax(attn_logits, index, ptr=ptr, num_nodes=size_i)
        attn = self.attn_dropout(attn)
        return msg * attn.unsqueeze(-1)

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
    def __init__(self, node_dim: int = 5, edge_dim: int = 7, hidden_dim: int = 128, num_layers: int = 3, dropout_p: float = 0.1):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([EdgeMLPConv(hidden_dim, dropout_p) for _ in range(num_layers)])
        #self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)

        for conv in self.convs:
            h = h + conv(h, edge_index, e)  # residual connection
        # for conv in self.convs:
        #     h = h + conv(h, edge_index, e)

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

class AttentionSocialForceGNN(nn.Module):
    """
    Same graph-level interface as SocialForceGNN, but uses attention-based
    edge-aware message passing layers.
    """
    def __init__(self, node_dim: int = 5, edge_dim: int = 7, hidden_dim: int = 128, num_layers: int = 3, dropout_p: float = 0.1):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([AttentionEdgeConv(hidden_dim, dropout_p) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)

        for conv in self.convs:
            h = h + conv(h, edge_index, e)

        if hasattr(data, "ptr") and data.ptr is not None:
            robot_indices = data.ptr[:-1]
            robot_h = h[robot_indices]
        else:
            robot_h = h[0].unsqueeze(0)

        y_hat = self.head(robot_h)
        return y_hat

class SimpleAttentionSocialForceGNN(nn.Module):
    """
    Same interface as SocialForceGNN, but uses lightweight attention where
    neighbor weights are predicted from edge features only.
    """
    def __init__(self, node_dim: int = 5, edge_dim: int = 7, hidden_dim: int = 128, num_layers: int = 3, dropout_p: float = 0.1):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([SimpleAttentionEdgeConv(hidden_dim, dropout_p) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)

        for conv in self.convs:
            h = h + conv(h, edge_index, e)

        if hasattr(data, "ptr") and data.ptr is not None:
            robot_indices = data.ptr[:-1]
            robot_h = h[robot_indices]
        else:
            robot_h = h[0].unsqueeze(0)

        y_hat = self.head(robot_h)
        return y_hat

class InteractionPoolNet(nn.Module):
    """
    Graph -> graph-level y (2D).
    Uses only robot-to-pedestrian edge features, encodes each interaction
    independently, then pools them per graph before regression.
    """
    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        del node_dim, num_layers  # Unused; kept for build_model compatibility.

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.count_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
        )
        pooled_dim = hidden_dim * 2 + hidden_dim // 2
        self.pool_norm = nn.LayerNorm(pooled_dim)
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        edge_index, edge_attr = data.edge_index, data.edge_attr

        if hasattr(data, "ptr") and data.ptr is not None:
            num_graphs = data.ptr.numel() - 1
            robot_indices = data.ptr[:-1]
            # Edges whose source is exactly at a graph boundary belong to the
            # following graph, so we need right=True here.
            edge_batch = torch.bucketize(edge_index[0], data.ptr[1:-1], right=True)
        else:
            num_graphs = 1
            robot_indices = edge_index.new_tensor([0])
            edge_batch = edge_index.new_zeros(edge_index.shape[1])

        src = edge_index[0]
        dst = edge_index[1]
        is_robot_to_ped = (src == robot_indices[edge_batch]) & (dst != src)

        pooled_dim = self.head[0].in_features
        pooled = edge_attr.new_zeros((num_graphs, pooled_dim))
        if is_robot_to_ped.any():
            interaction_h = self.edge_mlp(edge_attr[is_robot_to_ped])
            interaction_batch = edge_batch[is_robot_to_ped]

            attn_logits = self.attn_mlp(interaction_h).squeeze(-1)
            attn_weights = softmax(attn_logits, interaction_batch, num_nodes=num_graphs)
            weighted_sum = scatter(
                interaction_h * attn_weights.unsqueeze(-1),
                interaction_batch,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            )
            max_pool = scatter(
                interaction_h,
                interaction_batch,
                dim=0,
                dim_size=num_graphs,
                reduce="max",
            )
            counts = torch.bincount(
                interaction_batch,
                minlength=num_graphs,
            ).to(edge_attr.dtype).unsqueeze(-1)
            count_feat = self.count_proj(torch.log1p(counts))
            pooled = torch.cat([weighted_sum, max_pool, count_feat], dim=-1)

        pooled = self.pool_norm(pooled)
        y_hat = self.head(pooled)
        return y_hat


def build_model(
    model_name: str,
    node_dim: int = 5,
    edge_dim: int = 7,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout_p: float = 0.1,
) -> nn.Module:
    model_registry = {
        "SocialForceGNN": SocialForceGNN,
        "AttentionSocialForceGNN": AttentionSocialForceGNN,
        "SimpleAttentionSocialForceGNN": SimpleAttentionSocialForceGNN,
        "InteractionPoolNet": InteractionPoolNet,
    }

    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    model_cls = model_registry[model_name]
    return model_cls(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_p=dropout_p,
    )
