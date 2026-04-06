"""
End-to-end Unity XML -> GNN training samples

What you get per kept frame t:
  - x:         [V, 5]  node features  = [x_rel, y_rel, vx_rel, vy_rel, is_robot]
  - edge_index:[2, E]  bidirectional star robot<->pedestrians (robot is node 0)
  - edge_attr: [E, 7]  edge features  = [dx, dy, dvx, dvy, d, sin(theta), cos(theta)]
  - y:         [2]     robot-level CV residual label (offset)

IMPORTANT (per your requirement):
  - Positions x_i,y_i are already in robot frame (use directly).
  - Velocities vx_i,vy_i in XML are GLOBAL.
      * For node features: convert to robot-frame RELATIVE velocity:
            v_rel_i = v_global_i - v_global_robot
        so robot node has vx_rel_0=0, vy_rel_0=0.
      * For CV residual label: use GLOBAL robot velocity (vx_0, vy_0 from XML):
            y_t = p_t - (p_{t-k} + v_global_robot_{t-k} * k*dt)

Filtering:
  - Keep only frames where nearest pedestrian distance < d_thresh (default 1.0m)

Assumptions:
  - Root tag: <outputInfo>
  - Frames are children named like <t0 ... />, <t1 ... />, ...
  - Attributes: x_i, y_i, vx_i, vy_i for i=0..V-1
  - Robot is always node 0
  - Optional pedestrian "robot flag" attributes like rbt_i may exist; we still set is_robot feature as 1 for node0 else 0
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import os;

import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch_geometric.data import Data

from gnn import SocialForceGNN
from utils import *


def plot_prediction_scene(frame_nx5: np.ndarray, predicted_offset: np.ndarray, save_path: str = "predict_scene.png") -> None:
    """
    Plot robot and pedestrian positions with velocity arrows, plus the predicted
    offset arrow starting at the robot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    robot = frame_nx5[0]
    pedestrians = frame_nx5[1:]

    ax.scatter(robot[0], robot[1], c="tab:blue", s=120, label="robot", zorder=3)
    ax.arrow(
        robot[0],
        robot[1],
        robot[2],
        robot[3],
        color="tab:blue",
        width=0.01,
        length_includes_head=True,
        head_width=0.08,
        zorder=3,
    )
    ax.arrow(
        robot[0],
        robot[1],
        predicted_offset[0],
        predicted_offset[1],
        color="tab:red",
        width=0.01,
        length_includes_head=True,
        head_width=0.08,
        zorder=4,
    )
    predicted_tip = robot[0:2] + predicted_offset
    ax.text(
        predicted_tip[0],
        predicted_tip[1],
        f" pred=({predicted_offset[0]:.3f}, {predicted_offset[1]:.3f})",
        color="tab:red",
        fontsize=10,
        va="bottom",
    )
    ax.text(robot[0], robot[1], " robot", color="tab:blue", fontsize=10, va="bottom")

    for idx, ped in enumerate(pedestrians, start=1):
        ax.scatter(ped[0], ped[1], c="tab:green", s=80, zorder=3)
        ax.arrow(
            ped[0],
            ped[1],
            ped[2],
            ped[3],
            color="tab:green",
            width=0.008,
            length_includes_head=True,
            head_width=0.06,
            zorder=3,
        )
        ax.text(ped[0], ped[1], f" ped{idx}", color="tab:green", fontsize=9, va="bottom")

    all_points = np.vstack(
        [
            frame_nx5[:, 0:2],
            frame_nx5[:, 0:2] + frame_nx5[:, 2:4],
            robot[0:2] + predicted_offset.reshape(1, 2),
        ]
    )
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    half_range = 0.5 * np.max(max_xy - min_xy) + 0.3

    ax.set_xlim(center_xy[0] - half_range, center_xy[0] + half_range)
    ax.set_ylim(center_xy[1] - half_range, center_xy[1] + half_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prediction Scene")
    legend_handles = [
        Line2D([0], [0], color="tab:blue", marker="o", linestyle="-", label="robot"),
        Line2D([0], [0], color="tab:red", linestyle="-", label="predicted offset"),
        Line2D([0], [0], color="tab:green", marker="o", linestyle="-", label="pedestrian"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# ----------------------------
# Main
# ----------------------------

def main():
    # ============================================================
    # Build PyG Data object
    # ============================================================

    frame_nx5 = np.array(
    [
        [1.0, 0.0, 0.6, 0.0, 1.0],     # robot
        [1.2, -0.3, -0.2, 0.0, 0.0],    # pedestrian 1
        [2.1, 0.4, -0.1, -0.1, 0.0],   # pedestrian 2
    ],
    dtype=np.float32,
)

    x_node = preprocess_frame_to_node_features(frame_nx5)
    #edge_index, edge_attr = build_bidirectional_star(x_node)
    edge_index, edge_attr = build_directional_star(x_node)

    data = Data(
        x=torch.from_numpy(x_node),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )

    # ============================================================
    # Load model and predict
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("gnn_model.pt")
    model.eval()

    with torch.no_grad():
        y_hat = model(data.to(device)).cpu().numpy().reshape(-1)

    print("Predicted CV residual [dx, dy]:", y_hat.tolist())
    plot_prediction_scene(frame_nx5, y_hat)
    

if __name__ == "__main__":
    main()
