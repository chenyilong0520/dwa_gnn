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

import torch
from torch_geometric.data import Data

from gnn import SocialForceGNN
from utils import *

# ----------------------------
# Main
# ----------------------------

def main():
    # ============================================================
    # Build PyG Data object
    # ============================================================

    frame_nx5 = np.array(
    [
        [0.0, 0.0, 0.6, 0.0, 1.0],     # robot
        [1.2, 0.3, -0.2, 0.0, 0.0],    # pedestrian 1
        [2.1, -0.4, -0.1, 0.1, 0.0],   # pedestrian 2
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
    

if __name__ == "__main__":
    main()