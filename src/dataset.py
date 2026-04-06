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

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import os

import numpy as np
from utils import *


# =========================
# Data container
# =========================

@dataclass
class GraphSample:
    x: np.ndarray            # [V, 5]  node features (robot-frame position + relative velocity)
    edge_index: np.ndarray   # [2, E]
    edge_attr: np.ndarray    # [E, 7]
    y: np.ndarray            # [2]     robot CV residual (uses GLOBAL robot vel)
    meta: Dict[str, Any]


def augment_samples_flip_x(samples: List[GraphSample]) -> List[GraphSample]:
    """
    Mirror samples across the y-axis in robot frame by flipping x-related terms.
    """
    augmented_samples: List[GraphSample] = []
    for sample in samples:
        x_aug, edge_index_aug, edge_attr_aug, y_aug, meta_aug = flip_graph_sample_x(
            sample.x,
            sample.edge_index,
            sample.y,
            sample.meta,
        )
        augmented_samples.append(
            GraphSample(
                x=x_aug,
                edge_index=edge_index_aug,
                edge_attr=edge_attr_aug,
                y=y_aug,
                meta=meta_aug,
            )
        )
    return augmented_samples


# =========================
# Main loader
# =========================

def load_xml_graphs(
    xml_path: str,
    frame_rate: float = 60.0,
    k_horizon_frames: int = 1,
    d_thresh: float = 1.0,
    strict_less: bool = True,
) -> List[GraphSample]:
    """
    Load Unity XML and return graph samples with:
      - robot-frame pos, robot-relative vel as node features
      - edge features computed from those
      - robot CV residual label using GLOBAL robot position and velocity
      - filter frames by nearest pedestrian distance threshold

    Skips first k frames (cannot label).
    """
    dt = 1.0 / float(frame_rate)
    k = int(k_horizon_frames)
    if k < 1:
        raise ValueError("k_horizon_frames must be >= 1")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "outputInfo":
        # If your file uses a different root, change this check accordingly.
        raise ValueError(f"Expected root tag 'outputInfo', got '{root.tag}'")

    frames = list(root)
    if len(frames) <= k:
        raise ValueError(f"Need at least {k+1} frames, found {len(frames)}")

    # Parse all frames first (node features + robot global velocity per frame)
    X_all: List[np.ndarray] = [] # ndoe features
    Pg_all: List[np.ndarray] = [] # robot global position
    Vg_all: List[np.ndarray] = [] # robot global velocity
    for fr in frames:
        x, p_robot_glb, v_robot_glb = parse_frame_to_node_features(fr)
        X_all.append(x)
        Pg_all.append(p_robot_glb)
        Vg_all.append(v_robot_glb)

    samples: List[GraphSample] = []

    for t in range(k, len(frames)):
        x_curr = X_all[t] # node features
        p_curr_global = Pg_all[t]
        p_prev_global = Pg_all[t - k]
        v_prev_global = Vg_all[t - k]

        # Filter by nearest pedestrian distance (current frame)
        dmin = nearest_ped_distance(x_curr)
        keep = (dmin < d_thresh) if strict_less else (dmin <= d_thresh)
        if not keep:
            continue

        #edge_index, edge_attr = build_bidirectional_star(x_curr)
        edge_index, edge_attr = build_directional_star(x_curr)
        y = robot_cv_residual_label(
            p_curr_global=p_curr_global,
            p_prev_global=p_prev_global,
            v_prev_global=v_prev_global,
            dt=dt,
            k=k,
        )

        #if(t==562):
        #    print(t,p_curr_global,p_prev_global,v_prev_global,dt,k,y,p_prev_global+v_prev_global*dt*k)

        samples.append(
            GraphSample(
                x=x_curr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                meta={
                    "frame": frames[t].tag,
                    "t": t,
                    "k": k,
                    "dt": dt,
                    "dmin": dmin,
                },
            )
        )

    return samples


# =========================
# Script entry: sanity check + optional save
# =========================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="data/data_raw")
    ap.add_argument("--out_dir", type=str, default="data/data_processed")
    ap.add_argument("--augmented_out_dir", type=str, default="data/data_augmented")
    ap.add_argument("--xml_name", type=str, default="output4.xml")
    ap.add_argument("--frame_rate", type=float, default=60.0)
    ap.add_argument("--k_horizon_frames", type=int, default=15)
    ap.add_argument("--d_thresh", type=float, default=2.0)
    ap.add_argument("--strict_less", action="store_true", default=True)
    ap.add_argument("--non_strict_less", dest="strict_less", action="store_false")
    ap.add_argument(
        "--write_augmented",
        action="store_true",
        default=True,
        help="Also write y-axis mirrored datasets into augmented_out_dir.",
    )
    ap.add_argument(
        "--no_write_augmented",
        dest="write_augmented",
        action="store_false",
        help="Disable writing mirrored datasets.",
    )
    args = ap.parse_args()

    def natural_key(s: str):
        # Sort like: 1,2,10 instead of 1,10,2
        return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", s)]

    # 1) Discover candidate folders
    seq_folders = []
    for name in os.listdir(args.base_dir):
        full_path = os.path.join(args.base_dir, name)
        if not os.path.isdir(full_path):
            continue
        xml_path = os.path.join(full_path, args.xml_name)
        if os.path.exists(xml_path):
            seq_folders.append(name)

    seq_folders.sort(key=natural_key)

    print(f"Discovered {len(seq_folders)} sequences:")
    for name in seq_folders:
        print(" -", name)

    # 2) Process each discovered sequence
    all_datasets = {}  # folder_name -> samples
    kept_frames_total = 0

    for folder in seq_folders:
        xml_path = os.path.join(args.base_dir, folder, args.xml_name)
        print(f"\n=== Processing {folder} ===")

        samples = load_xml_graphs(
            xml_path=xml_path,
            frame_rate=args.frame_rate,
            k_horizon_frames=args.k_horizon_frames,
            d_thresh=args.d_thresh,
            strict_less=args.strict_less,
        )

        all_datasets[folder] = samples

        kept_frames = [s.meta["t"] for s in samples]
        kept_frames_total += len(kept_frames)
        print(f"Kept {len(kept_frames)} frames")
        #if samples:
        #    s = samples[0]
        #    print("First kept frame:", s.meta["frame"], "dmin:", s.meta["dmin"])
        #    print("y (robot CV residual):", s.y)

        #if kept_frames:
        #    print("First 20 kept frame indices:", kept_frames[:20])

        # 3) Save per-sequence output
        out_file = os.path.join(args.out_dir, f"gnn_dataset_{folder}.npz")
        save_graph_samples_npz(out_file, samples)
        print(f"Saved → {out_file}")

        if args.write_augmented:
            augmented_samples = augment_samples_flip_x(samples)
            augmented_out_file = os.path.join(
                args.augmented_out_dir,
                f"gnn_dataset_{folder}_flip_x.npz",
            )
            save_graph_samples_npz(augmented_out_file, augmented_samples)
            print(f"Saved augmented → {augmented_out_file}")
    print(f"\nKept {kept_frames_total} total frames")
