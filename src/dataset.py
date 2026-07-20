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
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Iterator, Optional
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


FEATURE_NAMES = ["x_rel", "y_rel", "vx_rel", "vy_rel", "is_robot"]
LABEL_NAMES = ["dx", "dy"]


def should_keep_robot_cv_residual(
    residual: np.ndarray,
    filter_angle_deg: float,
) -> bool:
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm == 0.0:
        return True

    x_axis = np.array([1.0, 0.0], dtype=np.float32)
    cos_theta = float(np.dot(residual, x_axis) / residual_norm)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))

    return filter_angle_deg <= angle_deg <= (180.0 - filter_angle_deg)


def augment_samples_flip_y(samples: List[GraphSample]) -> List[GraphSample]:
    """
    Mirror samples across the y-axis in robot frame by flipping y-related terms.
    """
    augmented_samples: List[GraphSample] = []
    for sample in samples:
        x_aug, edge_index_aug, edge_attr_aug, y_aug, meta_aug = flip_graph_sample_y(
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


def compute_node_feature_stats(samples: List[GraphSample]) -> Dict[str, Dict[str, float]]:
    """
    Compute min/max for each node feature across all nodes in all samples.
    """
    if not samples:
        return {}

    all_x = np.concatenate([sample.x for sample in samples], axis=0)
    feature_mins = np.min(all_x, axis=0)
    feature_maxs = np.max(all_x, axis=0)

    stats: Dict[str, Dict[str, float]] = {}
    for idx, feature_name in enumerate(FEATURE_NAMES):
        stats[feature_name] = {
            "min": float(feature_mins[idx]),
            "max": float(feature_maxs[idx]),
        }
    return stats


def compute_label_stats(samples: List[GraphSample]) -> Dict[str, Dict[str, float]]:
    """
    Compute min/max for each label dimension across all samples.
    """
    if not samples:
        return {}

    all_y = np.stack([sample.y for sample in samples], axis=0)
    label_mins = np.min(all_y, axis=0)
    label_maxs = np.max(all_y, axis=0)

    stats: Dict[str, Dict[str, float]] = {}
    for idx, label_name in enumerate(LABEL_NAMES):
        stats[label_name] = {
            "min": float(label_mins[idx]),
            "max": float(label_maxs[idx]),
        }
    return stats


def init_feature_stats_accumulator() -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.full(len(FEATURE_NAMES), np.inf, dtype=np.float64),
        np.full(len(FEATURE_NAMES), -np.inf, dtype=np.float64),
    )


def init_label_stats_accumulator() -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.full(len(LABEL_NAMES), np.inf, dtype=np.float64),
        np.full(len(LABEL_NAMES), -np.inf, dtype=np.float64),
    )


def update_feature_stats_accumulator(
    feature_mins: np.ndarray,
    feature_maxs: np.ndarray,
    x: np.ndarray,
) -> None:
    if x.size == 0:
        return

    feature_mins[:] = np.minimum(feature_mins, np.min(x, axis=0))
    feature_maxs[:] = np.maximum(feature_maxs, np.max(x, axis=0))


def update_label_stats_accumulator(
    label_mins: np.ndarray,
    label_maxs: np.ndarray,
    y: np.ndarray,
) -> None:
    if y.size == 0:
        return

    label_mins[:] = np.minimum(label_mins, y)
    label_maxs[:] = np.maximum(label_maxs, y)


def feature_stats_from_accumulator(
    feature_mins: np.ndarray,
    feature_maxs: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    if np.isinf(feature_mins).any() or np.isinf(feature_maxs).any():
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    for idx, feature_name in enumerate(FEATURE_NAMES):
        stats[feature_name] = {
            "min": float(feature_mins[idx]),
            "max": float(feature_maxs[idx]),
        }
    return stats


def label_stats_from_accumulator(
    label_mins: np.ndarray,
    label_maxs: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    if np.isinf(label_mins).any() or np.isinf(label_maxs).any():
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    for idx, label_name in enumerate(LABEL_NAMES):
        stats[label_name] = {
            "min": float(label_mins[idx]),
            "max": float(label_maxs[idx]),
        }
    return stats


def mirror_y_feature_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not stats:
        return {}

    mirrored = {
        feature_name: {
            "min": float(feature_stats["min"]),
            "max": float(feature_stats["max"]),
        }
        for feature_name, feature_stats in stats.items()
    }
    for feature_name in ("y_rel", "vy_rel"):
        if feature_name in mirrored:
            old_min = mirrored[feature_name]["min"]
            old_max = mirrored[feature_name]["max"]
            mirrored[feature_name]["min"] = -old_max
            mirrored[feature_name]["max"] = -old_min
    return mirrored


def mirror_y_label_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not stats:
        return {}

    mirrored = {
        label_name: {
            "min": float(label_stats["min"]),
            "max": float(label_stats["max"]),
        }
        for label_name, label_stats in stats.items()
    }
    if "dy" in mirrored:
        old_min = mirrored["dy"]["min"]
        old_max = mirrored["dy"]["max"]
        mirrored["dy"]["min"] = -old_max
        mirrored["dy"]["max"] = -old_min
    return mirrored


def normalize_node_features(
    x: np.ndarray,
    stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    if not stats:
        raise ValueError("Node feature stats are required for normalization.")

    x_norm = np.asarray(x, dtype=np.float32).copy()
    for idx, feature_name in enumerate(FEATURE_NAMES):
        if feature_name == "is_robot":
            continue

        feature_stats = stats.get(feature_name)
        if feature_stats is None:
            raise KeyError(f"Missing stats for node feature '{feature_name}'")

        f_min = float(feature_stats["min"])
        f_max = float(feature_stats["max"])
        scale = max(abs(f_min), abs(f_max))
        if scale < 1e-12:
            x_norm[:, idx] = 0.0
            continue

        x_norm[:, idx] = x_norm[:, idx] / scale

    return x_norm


def normalize_label(
    y: np.ndarray,
    stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    if not stats:
        raise ValueError("Label stats are required for normalization.")

    y_norm = np.asarray(y, dtype=np.float32).copy()
    for idx, label_name in enumerate(LABEL_NAMES):
        label_stats = stats.get(label_name)
        if label_stats is None:
            raise KeyError(f"Missing stats for label '{label_name}'")

        y_min = float(label_stats["min"])
        y_max = float(label_stats["max"])
        scale = max(abs(y_min), abs(y_max))
        if scale < 1e-12:
            y_norm[idx] = 0.0
            continue

        y_norm[idx] = y_norm[idx] / scale

    return y_norm


def print_stats(dataset_name: str, stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print min/max for each node feature.
    """
    print(f"\n=== Stats: {dataset_name} ===")
    if not stats:
        print("No samples available.")
        return

    for feature_name, feature_stats in stats.items():
        print(
            f"{feature_name:>8}: min={feature_stats['min']: .6f}, "
            f"max={feature_stats['max']: .6f}"
        )


def save_dataset_stats_json(
    json_path: str,
    processed_node_stats: Dict[str, Dict[str, float]],
    augmented_node_stats: Dict[str, Dict[str, float]],
    processed_label_stats: Dict[str, Dict[str, float]],
    augmented_label_stats: Dict[str, Dict[str, float]],
) -> None:
    """
    Save node and label ranges to JSON for later normalization.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    payload = {
        "data_processed": processed_node_stats,
        "data_augmented": augmented_node_stats,
        "label_processed": processed_label_stats,
        "label_augmented": augmented_label_stats,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved dataset stats JSON → {json_path}")


def filter_nodes_by_radius(
    x: np.ndarray,
    node_inclusion_radius: float,
) -> np.ndarray:
    """
    Keep robot node 0 and keep only non-robot nodes within the given radius.
    This radius applies to both pedestrians and the target node.
    """
    if x.shape[0] <= 1:
        return x

    other_positions = x[1:, 0:2]
    other_distances = np.linalg.norm(other_positions, axis=1)
    other_keep_indices = np.where(other_distances <= node_inclusion_radius)[0] + 1

    keep_indices = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            other_keep_indices.astype(np.int64),
        ]
    )
    return x[keep_indices]


def iter_kept_frame_data(
    xml_path: str,
    frame_rate: float = 60.0,
    k_horizon_frames: int = 1,
    d_thresh: float = 1.0,
    node_inclusion_radius: float = 4.0,
    strict_less: bool = True,
) -> Iterator[Tuple[ET.Element, int, int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yield the filtered per-frame data needed for both stats collection and
    final graph construction.
    """
    dt = 1.0 / float(frame_rate)
    k = int(k_horizon_frames)
    if k < 1:
        raise ValueError("k_horizon_frames must be >= 1")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "outputInfo":
        raise ValueError(f"Expected root tag 'outputInfo', got '{root.tag}'")

    frames = list(root)
    if len(frames) <= k:
        raise ValueError(f"Need at least {k+1} frames, found {len(frames)}")

    X_all: List[np.ndarray] = []
    Pg_all: List[np.ndarray] = []
    Vg_all: List[np.ndarray] = []
    for fr in frames:
        x, p_robot_glb, v_robot_glb = parse_frame_to_node_features(fr)
        X_all.append(x)
        Pg_all.append(p_robot_glb)
        Vg_all.append(v_robot_glb)

    for t in range(k, len(frames)):
        x_curr = X_all[t]
        p_curr_global = Pg_all[t]
        p_prev_global = Pg_all[t - k]
        v_prev_global = Vg_all[t - k]

        dmin = nearest_ped_distance(x_curr)
        keep = (dmin < d_thresh) if strict_less else (dmin <= d_thresh)
        if not keep:
            continue

        x_curr = filter_nodes_by_radius(x_curr, node_inclusion_radius)
        yield (
            frames[t],
            t,
            k,
            dt,
            dmin,
            x_curr,
            p_curr_global,
            p_prev_global,
            v_prev_global,
        )


def compute_node_feature_stats_from_xml_paths(
    xml_paths: List[str],
    frame_rate: float = 60.0,
    k_horizon_frames: int = 1,
    d_thresh: float = 1.0,
    node_inclusion_radius: float = 4.0,
    strict_less: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute node feature min/max directly from the filtered robot-frame node
    features, before any normalization or edge construction.
    """
    feature_mins, feature_maxs = init_feature_stats_accumulator()
    for xml_path in xml_paths:
        for _, _, _, _, _, x_curr, _, _, _ in iter_kept_frame_data(
            xml_path=xml_path,
            frame_rate=frame_rate,
            k_horizon_frames=k_horizon_frames,
            d_thresh=d_thresh,
            node_inclusion_radius=node_inclusion_radius,
            strict_less=strict_less,
        ):
            update_feature_stats_accumulator(feature_mins, feature_maxs, x_curr)

    return feature_stats_from_accumulator(feature_mins, feature_maxs)


def compute_label_stats_from_xml_paths(
    xml_paths: List[str],
    frame_rate: float = 60.0,
    k_horizon_frames: int = 1,
    d_thresh: float = 1.0,
    node_inclusion_radius: float = 4.0,
    strict_less: bool = True,
    filter_angle: float = 45.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compute label min/max directly from the filtered frames before any label
    normalization.
    """
    label_mins, label_maxs = init_label_stats_accumulator()
    for xml_path in xml_paths:
        for _, _, k, dt, _, _, p_curr_global, p_prev_global, v_prev_global in iter_kept_frame_data(
            xml_path=xml_path,
            frame_rate=frame_rate,
            k_horizon_frames=k_horizon_frames,
            d_thresh=d_thresh,
            node_inclusion_radius=node_inclusion_radius,
            strict_less=strict_less,
        ):
            y = robot_cv_residual_label(
                p_curr_global=p_curr_global,
                p_prev_global=p_prev_global,
                v_prev_global=v_prev_global,
                dt=dt,
                k=k,
            )
            if not should_keep_robot_cv_residual(y, filter_angle):
                continue
            update_label_stats_accumulator(label_mins, label_maxs, y)

    return label_stats_from_accumulator(label_mins, label_maxs)


# =========================
# Main loader
# =========================

def load_xml_graphs(
    xml_path: str,
    frame_rate: float = 60.0,
    k_horizon_frames: int = 1,
    d_thresh: float = 1.0,
    node_inclusion_radius: float = 4.0,
    strict_less: bool = True,
    filter_angle: float = 45.0,
    node_feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
    label_stats: Optional[Dict[str, Dict[str, float]]] = None,
    normalize_nodes: bool = False,
    normalize_labels: bool = False,
) -> List[GraphSample]:
    """
    Load Unity XML and return graph samples with:
      - robot-frame pos, robot-relative vel as node features
      - optional signed normalization on node features before edge construction
      - optional signed normalization on the 2D label
      - edge features computed from the node features actually fed to the model
      - robot CV residual label using GLOBAL robot position and velocity
      - filter frames by nearest pedestrian distance threshold
      - keep only non-robot nodes within node_inclusion_radius

    Skips first k frames (cannot label). When normalization is enabled, stats
    must come from a separate first pass over the same robot-frame features.
    Each continuous feature is scaled by max(abs(min), abs(max)) so its sign
    is preserved. The discrete is_robot channel is never normalized.
    """
    if normalize_nodes and node_feature_stats is None:
        raise ValueError("node_feature_stats must be provided when normalize_nodes=True")
    if normalize_labels and label_stats is None:
        raise ValueError("label_stats must be provided when normalize_labels=True")

    samples: List[GraphSample] = []

    for frame, t, k, dt, dmin, x_curr_raw, p_curr_global, p_prev_global, v_prev_global in iter_kept_frame_data(
        xml_path=xml_path,
        frame_rate=frame_rate,
        k_horizon_frames=k_horizon_frames,
        d_thresh=d_thresh,
        node_inclusion_radius=node_inclusion_radius,
        strict_less=strict_less,
    ):
        x_curr = x_curr_raw
        if normalize_nodes:
            x_curr = normalize_node_features(
                x_curr,
                node_feature_stats,
            )

        edge_index, edge_attr = build_bidirectional_star(x_curr)
        y = robot_cv_residual_label(
            p_curr_global=p_curr_global,
            p_prev_global=p_prev_global,
            v_prev_global=v_prev_global,
            dt=dt,
            k=k,
        )
        if not should_keep_robot_cv_residual(y, filter_angle):
            continue
        if normalize_labels:
            y = normalize_label(y, label_stats)

        #if(t==562):
        #    print(t,p_curr_global,p_prev_global,v_prev_global,dt,k,y,p_prev_global+v_prev_global*dt*k)

        samples.append(
            GraphSample(
                x=x_curr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                meta={
                    "frame": frame.tag,
                    "xml_path": xml_path,
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
    ap.add_argument("--base_dir", type=str, default="data/data_raw_goal")
    ap.add_argument("--out_dir", type=str, default="data/data_processed")
    ap.add_argument("--augmented_out_dir", type=str, default="data/data_augmented")
    ap.add_argument("--xml_name", type=str, default="output4.xml")
    ap.add_argument("--frame_rate", type=float, default=60.0)
    ap.add_argument("--k_horizon_frames", type=int, default=15) # 60
    ap.add_argument("--d_thresh", type=float, default=2.5) # 2.5 
    ap.add_argument("--node_inclusion_radius", type=float, default=4.0) # 4.0
    ap.add_argument("--strict_less", action="store_true", default=True)
    ap.add_argument("--non_strict_less", dest="strict_less", action="store_false")
    ap.add_argument("--filter_angle",type=float,default=45.0,help="Hide residual labels whose angle to the robot x-axis [1, 0] is within this many degrees of 0 or 180 in robot frame.",)
    ap.add_argument("--write_augmented", action="store_true", default=True, help="Also write y-axis mirrored datasets into augmented_out_dir.",)
    ap.add_argument("--no_write_augmented",dest="write_augmented",action="store_false",help="Disable writing mirrored datasets.",)
    ap.add_argument("--normalize_node_features", action="store_true", default=False, help="Run a first pass to compute node feature stats and save datasets with signed, max-abs normalized node and edge features.",)
    ap.add_argument("--normalize_labels", action="store_true", default=False, help="Run a first pass to compute 2D label stats and save datasets with signed, max-abs normalized labels.",)
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
    # for name in seq_folders:
    #     print(" -", name)

    xml_paths = [os.path.join(args.base_dir, folder, args.xml_name) for folder in seq_folders]
    raw_processed_stats = compute_node_feature_stats_from_xml_paths(
        xml_paths=xml_paths,
        frame_rate=args.frame_rate,
        k_horizon_frames=args.k_horizon_frames,
        d_thresh=args.d_thresh,
        node_inclusion_radius=args.node_inclusion_radius,
        strict_less=args.strict_less,
    )
    raw_augmented_stats = mirror_y_feature_stats(raw_processed_stats)
    raw_processed_label_stats = compute_label_stats_from_xml_paths(
        xml_paths=xml_paths,
        frame_rate=args.frame_rate,
        k_horizon_frames=args.k_horizon_frames,
        d_thresh=args.d_thresh,
        node_inclusion_radius=args.node_inclusion_radius,
        strict_less=args.strict_less,
        filter_angle=args.filter_angle,
    )
    raw_augmented_label_stats = mirror_y_label_stats(raw_processed_label_stats)

    # 2) Process each discovered sequence
    all_datasets = {}  # folder_name -> samples
    all_processed_samples: List[GraphSample] = []
    all_augmented_samples: List[GraphSample] = []
    kept_frames_total = 0
    stats_json_path = os.path.join("data", "node_feature_stats.json")

    for folder in seq_folders:
        xml_path = os.path.join(args.base_dir, folder, args.xml_name)
        print(f"=== Processing {folder} ===")

        samples = load_xml_graphs(
            xml_path=xml_path,
            frame_rate=args.frame_rate,
            k_horizon_frames=args.k_horizon_frames,
            d_thresh=args.d_thresh,
            node_inclusion_radius=args.node_inclusion_radius,
            strict_less=args.strict_less,
            filter_angle=args.filter_angle,
            node_feature_stats=raw_processed_stats if args.normalize_node_features else None,
            label_stats=raw_processed_label_stats if args.normalize_labels else None,
            normalize_nodes=args.normalize_node_features,
            normalize_labels=args.normalize_labels,
        )

        all_datasets[folder] = samples
        all_processed_samples.extend(samples)

        kept_frames = [s.meta["t"] for s in samples]
        kept_frames_total += len(kept_frames)
        #print(f"Kept {len(kept_frames)} frames")
        #if samples:
        #    s = samples[0]
        #    print("First kept frame:", s.meta["frame"], "dmin:", s.meta["dmin"])
        #    print("y (robot CV residual):", s.y)

        #if kept_frames:
        #    print("First 20 kept frame indices:", kept_frames[:20])

        # 3) Save per-sequence output
        out_file = os.path.join(args.out_dir, f"gnn_dataset_{folder}.npz")
        save_graph_samples_npz(out_file, samples)
        #print(f"Saved → {out_file}")

        if args.write_augmented:
            # 4) Create and save augmented dataset
            augmented_samples = augment_samples_flip_y(samples)
            all_augmented_samples.extend(augmented_samples)
            augmented_out_file = os.path.join(
                args.augmented_out_dir,
                f"gnn_dataset_{folder}_flip_y.npz",
            )
            save_graph_samples_npz(augmented_out_file, augmented_samples)
            #print(f"Saved augmented → {augmented_out_file}")
    print(f"\nKept {kept_frames_total} total frames")
    saved_processed_stats = compute_node_feature_stats(all_processed_samples)
    saved_processed_label_stats = compute_label_stats(all_processed_samples)
    augmented_stats = compute_node_feature_stats(all_augmented_samples)
    augmented_label_stats = compute_label_stats(all_augmented_samples)
    print_stats("node_processed_raw", raw_processed_stats)
    print_stats("node_processed_saved", saved_processed_stats)
    print_stats("label_processed_raw", raw_processed_label_stats)
    print_stats("label_processed_saved", saved_processed_label_stats)
    if args.write_augmented:
        print_stats("node_augmented_saved", augmented_stats)
        print_stats("label_augmented_saved", augmented_label_stats)
    save_dataset_stats_json(
        stats_json_path,
        raw_processed_stats,
        raw_augmented_stats,
        raw_processed_label_stats,
        raw_augmented_label_stats,
    )
