#!/usr/bin/env python3
"""
Visualize raw Unity trajectory data from a single XML file.

This script plots:
  - robot global trajectory
  - pedestrian global trajectories
  - CV-based predicted robot positions for kept frames
  - residual arrows from CV prediction to actual robot position

The plot uses global coordinates directly and does not transform into robot frame.
"""

from __future__ import annotations

import argparse
import os
from turtle import color
import xml.etree.ElementTree as ET
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def infer_num_nodes(attrib: dict) -> int:
    max_i = -1
    for k in attrib.keys():
        if k.startswith("x_"):
            try:
                idx = int(k.split("_", 1)[1])
            except ValueError:
                continue
            max_i = max(max_i, idx)
    if max_i < 0:
        raise ValueError("No node position keys found in XML frame.")
    return max_i + 1


def parse_global_frame(frame: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    attrib = frame.attrib
    V = infer_num_nodes(attrib)

    positions = np.zeros((V, 2), dtype=np.float32)
    velocities = np.zeros((V, 2), dtype=np.float32)

    for i in range(V):
        positions[i, 0] = float(attrib.get(f"x_{i}", 0.0))
        positions[i, 1] = float(attrib.get(f"y_{i}", 0.0))
        velocities[i, 0] = float(attrib.get(f"vx_{i}", 0.0))
        velocities[i, 1] = float(attrib.get(f"vy_{i}", 0.0))

    return positions, velocities


def load_global_sequence(
    xml_path: str,
    frame_rate: float = 60.0,
    k_horizon_frames: int = 15,
    d_thresh: float = 2.0,
    strict_less: bool = True,
):
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

    node_counts = [infer_num_nodes(fr.attrib) for fr in frames]
    if len(set(node_counts)) != 1:
        raise ValueError("Variable node count across frames is not supported.")
    V = node_counts[0]

    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    frame_names: List[str] = []

    for fr in frames:
        pos, vel = parse_global_frame(fr)
        positions.append(pos)
        velocities.append(vel)
        frame_names.append(fr.tag)

    robot_positions = np.stack([pos[0] for pos in positions], axis=0)
    robot_velocities = np.stack([vel[0] for vel in velocities], axis=0)
    pedestrian_positions = np.stack([pos[1:] for pos in positions], axis=0) if V > 1 else np.zeros((len(frames), 0, 2), dtype=np.float32)

    kept_indices: List[int] = []
    cv_pred_positions: List[np.ndarray] = []
    cv_residuals: List[np.ndarray] = []
    dmins: List[float] = []

    for t in range(k, len(frames)):
        robot_pos = robot_positions[t]
        if V > 1:
            ped_pos = pedestrian_positions[t]
            dmin = float(np.min(np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)))
        else:
            dmin = float("inf")

        dmins.append(dmin)
        keep = (dmin < d_thresh) if strict_less else (dmin <= d_thresh)
        if not keep:
            continue

        prev_pos = robot_positions[t - k]
        prev_vel = robot_velocities[t - k]
        predicted = prev_pos + prev_vel * (dt * k)
        residual = robot_pos - predicted

        kept_indices.append(t)
        cv_pred_positions.append(predicted)
        cv_residuals.append(residual)

    if len(kept_indices) == 0:
        raise ValueError(
            f"No frames kept after applying d_thresh={d_thresh} with strict_less={strict_less}."
        )

    return {
        "robot_positions": robot_positions,
        "pedestrian_positions": pedestrian_positions,
        "cv_pred_positions": np.stack(cv_pred_positions, axis=0),
        "cv_residuals": np.stack(cv_residuals, axis=0),
        "kept_indices": np.asarray(kept_indices, dtype=np.int32),
        "dmins": np.asarray(dmins, dtype=np.float32),
        "frame_names": frame_names,
        "dt": dt,
        "k": k,
    }


def plot_global_trajectory(
    robot_positions: np.ndarray,
    pedestrian_positions: np.ndarray,
    cv_pred_positions: np.ndarray,
    cv_residuals: np.ndarray,
    kept_indices: np.ndarray,
    xml_path: str,
    save_path: str = "visualize_data.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        robot_positions[:, 0],
        robot_positions[:, 1],
        marker=".",
        linewidth = 0.1,
        linestyle="-",
        color="black",
        label="robot actual",
    )

    #print(pedestrian_positions.shape)
    #print(kept_indices)
    if pedestrian_positions.shape[1] > 0:
        for pid in range(pedestrian_positions.shape[1]):
            ped = pedestrian_positions[:, pid, :]
            ax.plot(
                ped[:, 0],
                ped[:, 1],
                linestyle="--",
                marker=".",
                linewidth=1.0,
                alpha=0.7,
                label=f"pedestrian {pid + 1}",
            )
        #if pedestrian_positions.shape[1] > 1:
        #    ax.plot([], [], linestyle="--", color="black", alpha=0.7, label="other pedestrians")

    ax.scatter(
        cv_pred_positions[:, 0],
        cv_pred_positions[:, 1],
        marker=".",
        linestyle="--",
        color="tab:grey",
        label="CV prediction",
    )

    for pred, t in zip(cv_pred_positions, kept_indices):
        actual = robot_positions[t]
        ax.annotate(
            "",
            xy=actual,
            xytext=pred,
            arrowprops=dict(arrowstyle="->", color="tab:grey", alpha=0.7, linewidth=1.2),
        )

    ax.scatter(
        robot_positions[kept_indices, 0],
        robot_positions[kept_indices, 1],
        marker="o",
        color="tab:red",
        linewidth = 0.1,
        label="kept robot frames",
    )

    ax.set_title(f"Trajectory visualization for {os.path.basename(xml_path)}")
    ax.set_xlabel("x (global)")
    ax.set_ylabel("y (global)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved trajectory visualization to {save_path}")
    plt.show()


def resolve_xml_path(input_path: str, xml_name: str) -> str:
    if os.path.isdir(input_path):
        candidate = os.path.join(input_path, xml_name)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            f"Directory provided but {xml_name} was not found in {input_path}."
        )
    if os.path.isfile(input_path):
        return input_path
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw XML trajectory data.")
    parser.add_argument("--input",type=str,default="data/data_raw/1",help="Path to a raw XML file or to a data_raw sequence directory.",)
    parser.add_argument("--xml-name",type=str,default="output4.xml",help="XML filename to use when input is a directory.",)
    parser.add_argument("--frame-rate", type=float, default=60.0, help="Frame rate used for CV prediction.")
    parser.add_argument("--k_horizon_frames", type=int, default=60, help="CV horizon in frames.")#60
    parser.add_argument("--d_thresh", type=float, default=2.5, help="Distance threshold to keep a frame.")
    parser.add_argument("--strict-less",action="store_true",default=True,help="Keep only frames with d_min < d_thresh.",)
    parser.add_argument("--non-strict-less",dest="strict_less",action="store_false",help="Keep frames with d_min <= d_thresh.",)
    parser.add_argument("--save-path",type=str,default="visualize_data.png",help="Output plot file path.",)
    args = parser.parse_args()

    xml_path = resolve_xml_path(args.input, args.xml_name)

    sequence = load_global_sequence(
        xml_path=xml_path,
        frame_rate=args.frame_rate,
        k_horizon_frames=args.k_horizon_frames,
        d_thresh=args.d_thresh,
        strict_less=args.strict_less,
    )

    plot_global_trajectory(
        robot_positions=sequence["robot_positions"],
        pedestrian_positions=sequence["pedestrian_positions"],
        cv_pred_positions=sequence["cv_pred_positions"],
        cv_residuals=sequence["cv_residuals"],
        kept_indices=sequence["kept_indices"],
        xml_path=xml_path,
        save_path=args.save_path,
    )

# a standalone script that visualizes raw Unity trajectory data from a single XML file, plotting the robot's global trajectory, pedestrian trajectories, CV-based predicted robot positions, and residual arrows from CV prediction to actual robot position, while allowing filtering based on nearest pedestrian distance.
if __name__ == "__main__":
    main()
