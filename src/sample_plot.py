#!/usr/bin/env python3
"""
Plot trajectory with predicted offsets.

This script takes an array of robot trajectory points and predicts per-frame
offsets using the GNN model, then plots both the original trajectory and the
offset-adjusted trajectory for comparison.

Example:
  python3 src/trajectory_offset_plot.py --model-path gnn_model.pt
"""

from __future__ import annotations

import argparse
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from torch_geometric.data import Data

from utils import (
    build_directional_star,
    load_model,
    preprocess_frame_to_node_features,
    build_bidirectional_star,
)


def sample_trajectory_from_waypoints(
    waypoints: np.ndarray,
    samples_per_segment: int = 2,
) -> np.ndarray:
    """
    Linearly sample points along each trajectory segment.

    Each segment contributes exactly `samples_per_segment` points without
    duplicating junction waypoints; the final waypoint is appended once.
    """
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints must have shape [N, 2]")
    if len(waypoints) < 2:
        raise ValueError("need at least 2 waypoints to form segments")
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be > 0")

    sampled_points: List[np.ndarray] = []
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        t_values = np.linspace(0.0, 1.0, num=samples_per_segment, endpoint=False, dtype=np.float32)
        segment_points = (1.0 - t_values[:, None]) * p0 + t_values[:, None] * p1
        sampled_points.append(segment_points)

    sampled_points.append(waypoints[-1][None, :])
    return np.vstack(sampled_points).astype(np.float32)


def predict_offset_at_frame(
    frame_nx5: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    fallback_theta: Optional[float] = None,
) -> np.ndarray:
    """
    Predict the offset for a single frame with pedestrians.
    
    Args:
        frame_nx5: [V, 5] array with robot at index 0
        model: Trained GNN model
        device: Torch device
    
    Returns:
        [2] array with predicted offset (global frame)
    """
    x_node = preprocess_frame_to_node_features(frame_nx5)
    #edge_index, edge_attr = build_directional_star(x_node)
    edge_index, edge_attr = build_bidirectional_star(x_node)
    
    data = Data(
        x=torch.from_numpy(x_node),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )
    
    with torch.no_grad():
        y_hat_local = model(data.to(device)).cpu().numpy().reshape(-1)
    
    # Transform local offset to global
    robot_vx, robot_vy = frame_nx5[0, 2], frame_nx5[0, 3]
    speed = np.sqrt(robot_vx**2 + robot_vy**2)
    
    if speed < 0.5: # to suppress noise when robot is nearly stationary
        return np.zeros(2, dtype=np.float32)
    if speed > 1e-6:
        theta = np.arctan2(robot_vy, robot_vx)
    elif fallback_theta is not None:
        theta = fallback_theta
    else:
        theta = None

    if theta is not None:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_theta = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        y_hat = R_theta @ y_hat_local
    else:
        y_hat = y_hat_local
    
    #print("------------------------------------------------------------------------------")
    #print("data",data.x.numpy(),data.edge_index.numpy(),data.edge_attr.numpy())
    #print(f"Predicted CV residual [dx, dy] (local): {y_hat_local.tolist()}")
    #print(f"Predicted CV residual [dx, dy] (global): {y_hat.tolist()}")
    
    return y_hat


def plot_trajectory_with_offset(
    robot_positions: np.ndarray,
    offset_positions: np.ndarray,
    pedestrian_positions: Optional[np.ndarray] = None,
    pedestrian_velocities: Optional[np.ndarray] = None,
    save_path: str = "trajectory_offset_plot.png",
    title: str = "Trajectory with Predicted Offsets",
) -> None:
    """
    Plot original robot trajectory and offset-adjusted trajectory with pedestrians.
    
    Args:
        robot_positions: [N, 2] array of robot positions
        offset_positions: [N, 2] array of offset-adjusted positions
        pedestrian_positions: [M, 2] array of pedestrian positions
        pedestrian_velocities: [M, 2] array of pedestrian velocities
        save_path: Output file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot original trajectory
    if len(robot_positions) > 0:
        ax.plot(
            robot_positions[:, 0],
            robot_positions[:, 1],
            color="black",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="original trajectory",
        )
        ax.scatter(
            robot_positions[-1, 0],
            robot_positions[-1, 1],
            c="black",
            s=100,
            marker="o",
            zorder=5,
        )
    
    # Plot offset trajectory
    if len(offset_positions) > 0:
        ax.plot(
            offset_positions[:, 0],
            offset_positions[:, 1],
            color="tab:red",
            linewidth=2.0,
            linestyle="--",
            marker="x",
            markersize=4,
            label="offset trajectory",
        )
        ax.scatter(
            offset_positions[-1, 0],
            offset_positions[-1, 1],
            c="tab:red",
            s=100,
            marker="x",
            zorder=5,
        )
    
    # Plot pedestrians
    if pedestrian_positions is not None and len(pedestrian_positions) > 0:
        ped_vels = pedestrian_velocities if pedestrian_velocities is not None else np.zeros_like(pedestrian_positions)
        for idx, (ped_pos, ped_vel) in enumerate(zip(pedestrian_positions, ped_vels)):
            ax.scatter(ped_pos[0], ped_pos[1], c="tab:green", s=80, zorder=3)
            if pedestrian_velocities is not None and np.linalg.norm(ped_vel) > 1e-6:
                ax.arrow(
                    ped_pos[0],
                    ped_pos[1],
                    ped_vel[0],
                    ped_vel[1],
                    color="tab:green",
                    width=0.01,
                    length_includes_head=True,
                    head_width=0.08,
                    zorder=3,
                )
            ax.text(ped_pos[0], ped_pos[1], f" ped{idx}", color="tab:green", fontsize=9, va="bottom")
    
    # Compute bounds
    all_points = [robot_positions, offset_positions]
    if pedestrian_positions is not None:
        all_points.append(pedestrian_positions)
    all_points = np.vstack(all_points)
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    half_range = 0.5 * np.max(max_xy - min_xy) + 0.5
    
    ax.set_xlim(center_xy[0] - half_range, center_xy[0] + half_range)
    ax.set_ylim(center_xy[1] - half_range, center_xy[1] + half_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    
    # Build legend
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, marker="o", label="original trajectory"),
        Line2D([0], [0], color="tab:red", linewidth=2.0, linestyle="--", marker="x", label="offset trajectory"),
        Line2D([0], [0], color="tab:green", marker="o", linestyle="-", label="pedestrian"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="gnn_model.pt", help="Path to trained GNN model.")
    ap.add_argument("--save-path", type=str, default="sample_plot.png", help="Output plot file path.")
    ap.add_argument("--title", type=str, default="Trajectory with Predicted Offsets", help="Plot title.")
    args = ap.parse_args()

    # ============================================================
    # Example waypoints: sparse trajectory control points.
    # Only edit waypoints; dense trajectory points are auto-sampled below.
    # ============================================================
    robot_waypoints = np.array([
        [0.0, -5.0],
        [0.0, 5.0],
    ], dtype=np.float32)
    robot_trajectory = sample_trajectory_from_waypoints(robot_waypoints, samples_per_segment=100)

    # ============================================================
    # Simple pedestrian positions (fixed for demonstration)
    # ============================================================
    
    # test case 1: left-down
    pedestrian_positions = np.array([[-0.5, 0.0]], dtype=np.float32)
    pedestrian_velocities = np.array([[0.0, -2.0]], dtype=np.float32)
    # # test case 2: right-down
    # pedestrian_positions = np.array([[1.5, 0.0]], dtype=np.float32)
    # pedestrian_velocities = np.array([[0.0, -2.0]], dtype=np.float32)
    # test case 3: left-up
    # pedestrian_positions = np.array([[-1.5, 0.0]], dtype=np.float32)
    # pedestrian_velocities = np.array([[0.0, 2.0]], dtype=np.float32)
    # test case 4: right-up
    # pedestrian_positions = np.array([[1.5, 0.0]], dtype=np.float32)
    # pedestrian_velocities = np.array([[0.0, 2.0]], dtype=np.float32)
    # test case 5: left-right
    # pedestrian_positions = np.array([[-1.5, 0.0]], dtype=np.float32)
    # pedestrian_velocities = np.array([[2.0, 0.0]], dtype=np.float32)
    # # test case 6: right-left
    # pedestrian_positions = np.array([[1.5, 0.0]], dtype=np.float32)
    # pedestrian_velocities = np.array([[-2.0, 0.0]], dtype=np.float32)
    # test case 7: left-left
    # pedestrian_positions = np.array([[-1.5, 0.0]], dtype=np.float32)
    # pedestrian_velocities = np.array([[-2.0, 0.0]], dtype=np.float32)
    # test case 8: right-right
    # pedestrian_positions = np.array([[1.5, 0.0]], dtype=np.float32)
    # pedestrian_velocities = np.array([[2.0, 0.0]], dtype=np.float32)

    # ============================================================
    # Load model
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path)
    model.to(device)
    model.eval()

    # ============================================================
    # Predict offsets for each robot position
    # ============================================================
    offset_trajectory = []
    last_valid_theta: Optional[float] = None
    
    for i, robot_pos in enumerate(robot_trajectory):
        # Compute robot velocity direction from consecutive positions,
        # but fix the speed magnitude to 0.4 m/s.
        if i == 0:
            direction = robot_trajectory[i+1] - robot_trajectory[i]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                robot_vel = 2.0 * (direction / direction_norm) #2.0
            else:   
                robot_vel = np.array([0.0, 0.0], dtype=np.float32)
        else:
            direction = robot_trajectory[i] - robot_trajectory[i-1]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                robot_vel = 2.0 * (direction / direction_norm) #2.0
            else:
                robot_vel = np.array([0.0, 0.0], dtype=np.float32)

        speed = np.linalg.norm(robot_vel)
        if speed > 1e-6:
            last_valid_theta = float(np.arctan2(robot_vel[1], robot_vel[0]))
        
        # Build frame: robot + pedestrians
        frame_nodes = [[robot_pos[0], robot_pos[1], robot_vel[0], robot_vel[1], 1.0]]
        for ped_pos, ped_vel in zip(pedestrian_positions, pedestrian_velocities):
            frame_nodes.append([ped_pos[0], ped_pos[1], ped_vel[0], ped_vel[1], 0.0])
        
        frame_nx5 = np.array(frame_nodes, dtype=np.float32)
        
        # Predict offset
        try:
            predicted_offset = predict_offset_at_frame(
                frame_nx5,
                model,
                device,
                fallback_theta=last_valid_theta,
            )
            offset_pos = robot_pos + predicted_offset
            offset_trajectory.append(offset_pos)
            #print(f"Frame {i}: robot={robot_pos}, offset={predicted_offset}, offset_pos={offset_pos}")
        except Exception as e:
            #print(f"Frame {i}: prediction failed ({e}), using zero offset")
            offset_trajectory.append(robot_pos)
    
    offset_trajectory = np.array(offset_trajectory, dtype=np.float32)

    # ============================================================
    # Plot
    # ============================================================
    plot_trajectory_with_offset(
        robot_trajectory,
        offset_trajectory,
        pedestrian_positions=pedestrian_positions,
        pedestrian_velocities=pedestrian_velocities,
        save_path=args.save_path,
        title=args.title,
    )

# a standalone script that samples a robot trajectory from waypoints, simulates pedestrian positions and velocities, loads a pre-trained GNN model, predicts per-frame offsets for the robot, and visualizes the original and offset-adjusted trajectories along with pedestrians.
if __name__ == "__main__":
    main()
