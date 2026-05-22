#!/usr/bin/env python3
"""
Live sensor/pedestrian plotting with predicted offset trajectory.

This script is based on speed_plot.py, but it additionally predicts a per-frame
sensor offset using the GNN model and draws an offset version of the sensor
trajectory.

Example:
  python3 src/offset_plot.py --model-path gnn_model.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rospy
from matplotlib.figure import Figure
from torch_geometric.data import Data
import torch
from visualization_msgs.msg import Marker, MarkerArray

from utils import (
    build_directional_star,
    load_model,
    preprocess_frame_to_node_features,
)

Point3 = Tuple[float, float, float]  # (x, y, timestamp)


class OffsetPlotter:
    def __init__(
        self,
        sensor_topic: str,
        ped_topic: str,
        model_path: str,
        save_path: str,
        title: str,
        max_pedestrians: int,
        refresh_hz: float,
        max_time_gap: float,
        max_distance_gap: float,
        min_sensor_distance: float,
        show_frame: Optional[int],
    ):
        self.sensor_topic = sensor_topic
        self.ped_topic = ped_topic
        self.save_path = save_path
        self.title = title
        self.max_pedestrians = max_pedestrians
        self.refresh_hz = refresh_hz
        self.max_time_gap = max_time_gap
        self.max_distance_gap = max_distance_gap
        self.min_sensor_distance = min_sensor_distance
        self.show_frame = show_frame
        self.data_lock = Lock()

        self.sensor_positions: List[Point3] = []
        self.pedestrian_tracks: Dict[str, Dict[str, Any]] = {}
        self.offset_positions: List[Tuple[float, float]] = []
        self.last_predicted_offset: Optional[np.ndarray] = None
        self.last_sensor_index = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path).to(self.device)
        self.model.eval()

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.ion()
        self.fig.show()
        self.window_closed = False
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        rospy.Subscriber(self.sensor_topic, Marker, self.sensor_callback, queue_size=200)
        rospy.Subscriber(self.ped_topic, MarkerArray, self.pedestrian_callback, queue_size=500)

    def sensor_callback(self, msg: Marker) -> None:
        if msg.action != Marker.ADD:
            return
        with self.data_lock:
            timestamp = msg.header.stamp.to_sec()
            self.sensor_positions.append((msg.pose.position.x, msg.pose.position.y, timestamp))

    def pedestrian_callback(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            if marker.action != Marker.ADD:
                continue
            track_id = str(marker.id)
            timestamp = marker.header.stamp.to_sec()
            curr_x, curr_y = marker.pose.position.x, marker.pose.position.y
            color = (marker.color.r, marker.color.g, marker.color.b, marker.color.a)

            with self.data_lock:
                track = self.pedestrian_tracks.setdefault(track_id, {"positions": [], "color": color})
                track["color"] = color
                positions = track["positions"]
                positions.append((curr_x, curr_y, timestamp))

    def on_close(self, _event) -> None:
        self.window_closed = True

    def compute_speed_data(self) -> Dict[str, Dict[str, Any]]:
        with self.data_lock:
            current_sensor_pos = None
            if self.sensor_positions:
                current_sensor_pos = np.array(self.sensor_positions[-1][:2])

            ped_speeds: Dict[str, Dict[str, Any]] = {}
            for track_id, track in self.pedestrian_tracks.items():
                positions = track["positions"]
                color = track["color"]
                speeds = []
                last_valid_idx = 0  # Track the start of the most recent continuous segment
                if len(positions) >= 2:
                    if current_sensor_pos is not None:
                        ped_pos = np.array(positions[-1][:2])
                        distance_to_sensor = np.linalg.norm(ped_pos - current_sensor_pos)
                        if distance_to_sensor < self.min_sensor_distance:
                            continue

                    for i in range(1, len(positions)):
                        prev_x, prev_y, prev_t = positions[i - 1]
                        curr_x, curr_y, curr_t = positions[i]
                        dt = curr_t - prev_t
                        if dt <= 0:
                            continue
                        distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                        # Skip segments with large gaps (reset trajectory)
                        if dt > self.max_time_gap or distance > self.max_distance_gap:
                            last_valid_idx = i  # Mark gap point, reset from here
                            continue
                        else:
                            speed = distance / dt
                        speeds.append((prev_x, prev_y, curr_x, curr_y, speed))
                    
                    # Trim positions to only keep from the last continuous segment
                    if last_valid_idx > 0:
                        positions[:] = positions[last_valid_idx:]
                        
                if speeds:
                    ped_speeds[track_id] = {
                        "color": color,
                        "speeds": speeds,
                        "positions": positions,
                    }

            sorted_peds = sorted(ped_speeds.items(), key=lambda x: len(x[1]["speeds"]), reverse=True)
            top_peds = dict(sorted_peds[: self.max_pedestrians])

        return top_peds

    def build_live_graph(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        with self.data_lock:
            if len(self.sensor_positions) < 2:
                return None
            curr_x, curr_y, curr_t = self.sensor_positions[-1]
            prev_x, prev_y, prev_t = self.sensor_positions[-2]
            dt = curr_t - prev_t
            if dt <= 0.0:
                return None
            robot_vel = np.array([(curr_x - prev_x) / dt, (curr_y - prev_y) / dt], dtype=np.float32)

            frame_nodes: List[List[float]] = []
            frame_nodes.append([curr_x, curr_y, robot_vel[0], robot_vel[1], 1.0])

            current_sensor_pos = np.array([curr_x, curr_y], dtype=np.float32)
            ped_count = 0
            for track_id, track in self.pedestrian_tracks.items():
                positions = track["positions"]
                if len(positions) < 2:
                    continue
                prev_px, prev_py, prev_pt = positions[-2]
                curr_px, curr_py, curr_pt = positions[-1]
                dtp = curr_pt - prev_pt
                if dtp <= 0.0:
                    continue
                distance_to_sensor = np.linalg.norm(np.array([curr_px, curr_py], dtype=np.float32) - current_sensor_pos)
                if distance_to_sensor < self.min_sensor_distance:
                    continue
                ped_speed_x = (curr_px - prev_px) / dtp
                ped_speed_y = (curr_py - prev_py) / dtp
                frame_nodes.append([curr_px, curr_py, ped_speed_x, ped_speed_y, 0.0])
                ped_count += 1

            frame_array = np.asarray(frame_nodes, dtype=np.float32)
            x_node = preprocess_frame_to_node_features(frame_array)
            edge_index, edge_attr = build_directional_star(x_node)
            return frame_array, x_node, edge_index, edge_attr

    def predict_offset(self, x_node: np.ndarray, frame_array: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> np.ndarray:
        data = Data(
            x=torch.from_numpy(x_node),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
        )
        with torch.no_grad():
            y_hat_local = self.model(data.to(self.device)).cpu().numpy().reshape(-1)
        
        # Transform local offset to global
        robot_vx, robot_vy = frame_array[0, 2], frame_array[0, 3]
        speed = np.sqrt(robot_vx**2 + robot_vy**2)
        #if speed > 1e-6:
        theta = np.arctan2(robot_vy, robot_vx)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_theta = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        y_hat = R_theta @ y_hat_local
        #else:
        #    y_hat = y_hat_local
        
        return y_hat

    def draw_on_axis(self, ax, sensor_positions, offset_positions, ped_speeds) -> None:
        ax.clear()

        # Slice data if show_frame is set
        sensor_pos_slice = sensor_positions[-self.show_frame:] if self.show_frame and len(sensor_positions) > self.show_frame else sensor_positions
        offset_pos_slice = offset_positions[-self.show_frame:] if self.show_frame and len(offset_positions) > self.show_frame else offset_positions
        #print(len(sensor_pos_slice), len(offset_pos_slice))

        if sensor_pos_slice:
            sensor_pos_array = np.array([(x, y) for x, y, _ in sensor_pos_slice], dtype=np.float32)
            ax.plot(sensor_pos_array[:, 0], sensor_pos_array[:, 1], color="black", linewidth=2.0, label="sensor trajectory")
            curr_x, curr_y, _ = sensor_pos_slice[-1]
            ax.scatter(curr_x, curr_y, c="black", s=100, marker="o")

        if offset_pos_slice:
            offset_array = np.array(offset_pos_slice, dtype=np.float32)
            ax.plot(offset_array[:, 0], offset_array[:, 1], color="tab:red", linewidth=2.0, linestyle="--", label="predicted offset trajectory")
            ax.scatter(offset_array[-1, 0], offset_array[-1, 1], c="tab:red", s=100, marker="x")

        if self.last_predicted_offset is not None and sensor_pos_slice:
            curr_x, curr_y, _ = sensor_pos_slice[-1]
            ax.arrow(
                curr_x,
                curr_y,
                self.last_predicted_offset[0],
                self.last_predicted_offset[1],
                color="tab:red",
                width=0.02,
                length_includes_head=True,
                head_width=0.08,
                zorder=5,
                label="predicted offset",
            )

        # The sensor trajectory is already plotted as a black line above.
        # Remove the speed-based arrow overlay so the plot shows the actual trajectory clearly.

        for track_id, track_data in ped_speeds.items():
            color = track_data["color"]
            speeds = track_data["speeds"]
            speeds_slice = speeds[-self.show_frame:] if self.show_frame else speeds
            for prev_x, prev_y, curr_x, curr_y, speed in speeds_slice:
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                scale = min(speed * 0.1, 1.0)
                ax.arrow(prev_x, prev_y, dx * scale, dy * scale, head_width=0.1, head_length=0.2, fc=color, ec=color, alpha=0.6)
            if speeds:
                _, _, curr_x, curr_y, _ = speeds[-1]
                ax.scatter(curr_x, curr_y, c=[color], s=50, marker="o", label=f"ped {track_id}")

        all_points = []
        if sensor_pos_slice:
            all_points.extend([(x, y) for x, y, _ in sensor_pos_slice])
        for track_data in ped_speeds.values():
            for data in track_data["speeds"]:
                all_points.extend([(data[0], data[1]), (data[2], data[3])])
        if offset_pos_slice:
            all_points.extend(offset_pos_slice)

        if all_points:
            all_points_np = np.array(all_points, dtype=np.float32)
            min_xy = np.min(all_points_np, axis=0)
            max_xy = np.max(all_points_np, axis=0)
            center_xy = 0.5 * (min_xy + max_xy)
            half_range = 0.5 * np.max(max_xy - min_xy) + 0.5
            ax.set_xlim(center_xy[0] - half_range, center_xy[0] + half_range)
            ax.set_ylim(center_xy[1] - half_range, center_xy[1] + half_range)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(self.title)
        ax.legend(loc="upper right")

    def redraw(self) -> None:
        if self.window_closed or not plt.fignum_exists(self.fig.number):
            self.window_closed = True
            return

        ped_speeds = self.compute_speed_data()
        with self.data_lock:
            sensor_positions = list(self.sensor_positions)
            offset_positions = list(self.offset_positions)

        curr_x, curr_y, _ = sensor_positions[-1] if sensor_positions else (0.0, 0.0, 0.0)

        if len(sensor_positions) >= 2:
            graph_data = self.build_live_graph()
            if graph_data is not None:
                frame_array, x_node, edge_index, edge_attr = graph_data
                predicted_offset = self.predict_offset(x_node, frame_array, edge_index, edge_attr)
            else:
                predicted_offset = np.array([0.0, 0.0])
        else:
            predicted_offset = np.array([0.0, 0.0])

        self.last_predicted_offset = predicted_offset

        if len(sensor_positions) != self.last_sensor_index:
            self.offset_positions.append((curr_x + predicted_offset[0], curr_y + predicted_offset[1]))
            self.last_sensor_index = len(sensor_positions)

        self.draw_on_axis(self.ax, sensor_positions, offset_positions, ped_speeds)
        try:
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            self.window_closed = True

    def save_current_figure(self) -> None:
        ped_speeds = self.compute_speed_data()
        with self.data_lock:
            sensor_positions = list(self.sensor_positions)
            offset_positions = list(self.offset_positions)
        save_fig = Figure(figsize=(10, 10))
        save_ax = save_fig.subplots()
        self.draw_on_axis(save_ax, sensor_positions, offset_positions, ped_speeds)
        save_fig.savefig(self.save_path)
        rospy.loginfo("Saved offset plot figure to %s", self.save_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-topic", type=str, default="/motion_detector/visualization/lidar_pose")
    ap.add_argument("--ped-topic", type=str, default="/motion_detector/visualization/detections/centroid/dynamic")
    ap.add_argument("--model-path", type=str, default="gnn_model.pt")
    ap.add_argument("--save-path", type=str, default="offset_plot.png")
    ap.add_argument("--title", type=str, default="Offset Sensor Trajectory")
    ap.add_argument("--max-pedestrians", type=int, default=30)
    ap.add_argument("--refresh-hz", type=float, default=10.0, help="Plot refresh rate.")
    ap.add_argument("--max-time-gap", type=float, default=0.2, help="Max time gap (seconds) to consider same object.")
    ap.add_argument("--max-distance-gap", type=float, default=0.5, help="Max distance gap (meters) to consider same object.")
    ap.add_argument("--min-sensor-distance", type=float, default=0.2, help="Minimum distance from sensor to consider pedestrian valid (filter false positives).")
    ap.add_argument("--show-frame", type=int, default=None, help="Number of recent frames to show in trajectories. If not set, show all history.")
    args = ap.parse_args()

    rospy.init_node("offset_plotter", anonymous=True)
    plotter = OffsetPlotter(
        sensor_topic=args.sensor_topic,
        ped_topic=args.ped_topic,
        model_path=args.model_path,
        save_path=args.save_path,
        title=args.title,
        max_pedestrians=args.max_pedestrians,
        refresh_hz=args.refresh_hz,
        max_time_gap=args.max_time_gap,
        max_distance_gap=args.max_distance_gap,
        min_sensor_distance=args.min_sensor_distance,
        show_frame=args.show_frame,
    )

    rate = rospy.Rate(args.refresh_hz)
    while not rospy.is_shutdown() and not plotter.window_closed:
        plotter.redraw()
        rate.sleep()

    if plotter.save_path:
        plotter.save_current_figure()


if __name__ == "__main__":
    main()
