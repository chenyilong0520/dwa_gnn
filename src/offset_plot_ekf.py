#!/usr/bin/env python3
"""
Live sensor/pedestrian plotting with predicted offset trajectory.

This script is based on speed_plot.py, but it additionally predicts a per-frame
sensor offset using the GNN model and draws an offset version of the sensor
trajectory.

Example:
  python3 src/offset_plot_ekf.py --model-path gnn_model.pt
"""

from __future__ import annotations

import argparse
import json
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
    build_bidirectional_star,
    load_model,
    preprocess_frame_to_node_features,
)

Point3 = Tuple[float, float, float]  # (x, y, timestamp)
TrackedState = Tuple[float, float, float, float, float]  # (x, y, vx, vy, timestamp)


class ConstantVelocityEKF:
    def __init__(
        self,
        process_noise_position: float = 0.5,
        process_noise_velocity: float = 1.0,
        measurement_noise: float = 0.15,
        initial_covariance: float = 10.0,
    ) -> None:
        self.process_noise_position = process_noise_position
        self.process_noise_velocity = process_noise_velocity
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

    def initialize(self, x: float, y: float) -> None:
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * self.initial_covariance

    def predict(self, dt: float) -> None:
        if self.state is None or self.covariance is None or dt <= 0.0:
            return

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dt2 = dt * dt
        q_pos = self.process_noise_position
        q_vel = self.process_noise_velocity
        Q = np.array(
            [
                [0.25 * dt2 * dt2 * q_pos, 0.0, 0.5 * dt2 * dt * q_pos, 0.0],
                [0.0, 0.25 * dt2 * dt2 * q_pos, 0.0, 0.5 * dt2 * dt * q_pos],
                [0.5 * dt2 * dt * q_pos, 0.0, dt2 * q_vel, 0.0],
                [0.0, 0.5 * dt2 * dt * q_pos, 0.0, dt2 * q_vel],
            ],
            dtype=np.float64,
        )

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, measurement: np.ndarray) -> np.ndarray:
        if self.state is None or self.covariance is None:
            raise RuntimeError("EKF must be initialized before update().")

        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * self.measurement_noise
        innovation = measurement - self.measurement_function(self.state)
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - K @ H) @ self.covariance
        return self.state.copy()

    @staticmethod
    def measurement_function(state: np.ndarray) -> np.ndarray:
        return state[:2]


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
        max_sensor_distance: float,
        min_track_states: int,
        min_prediction_speed: float,
        write_record: bool,
        record_path: str,
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
        self.max_sensor_distance = max_sensor_distance
        self.min_track_states = min_track_states
        self.min_prediction_speed = min_prediction_speed
        self.write_record = write_record
        self.record_path = record_path
        self.show_frame = show_frame
        self.data_lock = Lock()

        self.sensor_positions: List[Point3] = []
        self.sensor_filtered_states: List[TrackedState] = []
        self.sensor_ekf = ConstantVelocityEKF()
        self.pedestrian_tracks: Dict[str, Dict[str, Any]] = {}
        self.offset_positions: List[Tuple[float, float]] = []
        self.last_predicted_offset: Optional[np.ndarray] = None
        self.last_sensor_index = 0
        self.record_frames: List[Dict[str, Any]] = []
        self.record_frame_index = 0
        self.results_saved = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path).to(self.device)
        self.model.eval()

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.ion()
        self.fig.show()
        self.window_closed = False
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        rospy.Subscriber(self.sensor_topic, Marker, self.sensor_callback, queue_size=200)
        rospy.Subscriber(self.ped_topic, MarkerArray, self.pedestrian_callback, queue_size=500)

    def sensor_callback(self, msg: Marker) -> None:
        if msg.action != Marker.ADD:
            return
        with self.data_lock:
            timestamp = msg.header.stamp.to_sec()
            curr_x = msg.pose.position.x
            curr_y = msg.pose.position.y

            if self.sensor_positions:
                prev_x, prev_y, prev_t = self.sensor_positions[-1]
                dt = timestamp - prev_t
                distance = np.hypot(curr_x - prev_x, curr_y - prev_y)
                if dt <= 0.0 or dt > self.max_time_gap or distance > self.max_distance_gap:
                    self.sensor_positions.clear()
                    self.sensor_filtered_states.clear()
                    self.offset_positions.clear()
                    self.last_sensor_index = 0
                    self.sensor_ekf = ConstantVelocityEKF()
                else:
                    self.sensor_ekf.predict(dt)
            else:
                self.sensor_ekf.initialize(curr_x, curr_y)

            self.sensor_positions.append((curr_x, curr_y, timestamp))
            if self.sensor_ekf.state is None:
                self.sensor_ekf.initialize(curr_x, curr_y)
            updated_state = self.sensor_ekf.update(np.array([curr_x, curr_y], dtype=np.float64))
            self.sensor_filtered_states.append(
                (
                    float(updated_state[0]),
                    float(updated_state[1]),
                    float(updated_state[2]),
                    float(updated_state[3]),
                    timestamp,
                )
            )

    def pedestrian_callback(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            if marker.action != Marker.ADD:
                continue
            track_id = str(marker.id)
            timestamp = marker.header.stamp.to_sec()
            curr_x, curr_y = marker.pose.position.x, marker.pose.position.y
            color = (marker.color.r, marker.color.g, marker.color.b, marker.color.a)

            with self.data_lock:
                track = self.pedestrian_tracks.setdefault(
                    track_id,
                    {
                        "positions": [],
                        "filtered_states": [],
                        "color": color,
                        "ekf": ConstantVelocityEKF(),
                    },
                )
                track["color"] = color
                positions = track["positions"]
                filtered_states = track["filtered_states"]
                ekf: ConstantVelocityEKF = track["ekf"]

                if positions:
                    prev_x, prev_y, prev_t = positions[-1]
                    dt = timestamp - prev_t
                    distance = np.hypot(curr_x - prev_x, curr_y - prev_y)
                    if dt <= 0.0 or dt > self.max_time_gap or distance > self.max_distance_gap:
                        positions.clear()
                        filtered_states.clear()
                        ekf = ConstantVelocityEKF()
                        track["ekf"] = ekf
                    else:
                        ekf.predict(dt)
                else:
                    ekf.initialize(curr_x, curr_y)

                positions.append((curr_x, curr_y, timestamp))
                if ekf.state is None:
                    ekf.initialize(curr_x, curr_y)
                updated_state = ekf.update(np.array([curr_x, curr_y], dtype=np.float64))
                filtered_states.append(
                    (
                        float(updated_state[0]),
                        float(updated_state[1]),
                        float(updated_state[2]),
                        float(updated_state[3]),
                        timestamp,
                    )
                )

    def on_close(self, _event) -> None:
        self.window_closed = True

    def on_key_press(self, event) -> None:
        if event.key in {"q", "escape"}:
            self.window_closed = True
            plt.close(self.fig)

    def get_valid_pedestrian_tracks(self, sensor_pos: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        valid_tracks: Dict[str, Dict[str, Any]] = {}
        for track_id, track in self.pedestrian_tracks.items():
            filtered_states = track["filtered_states"]
            if len(filtered_states) < self.min_track_states:
                continue

            latest_state = filtered_states[-1]
            if sensor_pos is not None:
                ped_pos = np.array(latest_state[:2], dtype=np.float32)
                distance_to_sensor = np.linalg.norm(ped_pos - sensor_pos)
                if distance_to_sensor < self.min_sensor_distance or distance_to_sensor > self.max_sensor_distance:
                    continue

            valid_tracks[track_id] = {
                "color": track["color"],
                "filtered_states": list(filtered_states),
                "latest_state": latest_state,
            }
        return valid_tracks

    def snapshot_latest_pedestrians(self, sensor_x: float, sensor_y: float) -> List[Dict[str, Any]]:
        pedestrians: List[Dict[str, Any]] = []
        current_sensor_pos = np.array([sensor_x, sensor_y], dtype=np.float32)
        valid_tracks = self.get_valid_pedestrian_tracks(current_sensor_pos)

        for track_id, track in valid_tracks.items():
            latest_state = track["latest_state"]
            color = track["color"]
            pedestrians.append(
                {
                    "track_id": track_id,
                    "color": [float(color[0]), float(color[1]), float(color[2]), float(color[3])],
                    "filtered_state": [
                        float(latest_state[0]),
                        float(latest_state[1]),
                        float(latest_state[2]),
                        float(latest_state[3]),
                    ],
                }
            )

        pedestrians.sort(key=lambda item: item["track_id"])
        return pedestrians

    def append_record_frame(self, sensor_state: TrackedState, offset_position: Tuple[float, float]) -> None:
        sensor_x, sensor_y, sensor_vx, sensor_vy, timestamp = sensor_state
        self.record_frames.append(
            {
                "frame_index": self.record_frame_index,
                "timestamp": float(timestamp),
                "sensor_filtered_state": [
                    float(sensor_x),
                    float(sensor_y),
                    float(sensor_vx),
                    float(sensor_vy),
                ],
                "offset_position": [float(offset_position[0]), float(offset_position[1])],
                "pedestrian_tracks": self.snapshot_latest_pedestrians(sensor_x, sensor_y),
            }
        )
        self.record_frame_index += 1

    def compute_tracking_data(self) -> Dict[str, Dict[str, Any]]:
        with self.data_lock:
            current_sensor_pos = None
            if self.sensor_filtered_states:
                current_sensor_pos = np.array(self.sensor_filtered_states[-1][:2], dtype=np.float32)

            ped_tracks: Dict[str, Dict[str, Any]] = {}
            valid_tracks = self.get_valid_pedestrian_tracks(current_sensor_pos)
            for track_id, track in valid_tracks.items():
                color = track["color"]
                filtered_states = track["filtered_states"]
                segments = []
                for i in range(1, len(filtered_states)):
                    prev_x, prev_y, _, _, _ = filtered_states[i - 1]
                    curr_x, curr_y, curr_vx, curr_vy, _ = filtered_states[i]
                    speed = float(np.hypot(curr_vx, curr_vy))
                    segments.append((prev_x, prev_y, curr_x, curr_y, speed))

                if segments:
                    ped_tracks[track_id] = {
                        "color": color,
                        "segments": segments,
                        "filtered_states": list(filtered_states),
                        "latest_state": filtered_states[-1],
                    }

            sorted_peds = sorted(ped_tracks.items(), key=lambda x: len(x[1]["segments"]), reverse=True)
            top_peds = dict(sorted_peds[: self.max_pedestrians])

        return top_peds

    def build_live_graph(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        with self.data_lock:
            if len(self.sensor_filtered_states) < 2:
                return None
            curr_x, curr_y, curr_vx, curr_vy, _ = self.sensor_filtered_states[-1]
            robot_vel = np.array([curr_vx, curr_vy], dtype=np.float32)

            frame_nodes: List[List[float]] = []
            frame_nodes.append([curr_x, curr_y, robot_vel[0], robot_vel[1], 1.0])

            current_sensor_pos = np.array([curr_x, curr_y], dtype=np.float32)
            valid_tracks = self.get_valid_pedestrian_tracks(current_sensor_pos)
            for track in valid_tracks.values():
                curr_px, curr_py, curr_vx, curr_vy, _ = track["latest_state"]
                frame_nodes.append([curr_px, curr_py, curr_vx, curr_vy, 0.0])

            frame_array = np.asarray(frame_nodes, dtype=np.float32)
            x_node = preprocess_frame_to_node_features(frame_array)
            #edge_index, edge_attr = build_directional_star(x_node)
            edge_index, edge_attr = build_bidirectional_star(x_node)
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
        if speed < self.min_prediction_speed: # to suppress noise when robot is nearly stationary
            return np.zeros(2, dtype=np.float32)
        if speed > 1e-6:
            theta = np.arctan2(robot_vy, robot_vx)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R_theta = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            y_hat = R_theta @ y_hat_local
        else:
            y_hat = y_hat_local
        
        return y_hat

    def draw_on_axis(self, ax, sensor_positions, offset_positions, ped_tracks) -> None:
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
        #print(len(sensor_pos_slice), len(offset_pos_slice), len(ped_tracks))

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

        for track_id, track_data in ped_tracks.items():
            color = track_data["color"]
            segments = track_data["segments"]
            filtered_states = track_data["filtered_states"]
            segments_slice = segments[-self.show_frame:] if self.show_frame else segments
            states_slice = filtered_states[-self.show_frame:] if self.show_frame else filtered_states

            for prev_x, prev_y, curr_x, curr_y, speed in segments_slice:
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = np.hypot(dx, dy)
                if distance > self.max_distance_gap:
                    continue
                scale = min(speed * 0.1, 1.0)
                ax.arrow(prev_x, prev_y, dx * scale, dy * scale, head_width=0.1, head_length=0.2, fc=color, ec=color, alpha=0.6)

            if states_slice:
                state_array = np.array([(x, y) for x, y, _, _, _ in states_slice], dtype=np.float32)
                ax.plot(state_array[:, 0], state_array[:, 1], color=color, linewidth=1.5, alpha=0.8)
                curr_x, curr_y, curr_vx, curr_vy, _ = states_slice[-1]
                ax.scatter(curr_x, curr_y, c=[color], s=50, marker="o", label=f"ped {track_id}")
                ax.arrow(
                    curr_x,
                    curr_y,
                    curr_vx * 0.3,
                    curr_vy * 0.3,
                    head_width=0.1,
                    head_length=0.15,
                    fc=color,
                    ec=color,
                    alpha=0.9,
                )

        all_points = []
        if sensor_pos_slice:
            all_points.extend([(x, y) for x, y, _ in sensor_pos_slice])
        for track_data in ped_tracks.values():
            for data in track_data["segments"]:
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

        ped_tracks = self.compute_tracking_data()
        with self.data_lock:
            sensor_positions = [(x, y, t) for x, y, _, _, t in self.sensor_filtered_states]

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
            # Append an offset entry for every new sensor frame since last update.
            with self.data_lock:
                for i in range(self.last_sensor_index, len(self.sensor_filtered_states)):
                    sensor_state = self.sensor_filtered_states[i]
                    x, y, _, _, _ = sensor_state
                    new_offset = (x + predicted_offset[0], y + predicted_offset[1])
                    self.offset_positions.append(new_offset)
                    if self.write_record:
                        self.append_record_frame(sensor_state, new_offset)
                self.last_sensor_index = len(self.sensor_filtered_states)

        # Snapshot offsets after the update so drawing sees the latest copy.
        with self.data_lock:
            offset_positions = list(self.offset_positions)

        #print(len(self.sensor_positions), len(self.offset_positions), len(ped_tracks))

        self.draw_on_axis(self.ax, sensor_positions, offset_positions, ped_tracks)
        try:
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            self.fig.canvas.flush_events()
        except Exception:
            self.window_closed = True

    def save_current_figure(self) -> None:
        ped_tracks = self.compute_tracking_data()
        with self.data_lock:
            sensor_positions = [(x, y, t) for x, y, _, _, t in self.sensor_filtered_states]
            offset_positions = list(self.offset_positions)
        save_fig = Figure(figsize=(10, 10))
        save_ax = save_fig.subplots()
        self.draw_on_axis(save_ax, sensor_positions, offset_positions, ped_tracks)
        save_fig.savefig(self.save_path)
        plt.close(save_fig)
        rospy.loginfo("Saved offset plot figure to %s", self.save_path)

    def save_record(self) -> None:
        if not self.write_record:
            return
        with self.data_lock:
            record_payload = {
                "frames": list(self.record_frames),
            }
        with open(self.record_path, "w", encoding="ascii") as f:
            json.dump(record_payload, f, indent=2)
        rospy.loginfo("Saved record file to %s", self.record_path)

    def save_results_once(self) -> None:
        if self.results_saved:
            return
        self.results_saved = True

        try:
            if self.save_path:
                self.save_current_figure()
        except Exception as exc:
            rospy.logwarn("Failed to save offset plot figure: %s", exc)

        try:
            self.save_record()
        except Exception as exc:
            rospy.logwarn("Failed to save record file: %s", exc)

    def shutdown(self) -> None:
        self.window_closed = True
        self.save_results_once()
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

# plot the offset trajectory using EKF in real-time, while also saving the final figure and record file on shutdown
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-topic", type=str, default="/motion_detector/visualization/lidar_pose")
    ap.add_argument("--ped-topic", type=str, default="/motion_detector/visualization/detections/centroid/dynamic")
    ap.add_argument("--model-path", type=str, default="gnn_model_best.pt")
    ap.add_argument("--save-path", type=str, default="offset_plot_ekf.png")
    ap.add_argument("--title", type=str, default="Offset Sensor Trajectory")
    ap.add_argument("--max-pedestrians", type=int, default=30)
    ap.add_argument("--refresh-hz", type=float, default=10.0, help="Plot refresh rate.")
    ap.add_argument("--max-time-gap", type=float, default=0.2, help="Max time gap (seconds) to consider same object.")
    ap.add_argument("--max-distance-gap", type=float, default=0.5, help="Max distance gap (meters) to consider same object.")
    ap.add_argument("--min-sensor-distance", type=float, default=0.45, help="Minimum distance from sensor to consider pedestrian valid (filter false positives).")
    ap.add_argument("--max-sensor-distance", type=float, default=2.5, help="Maximum distance from sensor to consider pedestrian valid (filter false positives).")
    ap.add_argument("--min-track-states", type=int, default=2, help="Minimum number of filtered states required before a pedestrian track is considered valid.")
    ap.add_argument("--min-prediction-speed", type=float, default=0.5, help="Suppress predicted offset when robot linear speed is below this threshold.")
    ap.add_argument("--write-record", action="store_true", default=True, help="Write per-frame filtered robot/pedestrian snapshots to a JSON record file.")
    ap.add_argument("--record-path", type=str, default="record.json", help="Path to the JSON record file.")
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
        max_sensor_distance=args.max_sensor_distance,
        min_track_states=args.min_track_states,
        min_prediction_speed=args.min_prediction_speed,
        write_record=args.write_record,
        record_path=args.record_path,
        show_frame=args.show_frame,
    )
    rospy.on_shutdown(plotter.shutdown)

    rate = rospy.Rate(args.refresh_hz)
    try:
        while not rospy.is_shutdown() and not plotter.window_closed:
            plotter.redraw()
            rate.sleep()
    except KeyboardInterrupt:
        plotter.shutdown()
    finally:
        plotter.shutdown()


if __name__ == "__main__":
    main()
