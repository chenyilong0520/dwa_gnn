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
import os
import time
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
    ConstantVelocityEKF,
)

Point3 = Tuple[float, float, float]  # (x, y, timestamp)
TrackedState = Tuple[float, float, float, float, float]  # (x, y, vx, vy, timestamp)
NODE_FEATURE_NAMES = ["x_rel", "y_rel", "vx_rel", "vy_rel", "is_robot"]
LABEL_NAMES = ["dx", "dy"]


def load_normalization_stats(json_path: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    node_stats = payload.get("data_processed")
    label_stats = payload.get("label_processed")
    if not isinstance(node_stats, dict) or not isinstance(label_stats, dict):
        raise ValueError(
            f"Expected 'data_processed' and 'label_processed' entries in normalization stats file: {json_path}"
        )
    return node_stats, label_stats


def normalize_node_features_for_inference(
    x: np.ndarray,
    stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    x_norm = np.asarray(x, dtype=np.float32).copy()
    for idx, feature_name in enumerate(NODE_FEATURE_NAMES):
        if feature_name == "is_robot":
            continue

        feature_stats = stats.get(feature_name)
        if feature_stats is None:
            raise KeyError(f"Missing node stats for '{feature_name}'")

        scale = max(abs(float(feature_stats["min"])), abs(float(feature_stats["max"])))
        if scale < 1e-12:
            x_norm[:, idx] = 0.0
            continue
        x_norm[:, idx] = x_norm[:, idx] / scale
    return x_norm


def denormalize_label_prediction(
    y: np.ndarray,
    stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    y_denorm = np.asarray(y, dtype=np.float32).copy()
    for idx, label_name in enumerate(LABEL_NAMES):
        label_stats = stats.get(label_name)
        if label_stats is None:
            raise KeyError(f"Missing label stats for '{label_name}'")

        scale = max(abs(float(label_stats["min"])), abs(float(label_stats["max"])))
        if scale < 1e-12:
            y_denorm[idx] = 0.0
            continue
        y_denorm[idx] = y_denorm[idx] * scale
    return y_denorm

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
        sensor_noise_thresh: float,
        min_sensor_distance: float,
        max_sensor_distance: float,
        min_track_states: int,
        min_prediction_speed: float,
        visualize: bool,
        write_record: bool,
        record_path: str,
        stats_path: str,
        show_frame: Optional[int],
        idle_timeout: float,
        shutdown_grace_seconds: float,
    ):
        self.sensor_topic = sensor_topic
        self.ped_topic = ped_topic
        self.save_path = save_path
        self.title = title
        self.max_pedestrians = max_pedestrians
        self.refresh_hz = refresh_hz
        self.max_time_gap = max_time_gap
        self.max_distance_gap = max_distance_gap
        self.sensor_noise_thresh = sensor_noise_thresh
        self.min_sensor_distance = min_sensor_distance
        self.max_sensor_distance = max_sensor_distance
        self.min_track_states = min_track_states
        self.min_prediction_speed = min_prediction_speed
        self.visualize = visualize
        self.write_record = write_record
        self.record_path = record_path
        self.stats_path = stats_path
        self.show_frame = show_frame
        self.idle_timeout = idle_timeout
        self.shutdown_grace_seconds = shutdown_grace_seconds
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
        self.pending_redraw = False
        self.last_sensor_msg_wall_time: Optional[float] = None
        self.first_sensor_msg_wall_time: Optional[float] = None
        self.active_sensor_callback_count = 0
        self.finalizing = False
        self.save_completed_wall_time: Optional[float] = None
        self.idle_wait_logged = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path).to(self.device)
        self.model.eval()
        self.node_feature_stats, self.label_stats = load_normalization_stats(self.stats_path)

        self.fig = None
        self.ax = None
        self.window_closed = False
        if self.visualize:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()
            self.fig.show()
            self.fig.canvas.mpl_connect("close_event", self.on_close)
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        rospy.Subscriber(self.sensor_topic, Marker, self.sensor_callback, queue_size=200)
        rospy.Subscriber(self.ped_topic, MarkerArray, self.pedestrian_callback, queue_size=500)

    def sensor_callback(self, msg: Marker) -> None:
        if msg.action != Marker.ADD:
            return
        sensor_state: Optional[TrackedState] = None
        valid_tracks_snapshot: Dict[str, Dict[str, Any]] = {}
        with self.data_lock:
            if self.finalizing:
                return
            now_wall = time.monotonic()
            self.active_sensor_callback_count += 1
            self.last_sensor_msg_wall_time = now_wall
            if self.first_sensor_msg_wall_time is None:
                self.first_sensor_msg_wall_time = now_wall
            self.idle_wait_logged = False

        try:
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
                sensor_state = (
                    float(updated_state[0]),
                    float(updated_state[1]),
                    float(updated_state[2]),
                    float(updated_state[3]),
                    timestamp,
                )
                self.sensor_filtered_states.append(sensor_state)
                sensor_pos = np.array(sensor_state[:2], dtype=np.float32)
                valid_tracks_snapshot = self.get_valid_pedestrian_tracks(timestamp, sensor_pos)

            if sensor_state is None:
                return

            predicted_offset = self.predict_offset_for_sensor_state(sensor_state, valid_tracks_snapshot)
            pedestrian_snapshot = self.serialize_pedestrian_snapshot(valid_tracks_snapshot)

            with self.data_lock:
                if self.finalizing:
                    return
                self.last_predicted_offset = predicted_offset
                new_offset = (sensor_state[0] + predicted_offset[0], sensor_state[1] + predicted_offset[1])
                self.offset_positions.append(new_offset)
                self.last_sensor_index = len(self.sensor_filtered_states)
                if self.write_record:
                    self.append_record_frame(sensor_state, new_offset, pedestrian_snapshot)
                if self.visualize:
                    self.pending_redraw = True
        finally:
            with self.data_lock:
                self.active_sensor_callback_count -= 1

    def pedestrian_callback(self, msg: MarkerArray) -> None:
        with self.data_lock:
            if self.finalizing:
                return

        latest_msg_timestamp: Optional[float] = None
        try:
            for marker in msg.markers:
                if marker.action != Marker.ADD:
                    continue
                track_id = str(marker.id)
                timestamp = marker.header.stamp.to_sec()
                if latest_msg_timestamp is None or timestamp > latest_msg_timestamp:
                    latest_msg_timestamp = timestamp
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

            if latest_msg_timestamp is None:
                return

            with self.data_lock:
                stale_track_ids = []
                for track_id, track in self.pedestrian_tracks.items():
                    filtered_states = track["filtered_states"]
                    if not filtered_states:
                        stale_track_ids.append(track_id)
                        continue

                    last_state_timestamp = filtered_states[-1][4]
                    if latest_msg_timestamp - last_state_timestamp > self.max_time_gap:
                        stale_track_ids.append(track_id)

                for track_id in stale_track_ids:
                    del self.pedestrian_tracks[track_id]
        finally:
            pass

    def on_close(self, _event) -> None:
        self.window_closed = True

    def on_key_press(self, event) -> None:
        if event.key in {"q", "escape"}:
            self.window_closed = True
            if self.fig is not None:
                plt.close(self.fig)

    def get_valid_pedestrian_tracks(
        self,
        reference_timestamp: Optional[float],
        sensor_pos: Optional[np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        valid_tracks_with_distance: List[Tuple[str, Dict[str, Any], Optional[float]]] = []
        for track_id, track in self.pedestrian_tracks.items():
            filtered_states = track["filtered_states"]
            if len(filtered_states) < self.min_track_states:
                continue

            latest_state = filtered_states[-1]
            if reference_timestamp is not None and reference_timestamp - latest_state[4] > self.max_time_gap:
                continue

            distance_to_sensor: Optional[float] = None
            if sensor_pos is not None:
                ped_pos = np.array(latest_state[:2], dtype=np.float32)
                distance_to_sensor = float(np.linalg.norm(ped_pos - sensor_pos))

            valid_tracks_with_distance.append(
                (
                    track_id,
                    {
                        "color": track["color"],
                        "filtered_states": list(filtered_states),
                        "latest_state": latest_state,
                    },
                    distance_to_sensor,
                )
            )

        if sensor_pos is not None:
            sorted_tracks = sorted(valid_tracks_with_distance, key=lambda item: item[2] if item[2] is not None else float("inf"))
            denoised_tracks = [item for item in sorted_tracks if item[2] is not None and item[2] >= self.sensor_noise_thresh]
            if not denoised_tracks:
                return {}

            closest_distance = denoised_tracks[0][2]
            if closest_distance is None or closest_distance > self.min_sensor_distance:
                return {}

            bounded_tracks = [
                (track_id, track)
                for track_id, track, distance in denoised_tracks
                if distance is not None and distance <= self.max_sensor_distance
            ]
            return dict(bounded_tracks[: self.max_pedestrians])

        sorted_tracks = sorted(
            valid_tracks_with_distance,
            key=lambda item: len(item[1]["filtered_states"]),
            reverse=True,
        )
        return dict((track_id, track) for track_id, track, _ in sorted_tracks[: self.max_pedestrians])

    def serialize_pedestrian_snapshot(self, valid_tracks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        pedestrians: List[Dict[str, Any]] = []

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

    def append_record_frame(
        self,
        sensor_state: TrackedState,
        offset_position: Tuple[float, float],
        pedestrian_snapshot: List[Dict[str, Any]],
    ) -> None:
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
                "pedestrian_tracks": pedestrian_snapshot,
            }
        )
        self.record_frame_index += 1

    def compute_tracking_data(self) -> Dict[str, Dict[str, Any]]:
        with self.data_lock:
            current_sensor_pos = None
            current_timestamp = None
            if self.sensor_filtered_states:
                current_sensor_pos = np.array(self.sensor_filtered_states[-1][:2], dtype=np.float32)
                current_timestamp = self.sensor_filtered_states[-1][4]

            ped_tracks: Dict[str, Dict[str, Any]] = {}
            valid_tracks = self.get_valid_pedestrian_tracks(current_timestamp, current_sensor_pos)
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
        return ped_tracks

    def build_live_graph_from_snapshot(
        self,
        sensor_state: TrackedState,
        valid_tracks: Dict[str, Dict[str, Any]],
    ) -> np.ndarray:
        curr_x, curr_y, curr_vx, curr_vy, _ = sensor_state
        frame_nodes: List[List[float]] = []
        frame_nodes.append([curr_x, curr_y, curr_vx, curr_vy, 1.0])

        for track in valid_tracks.values():
            curr_px, curr_py, ped_vx, ped_vy, _ = track["latest_state"]
            frame_nodes.append([curr_px, curr_py, ped_vx, ped_vy, 0.0])

        return np.asarray(frame_nodes, dtype=np.float32)

    def predict_offset_for_sensor_state(
        self,
        sensor_state: TrackedState,
        valid_tracks: Dict[str, Dict[str, Any]],
    ) -> np.ndarray:
        if len(self.sensor_filtered_states) < 2 or len(valid_tracks) < 1:
            return np.zeros(2, dtype=np.float32)

        frame_array = self.build_live_graph_from_snapshot(sensor_state, valid_tracks)
        x_node = preprocess_frame_to_node_features(frame_array)
        x_node = normalize_node_features_for_inference(x_node, self.node_feature_stats)
        edge_index, edge_attr = build_bidirectional_star(x_node)
        return self.predict_offset(x_node, frame_array, edge_index, edge_attr)

    def predict_offset(self, x_node: np.ndarray, frame_array: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> np.ndarray:
        data = Data(
            x=torch.from_numpy(x_node),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
        )
        with torch.no_grad():
            y_hat_local = self.model(data.to(self.device)).cpu().numpy().reshape(-1)
        y_hat_local = denormalize_label_prediction(y_hat_local, self.label_stats)
        
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

            # for prev_x, prev_y, curr_x, curr_y, speed in segments_slice:
            #     dx = curr_x - prev_x
            #     dy = curr_y - prev_y
            #     distance = np.hypot(dx, dy)
            #     if distance > self.max_distance_gap:
            #         continue
            #     scale = min(speed * 0.1, 1.0)
            #     ax.arrow(prev_x, prev_y, dx * scale, dy * scale, head_width=0.1, head_length=0.2, fc=color, ec=color, alpha=0.6)

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
        if not self.visualize or self.fig is None or self.ax is None:
            return
        if self.window_closed or not plt.fignum_exists(self.fig.number):
            self.window_closed = True
            return

        ped_tracks = self.compute_tracking_data()
        with self.data_lock:
            sensor_positions = [(x, y, t) for x, y, _, _, t in self.sensor_filtered_states]

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
            frame_count = len(self.record_frames)
        with open(self.record_path, "w", encoding="ascii") as f:
            json.dump(record_payload, f, indent=2)
        rospy.loginfo("Saved record file to %s", self.record_path)
        print(f"Final total frames saved: {frame_count}")

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

    def log_autostop_state(self, reason: str) -> None:
        with self.data_lock:
            rospy.loginfo(
                "%s auto-stop state: first_sensor_msg_wall_time=%s, last_sensor_msg_wall_time=%s, "
                "active_sensor_callback_count=%d, finalizing=%s, save_completed_wall_time=%s, "
                "results_saved=%s, record_frame_count=%d",
                reason,
                self.first_sensor_msg_wall_time,
                self.last_sensor_msg_wall_time,
                self.active_sensor_callback_count,
                self.finalizing,
                self.save_completed_wall_time,
                self.results_saved,
                len(self.record_frames),
            )

    def close_figure(self) -> None:
        self.window_closed = True
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)


# plot the offset trajectory using EKF in real-time, then auto-save and stop after both topics go idle
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-topic", type=str, default="/motion_detector/visualization/lidar_pose")
    ap.add_argument("--ped-topic", type=str, default="/motion_detector/visualization/detections/centroid/dynamic")
    ap.add_argument("--model-path", type=str, default="gnn_model.pt")
    ap.add_argument("--stats-path", type=str, default=os.path.join("data", "node_feature_stats.json"))
    ap.add_argument("--save-path", type=str, default="offset_plot_ekf.png")
    ap.add_argument("--title", type=str, default="Offset Sensor Trajectory")
    ap.add_argument("--max-pedestrians", type=int, default=50)
    ap.add_argument("--refresh-hz", type=float, default=10.0, help="Plot refresh rate.")
    ap.add_argument("--max-time-gap", type=float, default=0.2, help="Max time gap (seconds) to consider same object.")
    ap.add_argument("--max-distance-gap", type=float, default=0.5, help="Max distance gap (meters) to consider same object.")
    ap.add_argument("--sensor-noise-thresh", type=float, default=0.45, help="Remove detections closer than this distance to the sensor as noise.")
    ap.add_argument("--min-sensor-distance", type=float, default=2.5, help="If the closest remaining detection is farther than this distance from the sensor, return no pedestrian tracks.")
    ap.add_argument("--max-sensor-distance", type=float, default=50, help="Keep only detections within this maximum distance from the sensor.")
    ap.add_argument("--min-track-states", type=int, default=2, help="Minimum number of filtered states required before a pedestrian track is considered valid.")
    ap.add_argument("--min-prediction-speed", type=float, default=0.5, help="Suppress predicted offset when robot linear speed is below this threshold.")
    ap.add_argument("--visualize", action="store_true", help="Show the trajectory plot in real time.")
    ap.add_argument("--write-record", action="store_true", default=True, help="Write per-frame filtered robot/pedestrian snapshots to a JSON record file.")
    ap.add_argument("--record-path", type=str, default="record.json", help="Path to the JSON record file.")
    ap.add_argument("--show-frame", type=int, default=None, help="Number of recent frames to show in trajectories. If not set, show all history.")
    ap.add_argument("--idle-timeout", type=float, default=0.5, help="If both topics are quiet for at least this many wall-clock seconds, begin auto-stop.")
    ap.add_argument("--shutdown-grace-seconds", type=float, default=2.0, help="After files are saved, keep the process alive for this many seconds before exiting.")
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
        sensor_noise_thresh=args.sensor_noise_thresh,
        min_sensor_distance=args.min_sensor_distance,
        max_sensor_distance=args.max_sensor_distance,
        min_track_states=args.min_track_states,
        min_prediction_speed=args.min_prediction_speed,
        visualize=args.visualize,
        write_record=args.write_record,
        record_path=args.record_path,
        stats_path=args.stats_path,
        show_frame=args.show_frame,
        idle_timeout=args.idle_timeout,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
    )
    try:
        while not rospy.is_shutdown():
            should_redraw = False
            should_finalize = False
            now_wall = time.monotonic()
            if plotter.visualize:
                with plotter.data_lock:
                    should_redraw = plotter.pending_redraw
                    plotter.pending_redraw = False

            with plotter.data_lock:
                if plotter.first_sensor_msg_wall_time is not None:
                    sensor_quiet = (
                        plotter.last_sensor_msg_wall_time is not None
                        and now_wall - plotter.last_sensor_msg_wall_time >= plotter.idle_timeout
                    )
                    if sensor_quiet:
                        if plotter.active_sensor_callback_count == 0 and not plotter.finalizing:
                            rospy.loginfo(
                                "Sensor topic idle for %.2f s; preparing to save.",
                                plotter.idle_timeout,
                            )
                            plotter.finalizing = True
                            should_finalize = True
                        elif not plotter.finalizing and not plotter.idle_wait_logged:
                            rospy.loginfo(
                                "Sensor topic is idle, but waiting for %d in-flight sensor callback(s) before saving.",
                                plotter.active_sensor_callback_count,
                            )
                            plotter.idle_wait_logged = True
                    else:
                        plotter.idle_wait_logged = False

            if should_finalize:
                plotter.save_results_once()
                with plotter.data_lock:
                    plotter.save_completed_wall_time = time.monotonic()
                rospy.loginfo(
                    "Save complete. Waiting %.2f s before exit.",
                    plotter.shutdown_grace_seconds,
                )

            with plotter.data_lock:
                save_completed_wall_time = plotter.save_completed_wall_time
            if save_completed_wall_time is not None:
                if now_wall - save_completed_wall_time >= plotter.shutdown_grace_seconds:
                    plotter.close_figure()
                    break

            if should_redraw:
                plotter.redraw()
            else:
                rospy.sleep(0.01)
    except (KeyboardInterrupt, rospy.ROSInterruptException) as exc:
        plotter.log_autostop_state(f"Interrupted ({type(exc).__name__})")
        with plotter.data_lock:
            plotter.finalizing = True
        plotter.save_results_once()
    finally:
        plotter.close_figure()


if __name__ == "__main__":
    main()
