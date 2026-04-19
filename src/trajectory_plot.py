#!/usr/bin/env python3
"""
Plot Dynablox lidar and pedestrian trajectories in real time from ROS topics.

Default topics:
  - Sensor:     /motion_detector/visualization/lidar_pose
  - Pedestrian: /motion_detector/visualization/detections/centroid/dynamic

Pedestrian topic is expected to publish visualization_msgs/MarkerArray with SPHERE markers.

Example:
  python3 src/dynablox_trajectory.py

Optional:
  python3 src/dynablox_trajectory.py \
      --sensor-topic /motion_detector/visualization/lidar_pose \
      --ped-topic /motion_detector/visualization/detections/object/dynamic \
      --save-path dynablox_trajectory.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from threading import Lock
from typing import Any, DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.figure import Figure


Point2 = Tuple[float, float]
TrajectoryMap = Dict[str, Dict[str, Any]]


def append_point(trajectory: List[Point2], x: float, y: float) -> None:
    trajectory.append((float(x), float(y)))


def append_track_point(tracks: TrajectoryMap, key: str, x: float, y: float) -> None:
    tracks[key].append((float(x), float(y)))


class TrajectoryPlotter:
    def __init__(self, sensor_topic: str, ped_topic: str, save_path: str, title: str, max_pedestrians: int, refresh_hz: float):
        self.sensor_topic = sensor_topic
        self.ped_topic = ped_topic
        self.save_path = save_path
        self.title = title
        self.max_pedestrians = max_pedestrians
        self.refresh_hz = refresh_hz
        self.sensor_traj: List[Point2] = []
        self.pedestrian_trajs: TrajectoryMap = {}
        self.data_lock = Lock()

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
            append_point(self.sensor_traj, msg.pose.position.x, msg.pose.position.y)

    def pedestrian_callback(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            if marker.action != Marker.ADD:
                continue
            track_id = str(marker.id)
            with self.data_lock:
                if track_id not in self.pedestrian_trajs:
                    self.pedestrian_trajs[track_id] = {
                        'positions': [],
                        'color': (marker.color.r, marker.color.g, marker.color.b, marker.color.a)
                    }
                append_point(self.pedestrian_trajs[track_id]['positions'], marker.pose.position.x, marker.pose.position.y)

    def on_close(self, _event) -> None:
        self.window_closed = True

    def snapshot_data(self) -> Tuple[List[Point2], Dict[str, Dict[str, Any]]]:
        with self.data_lock:
            sensor_traj = list(self.sensor_traj)
            pedestrian_trajs = {k: {'positions': list(v['positions']), 'color': v['color']} for k, v in self.pedestrian_trajs.items()}
        ped_items = sorted(pedestrian_trajs.items(), key=lambda kv: len(kv[1]['positions']), reverse=True)
        return sensor_traj, dict(ped_items[: self.max_pedestrians])

    def draw_on_axis(self, ax, sensor_traj: List[Point2], top_tracks: Dict[str, Dict[str, Any]]) -> None:
        ax.clear()
        plotted_points: List[np.ndarray] = []

        if top_tracks:
            for ped_id, data in sorted(top_tracks.items()):
                traj = data['positions']
                color = data['color']
                traj_arr = np.asarray(traj, dtype=np.float32)
                if traj_arr.shape[0] == 0:
                    continue
                plotted_points.append(traj_arr)
                ax.scatter(traj_arr[:, 0], traj_arr[:, 1], color=color, s=20, alpha=0.6, label=f"ped {ped_id}")
                ax.scatter(traj_arr[0, 0], traj_arr[0, 1], color=color, s=25, marker="o")
                ax.scatter(traj_arr[-1, 0], traj_arr[-1, 1], color=color, s=35, marker="x")

        if sensor_traj:
            sensor_arr = np.asarray(sensor_traj, dtype=np.float32)
            plotted_points.append(sensor_arr)
            ax.plot(sensor_arr[:, 0], sensor_arr[:, 1], color="black", linewidth=5.0, label="sensor", zorder=10)
            ax.scatter(sensor_arr[0, 0], sensor_arr[0, 1], color="black", s=50, marker="o", zorder=10)
            ax.scatter(sensor_arr[-1, 0], sensor_arr[-1, 1], color="black", s=70, marker="x", zorder=10)

        if plotted_points:
            all_points = np.vstack(plotted_points)
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
        ax.set_title(self.title)
        if sensor_traj:
            ax.legend(loc="upper right")

    def redraw(self) -> None:
        if self.window_closed or not plt.fignum_exists(self.fig.number):
            self.window_closed = True
            return
        sensor_traj, top_tracks = self.snapshot_data()
        self.draw_on_axis(self.ax, sensor_traj, top_tracks)
        try:
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            # GUI backend may be destroyed between close event and redraw.
            self.window_closed = True

    def save_current_figure(self) -> None:
        sensor_traj, top_tracks = self.snapshot_data()
        save_fig = Figure(figsize=(10, 10))
        save_ax = save_fig.subplots()
        self.draw_on_axis(save_ax, sensor_traj, top_tracks)
        save_fig.savefig(self.save_path)
        rospy.loginfo("Saved trajectory figure to %s", self.save_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-topic", type=str, default="/motion_detector/visualization/lidar_pose")
    ap.add_argument("--ped-topic", type=str, default="/motion_detector/visualization/detections/centroid/dynamic")
    ap.add_argument("--save-path", type=str, default="trajectory_plot.png")
    ap.add_argument("--title", type=str, default="Dynablox Trajectories")
    ap.add_argument("--max-pedestrians", type=int, default=30)
    ap.add_argument("--refresh-hz", type=float, default=2.0, help="Plot refresh rate.")
    args = ap.parse_args()

    rospy.init_node("dynablox_trajectory_plotter", anonymous=True)
    plotter = TrajectoryPlotter(
        sensor_topic=args.sensor_topic,
        ped_topic=args.ped_topic,
        save_path=args.save_path,
        title=args.title,
        max_pedestrians=args.max_pedestrians,
        refresh_hz=args.refresh_hz,
    )

    rospy.loginfo("Listening for sensor markers on %s", args.sensor_topic)
    rospy.loginfo("Listening for centroid markers on %s", args.ped_topic)
    rospy.loginfo("Close the plot window or Ctrl+C to stop. Figure will be saved to %s", args.save_path)

    try:
        while not rospy.is_shutdown() and not plotter.window_closed:
            plotter.redraw()
            if plotter.window_closed:
                break
            try:
                plt.pause(1.0 / max(args.refresh_hz, 1e-3))
            except Exception:
                plotter.window_closed = True
                break
    except KeyboardInterrupt:
        pass
    finally:
        plotter.save_current_figure()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
