#!/usr/bin/env python3
"""
Plot Dynablox lidar and pedestrian speeds in real time from ROS topics.

Displays speed vectors as scaled arrows connecting previous and current poses.
Uses finite difference method for speed estimation with ID reuse protection.

Default topics:
  - Sensor:     /motion_detector/visualization/lidar_pose
  - Pedestrian: /motion_detector/visualization/detections/centroid/dynamic

Pedestrian topic is expected to publish visualization_msgs/MarkerArray with SPHERE markers.

Example:
  python3 src/speed_plot.py

Optional:
  python3 src/speed_plot.py \
      --sensor-topic /motion_detector/visualization/lidar_pose \
      --ped-topic /motion_detector/visualization/detections/object/dynamic \
      --save-path speed.png \
      --max-time-gap 1.0 \
      --max-distance-gap 2.0 \
      --min-sensor-distance 0.5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.figure import Figure


Point3 = Tuple[float, float, float]  # (x, y, timestamp)
SpeedData = Tuple[float, float, float, float, float]  # (prev_x, prev_y, curr_x, curr_y, speed)


class SpeedPlotter:
    def __init__(self, sensor_topic: str, ped_topic: str, save_path: str, title: str, max_pedestrians: int, refresh_hz: float, max_time_gap: float, max_distance_gap: float, min_sensor_distance: float):
        self.sensor_topic = sensor_topic
        self.ped_topic = ped_topic
        self.save_path = save_path
        self.title = title
        self.max_pedestrians = max_pedestrians
        self.refresh_hz = refresh_hz
        self.max_time_gap = max_time_gap
        self.max_distance_gap = max_distance_gap
        self.min_sensor_distance = min_sensor_distance
        self.data_lock = Lock()

        # Store positions with timestamps: (x, y, timestamp)
        self.sensor_positions: List[Point3] = []
        self.pedestrian_tracks: Dict[str, Dict[str, Any]] = {}

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
                track = self.pedestrian_tracks.setdefault(track_id, {'positions': [], 'color': color})
                track['color'] = color
                positions = track['positions']
                if positions:
                    prev_x, prev_y, prev_time = positions[-1]
                    time_gap = timestamp - prev_time
                    distance_gap = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)

                    # Keep the position history, but mark the speed as invalid if the gap is large.
                    positions.append((curr_x, curr_y, timestamp))
                else:
                    positions.append((curr_x, curr_y, timestamp))

    def on_close(self, _event) -> None:
        self.window_closed = True

    def compute_speed_data(self) -> Tuple[List[SpeedData], Dict[str, Dict[str, Any]]]:
        with self.data_lock:
            # Get current sensor position for filtering
            current_sensor_pos = None
            if self.sensor_positions:
                current_sensor_pos = np.array(self.sensor_positions[-1][:2])

            # Sensor speed data
            sensor_speeds = []
            if len(self.sensor_positions) >= 2:
                for i in range(1, len(self.sensor_positions)):
                    prev_x, prev_y, prev_t = self.sensor_positions[i-1]
                    curr_x, curr_y, curr_t = self.sensor_positions[i]
                    dt = curr_t - prev_t
                    if dt > 0:
                        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                        speed = distance / dt
                        sensor_speeds.append((prev_x, prev_y, curr_x, curr_y, speed))

            # Pedestrian speed data
            ped_speeds: Dict[str, Dict[str, Any]] = {}
            for track_id, track in self.pedestrian_tracks.items():
                positions = track['positions']
                color = track['color']
                speeds = []
                if len(positions) >= 2:
                    # Check if pedestrian is too close to sensor (filter out false positives)
                    if current_sensor_pos is not None:
                        # Check distance of the most recent position
                        ped_pos = np.array(positions[-1][:2])
                        distance_to_sensor = np.linalg.norm(ped_pos - current_sensor_pos)
                        if distance_to_sensor < self.min_sensor_distance:
                            continue  # Skip this pedestrian

                    for i in range(1, len(positions)):
                        prev_x, prev_y, prev_t = positions[i-1]
                        curr_x, curr_y, curr_t = positions[i]
                        dt = curr_t - prev_t
                        if dt <= 0:
                            continue
                        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                        if dt > self.max_time_gap or distance > self.max_distance_gap:
                            speed = 0.0
                        else:
                            speed = distance / dt
                        speeds.append((prev_x, prev_y, curr_x, curr_y, speed))
                if speeds:
                    ped_speeds[track_id] = {
                        'color': color,
                        'speeds': speeds,
                    }

            # Sort pedestrians by number of speed measurements and take top max_pedestrians
            sorted_peds = sorted(ped_speeds.items(), key=lambda x: len(x[1]['speeds']), reverse=True)
            top_peds = dict(sorted_peds[:self.max_pedestrians])

        return sensor_speeds, top_peds

    def draw_on_axis(self, ax, sensor_speeds: List[SpeedData], ped_speeds: Dict[str, Dict[str, Any]]) -> None:
        ax.clear()

        # Plot all sensor positions as a trajectory line
        if self.sensor_positions:
            sensor_pos_array = np.array([(x, y) for x, y, _ in self.sensor_positions])
            ax.plot(sensor_pos_array[:, 0], sensor_pos_array[:, 1], color='black', linewidth=2.0, label='sensor trajectory')
            # Plot current sensor position
            curr_x, curr_y, _ = self.sensor_positions[-1]
            ax.scatter(curr_x, curr_y, c='black', s=100, marker='o')

        # Plot sensor speed arrows
        if sensor_speeds:
            for prev_x, prev_y, curr_x, curr_y, speed in sensor_speeds[-10:]:  # Show last 10 for clarity
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                # Scale arrow by speed (arbitrary scaling factor)
                scale = min(speed * 0.1, 1.0)  # Cap at 1.0 for visibility
                ax.arrow(prev_x, prev_y, dx * scale, dy * scale,
                        head_width=0.05, head_length=0.1, fc='black', ec='black', alpha=0.7)

        # Plot pedestrian speed arrows
        for track_id, track_data in ped_speeds.items():
            color = track_data['color']
            speeds = track_data['speeds']
            for prev_x, prev_y, curr_x, curr_y, speed in speeds:
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                # Scale arrow by speed
                scale = min(speed * 0.1, 1.0)
                ax.arrow(prev_x, prev_y, dx * scale, dy * scale,
                        head_width=0.1, head_length=0.2, fc=color, ec=color, alpha=0.6)
            # Plot current pedestrian position
            if speeds:
                _, _, curr_x, curr_y, _ = speeds[-1]
                ax.scatter(curr_x, curr_y, c=[color], s=50, marker='o', label=f'ped {track_id}')

        # Set axis limits based on all points
        all_points = []
        if sensor_speeds:
            for data in sensor_speeds:
                all_points.extend([(data[0], data[1]), (data[2], data[3])])
        for track_data in ped_speeds.values():
            for data in track_data['speeds']:
                all_points.extend([(data[0], data[1]), (data[2], data[3])])

        if all_points:
            all_points = np.array(all_points, dtype=np.float32)
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
        ax.legend(loc="upper right")

    def redraw(self) -> None:
        if self.window_closed or not plt.fignum_exists(self.fig.number):
            self.window_closed = True
            return
        sensor_speeds, ped_speeds = self.compute_speed_data()
        self.draw_on_axis(self.ax, sensor_speeds, ped_speeds)
        try:
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            self.window_closed = True

    def save_current_figure(self) -> None:
        sensor_speeds, ped_speeds = self.compute_speed_data()
        save_fig = Figure(figsize=(10, 10))
        save_ax = save_fig.subplots()
        self.draw_on_axis(save_ax, sensor_speeds, ped_speeds)
        save_fig.savefig(self.save_path)
        rospy.loginfo("Saved speed plot figure to %s", self.save_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-topic", type=str, default="/motion_detector/visualization/lidar_pose")
    ap.add_argument("--ped-topic", type=str, default="/motion_detector/visualization/detections/centroid/dynamic")
    ap.add_argument("--save-path", type=str, default="speed.png")
    ap.add_argument("--title", type=str, default="Speed Vectors")
    ap.add_argument("--max-pedestrians", type=int, default=30)
    ap.add_argument("--refresh-hz", type=float, default=2.0, help="Plot refresh rate.")
    ap.add_argument("--max-time-gap", type=float, default=1.0, help="Max time gap (seconds) to consider same object.")
    ap.add_argument("--max-distance-gap", type=float, default=2.0, help="Max distance gap (meters) to consider same object.")
    ap.add_argument("--min-sensor-distance", type=float, default=0.2, help="Minimum distance from sensor to consider pedestrian valid (filter false positives).")
    args = ap.parse_args()

    rospy.init_node("speed_plotter", anonymous=True)
    plotter = SpeedPlotter(
        sensor_topic=args.sensor_topic,
        ped_topic=args.ped_topic,
        save_path=args.save_path,
        title=args.title,
        max_pedestrians=args.max_pedestrians,
        refresh_hz=args.refresh_hz,
        max_time_gap=args.max_time_gap,
        max_distance_gap=args.max_distance_gap,
        min_sensor_distance=args.min_sensor_distance,
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