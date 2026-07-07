#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


Point3 = Tuple[float, float, float]
TrackedState = Tuple[float, float, float, float, float]


def load_frames(record_path: str) -> List[Dict[str, Any]]:
    with open(record_path, "r", encoding="ascii") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("record file must contain a top-level 'frames' list.")
    return frames


def slice_frames(
    frames: Sequence[Dict[str, Any]],
    start_frame: int,
    end_frame: Optional[int],
) -> List[Dict[str, Any]]:
    if not frames:
        return []

    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")

    if end_frame is None:
        end_frame = len(frames) - 1

    if end_frame < start_frame:
        raise ValueError("--end-frame must be >= --start-frame")

    selected = [
        frame
        for frame in frames
        if start_frame <= int(frame["frame_index"]) <= end_frame
    ]
    return selected


def build_sensor_positions(frames: Sequence[Dict[str, Any]]) -> List[Point3]:
    sensor_positions: List[Point3] = []
    for frame in frames:
        state = frame["sensor_filtered_state"]
        sensor_positions.append((float(state[0]), float(state[1]), float(frame["timestamp"])))
    return sensor_positions


def build_offset_positions(frames: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    return [
        (float(frame["offset_position"][0]), float(frame["offset_position"][1]))
        for frame in frames
    ]


def build_pedestrian_tracks(frames: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tracks: Dict[str, Dict[str, Any]] = {}
    for frame in frames:
        timestamp = float(frame["timestamp"])
        for ped in frame.get("pedestrian_tracks", []):
            track_id = str(ped["track_id"])
            color = tuple(float(v) for v in ped["color"])
            state = ped["filtered_state"]
            track = tracks.setdefault(track_id, {"color": color, "filtered_states": []})
            track["color"] = color
            track["filtered_states"].append(
                (
                    float(state[0]),
                    float(state[1]),
                    float(state[2]),
                    float(state[3]),
                    timestamp,
                )
            )
    return tracks


def build_segments(filtered_states: Sequence[TrackedState]) -> List[Tuple[float, float, float, float, float]]:
    segments = []
    for i in range(1, len(filtered_states)):
        prev_x, prev_y, _, _, _ = filtered_states[i - 1]
        curr_x, curr_y, curr_vx, curr_vy, _ = filtered_states[i]
        speed = float(np.hypot(curr_vx, curr_vy))
        segments.append((prev_x, prev_y, curr_x, curr_y, speed))
    return segments


def draw_on_axis(
    ax,
    sensor_positions: Sequence[Point3],
    offset_positions: Sequence[Tuple[float, float]],
    pedestrian_tracks: Dict[str, Dict[str, Any]],
    title: str,
) -> None:
    ax.clear()

    if sensor_positions:
        sensor_pos_array = np.array([(x, y) for x, y, _ in sensor_positions], dtype=np.float32)
        ax.plot(sensor_pos_array[:, 0], sensor_pos_array[:, 1], color="black", linewidth=2.0, label="sensor trajectory")
        ax.scatter(sensor_pos_array[-1, 0], sensor_pos_array[-1, 1], c="black", s=100, marker="o")

    if offset_positions:
        offset_array = np.array(offset_positions, dtype=np.float32)
        ax.plot(offset_array[:, 0], offset_array[:, 1], color="tab:red", linewidth=2.0, linestyle="--", label="predicted offset trajectory")
        ax.scatter(offset_array[-1, 0], offset_array[-1, 1], c="tab:red", s=100, marker="x")

    for track_id, track in pedestrian_tracks.items():
        color = track["color"]
        filtered_states = track["filtered_states"]
        segments = build_segments(filtered_states)

        for prev_x, prev_y, curr_x, curr_y, speed in segments:
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            scale = min(speed * 0.1, 1.0)
            ax.arrow(prev_x, prev_y, dx * scale, dy * scale, head_width=0.1, head_length=0.2, fc=color, ec=color, alpha=0.6)

        if filtered_states:
            state_array = np.array([(x, y) for x, y, _, _, _ in filtered_states], dtype=np.float32)
            ax.plot(state_array[:, 0], state_array[:, 1], color=color, linewidth=1.5, alpha=0.8)
            curr_x, curr_y, curr_vx, curr_vy, _ = filtered_states[-1]
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

    all_points: List[Tuple[float, float]] = []
    all_points.extend([(x, y) for x, y, _ in sensor_positions])
    all_points.extend(offset_positions)
    for track in pedestrian_tracks.values():
        all_points.extend([(x, y) for x, y, _, _, _ in track["filtered_states"]])

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
    ax.set_title(title)
    ax.legend(loc="upper right")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-path", type=str, default="record.json")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--title", type=str, default="Recorded Sensor / Offset / Pedestrian Trajectories")
    ap.add_argument("--save-path", type=str, default="visualize_record.png")
    args = ap.parse_args()

    frames = load_frames(args.record_path)
    selected_frames = slice_frames(frames, args.start_frame, args.end_frame)
    if not selected_frames:
        raise ValueError("No frames selected. Check --start-frame and --end-frame.")

    sensor_positions = build_sensor_positions(selected_frames)
    offset_positions = build_offset_positions(selected_frames)
    pedestrian_tracks = build_pedestrian_tracks(selected_frames)

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_on_axis(ax, sensor_positions, offset_positions, pedestrian_tracks, args.title)
    fig.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path)
    backend = plt.get_backend().lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
