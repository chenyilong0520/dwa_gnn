#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Sequence, Tuple


Vec2 = Tuple[float, float]


def load_frames(record_path: str) -> List[Dict[str, Any]]:
    with open(record_path, "r", encoding="ascii") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("record.json must contain a top-level 'frames' list.")
    return frames


def extract_original_path(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    path = []
    for frame in frames:
        state = frame["sensor_filtered_state"]
        path.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp": float(frame["timestamp"]),
                "position": (float(state[0]), float(state[1])),
                "velocity": (float(state[2]), float(state[3])),
                "pedestrians": frame.get("pedestrian_tracks", []),
            }
        )
    return path


def estimate_offset_velocities(frames: Sequence[Dict[str, Any]]) -> List[Vec2]:
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]

    positions = [frame["offset_position"] for frame in frames]
    times = [float(frame["timestamp"]) for frame in frames]
    velocities: List[Vec2] = []

    for i in range(n):
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1

        dt = times[j1] - times[j0]
        if dt <= 0.0:
            velocities.append((0.0, 0.0))
            continue

        dx = float(positions[j1][0]) - float(positions[j0][0])
        dy = float(positions[j1][1]) - float(positions[j0][1])
        velocities.append((dx / dt, dy / dt))

    return velocities


def extract_offset_path(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    velocities = estimate_offset_velocities(frames)
    path = []
    for frame, velocity in zip(frames, velocities):
        position = frame["offset_position"]
        path.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp": float(frame["timestamp"]),
                "position": (float(position[0]), float(position[1])),
                "velocity": velocity,
                "pedestrians": frame.get("pedestrian_tracks", []),
            }
        )
    return path


def calculate_path_length(path: Sequence[Dict[str, Any]]) -> float:
    if len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]["position"]
        x1, y1 = path[i]["position"]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def calculate_path_irregularity(path: Sequence[Dict[str, Any]], eps_len: float = 1e-9) -> float:
    n = len(path)
    if n < 2:
        return 0.0

    pts = [item["position"] for item in path]
    path_length = calculate_path_length(path)

    total_abs_turn = 0.0
    prev_h = None
    for i in range(1, n):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        if math.hypot(dx, dy) < eps_len:
            continue
        h = math.atan2(dy, dx)
        if prev_h is not None:
            delta_h = h - prev_h
            while delta_h <= -math.pi:
                delta_h += 2.0 * math.pi
            while delta_h > math.pi:
                delta_h -= 2.0 * math.pi
            total_abs_turn += abs(delta_h)
        prev_h = h

    return (total_abs_turn / path_length) if path_length >= eps_len else 0.0


def calculate_closest_pedestrian_distance(path: Sequence[Dict[str, Any]]) -> float:
    per_waypoint_minima: List[float] = []
    for item in path:
        rx, ry = item["position"]
        waypoint_best = math.inf
        for ped in item["pedestrians"]:
            px, py, _, _ = ped["filtered_state"]
            waypoint_best = min(waypoint_best, math.hypot(float(px) - rx, float(py) - ry))
        if not math.isinf(waypoint_best):
            per_waypoint_minima.append(waypoint_best)

    if not per_waypoint_minima:
        return math.inf
    return sum(per_waypoint_minima) / len(per_waypoint_minima)


def compute_ttc(robot_pos: Vec2, robot_vel: Vec2, ped_pos: Vec2, ped_vel: Vec2, collision_radius: float) -> float:
    rx = ped_pos[0] - robot_pos[0]
    ry = ped_pos[1] - robot_pos[1]
    vx = ped_vel[0] - robot_vel[0]
    vy = ped_vel[1] - robot_vel[1]

    c = rx * rx + ry * ry - collision_radius * collision_radius
    if c <= 0.0:
        return 0.0

    a = vx * vx + vy * vy
    if a <= 1e-12:
        return math.inf

    b = 2.0 * (rx * vx + ry * vy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return math.inf

    sqrt_disc = math.sqrt(disc)
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)

    if t0 >= 0.0:
        return t0
    if t1 >= 0.0:
        return t1
    return math.inf


def calculate_min_ttc(path: Sequence[Dict[str, Any]], collision_radius: float) -> float:
    per_waypoint_minima: List[float] = []
    for item in path:
        robot_pos = item["position"]
        robot_vel = item["velocity"]
        waypoint_best = math.inf
        for ped in item["pedestrians"]:
            px, py, vx, vy = ped["filtered_state"]
            ttc = compute_ttc(
                robot_pos=robot_pos,
                robot_vel=robot_vel,
                ped_pos=(float(px), float(py)),
                ped_vel=(float(vx), float(vy)),
                collision_radius=collision_radius,
            )
            waypoint_best = min(waypoint_best, ttc)
        if not math.isinf(waypoint_best):
            per_waypoint_minima.append(waypoint_best)

    if not per_waypoint_minima:
        return math.inf
    return sum(per_waypoint_minima) / len(per_waypoint_minima)


def evaluate_path(path: Sequence[Dict[str, Any]], collision_radius: float) -> Dict[str, float]:
    return {
        "path_length": calculate_path_length(path),
        "path_irregularity": calculate_path_irregularity(path),
        "closest_pedestrian_distance": calculate_closest_pedestrian_distance(path),
        "minimum_time_to_collision": calculate_min_ttc(path, collision_radius),
    }


def format_metric(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def write_report(
    output_path: str,
    record_path: str,
    frame_count: int,
    collision_radius: float,
    original_metrics: Dict[str, float],
    offset_metrics: Dict[str, float],
) -> None:
    lines = [
        f"record_path: {record_path}",
        f"frame_count: {frame_count}",
        f"collision_radius: {collision_radius:.6f}",
        "",
        "[original_path]",
        f"path_length: {format_metric(original_metrics['path_length'])}",
        f"path_irregularity: {format_metric(original_metrics['path_irregularity'])}",
        f"closest_pedestrian_distance: {format_metric(original_metrics['closest_pedestrian_distance'])}",
        f"minimum_time_to_collision: {format_metric(original_metrics['minimum_time_to_collision'])}",
        "",
        "[offset_path]",
        f"path_length: {format_metric(offset_metrics['path_length'])}",
        f"path_irregularity: {format_metric(offset_metrics['path_irregularity'])}",
        f"closest_pedestrian_distance: {format_metric(offset_metrics['closest_pedestrian_distance'])}",
        f"minimum_time_to_collision: {format_metric(offset_metrics['minimum_time_to_collision'])}",
        "",
    ]

    with open(output_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-path", type=str, default="record.json")
    ap.add_argument("--output-path", type=str, default="evaluation.txt")
    ap.add_argument("--collision-radius", type=float, default=1.0)
    args = ap.parse_args()

    frames = load_frames(args.record_path)
    original_path = extract_original_path(frames)
    offset_path = extract_offset_path(frames)

    original_metrics = evaluate_path(original_path, args.collision_radius)
    offset_metrics = evaluate_path(offset_path, args.collision_radius)

    write_report(
        output_path=args.output_path,
        record_path=args.record_path,
        frame_count=len(frames),
        collision_radius=args.collision_radius,
        original_metrics=original_metrics,
        offset_metrics=offset_metrics,
    )

    print(f"Evaluation complete. Results written to '{args.output_path}'.")

# a standalone script that evaluates a recorded trajectory from a JSON file, computing metrics such as path length, irregularity, closest pedestrian distance, and minimum time-to-collision for both the original and offset-adjusted paths, and writes the results to a text report.
if __name__ == "__main__":
    main()
