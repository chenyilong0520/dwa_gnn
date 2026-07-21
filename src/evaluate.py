#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import ConstantVelocityEKF


Vec2 = Tuple[float, float]


def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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


def extract_offset_path(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    path = []
    for frame in frames:
        position = frame["offset_position"]
        sensor_state = frame["sensor_filtered_state"]
        path.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp": float(frame["timestamp"]),
                "position": (float(position[0]), float(position[1])),
                "velocity": (float(sensor_state[2]), float(sensor_state[3])),
                "pedestrians": frame.get("pedestrian_tracks", []),
            }
        )
    return path


def filter_offset_path(path: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not path:
        return []

    ekf = ConstantVelocityEKF(
        process_noise_position=3,
        process_noise_velocity=1.0, 
        measurement_noise=0.002)
    
    filtered_path: List[Dict[str, Any]] = []
    prev_timestamp: Optional[float] = None

    for item in path:
        timestamp = float(item["timestamp"])
        curr_x, curr_y = item["position"]

        if ekf.state is None:
            ekf.initialize(float(curr_x), float(curr_y))
        elif prev_timestamp is not None:
            ekf.predict(timestamp - prev_timestamp)

        filtered_state = ekf.update(np.array([curr_x, curr_y], dtype=np.float64))
        filtered_item = dict(item)
        filtered_item["position"] = (float(filtered_state[0]), float(filtered_state[1]))
        filtered_path.append(filtered_item)
        prev_timestamp = timestamp

    return filtered_path


def compute_social_force_offset(position: Vec2, pedestrians: Sequence[Dict[str, Any]]) -> Vec2:
    total_fx = 0.0
    total_fy = 0.0
    robot_x, robot_y = position

    for ped in pedestrians:
        px, py, vx, vy = ped["filtered_state"]
        diff_x = robot_x - (float(px) + float(vx))
        diff_y = robot_y - (float(py) + float(vy))
        dist = math.hypot(diff_x, diff_y)
        if dist < 1e-3:
            continue

        direction_x = diff_x / dist
        direction_y = diff_y / dist
        force_mag = math.exp(-dist / 0.5) * 1.0
        total_fx += force_mag * direction_x
        total_fy += force_mag * direction_y

    return (total_fx, total_fy)


def extract_social_force_offset_path(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    path = []
    for frame in frames:
        sensor_state = frame["sensor_filtered_state"]
        sensor_pos = (float(sensor_state[0]), float(sensor_state[1]))
        pedestrians = frame.get("pedestrian_tracks", [])
        offset = compute_social_force_offset(sensor_pos, pedestrians)
        path.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp": float(frame["timestamp"]),
                "position": (sensor_pos[0] + offset[0], sensor_pos[1] + offset[1]),
                "velocity": (float(sensor_state[2]), float(sensor_state[3])),
                "pedestrians": pedestrians,
            }
        )
    return path


def summarize_samples(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        return {
            "mean": math.inf,
            "std": math.inf,
        }

    mean = sum(samples) / len(samples)
    variance = sum((value - mean) * (value - mean) for value in samples) / len(samples)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
    }


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


def iter_pedestrians_within_distance(
    pedestrians: Sequence[Dict[str, Any]],
    robot_pos: Vec2,
    max_sensor_distance: float,
):
    rx, ry = robot_pos
    for ped in pedestrians:
        px, py, _, _ = ped["filtered_state"]
        distance = math.hypot(float(px) - rx, float(py) - ry)
        if distance <= max_sensor_distance:
            yield ped


def calculate_closest_pedestrian_distance(
    path: Sequence[Dict[str, Any]],
    max_sensor_distance: float,
) -> Dict[str, float]:
    per_waypoint_minima: List[float] = []
    for item in path:
        rx, ry = item["position"]
        waypoint_best = math.inf
        for ped in iter_pedestrians_within_distance(item["pedestrians"], (rx, ry), max_sensor_distance):
            px, py, _, _ = ped["filtered_state"]
            waypoint_best = min(waypoint_best, math.hypot(float(px) - rx, float(py) - ry))
        if not math.isinf(waypoint_best):
            per_waypoint_minima.append(waypoint_best)

    return summarize_samples(per_waypoint_minima)


def calculate_min_ttc(
    path: Sequence[Dict[str, Any]],
    collision_radius: float,
    max_sensor_distance: float,
) -> Dict[str, float]:
    per_waypoint_minima: List[float] = []
    collision_count = 0
    for item in path:
        robot_pos = item["position"]
        robot_vel = item["velocity"]
        waypoint_best = math.inf
        for ped in iter_pedestrians_within_distance(item["pedestrians"], robot_pos, max_sensor_distance):
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
            collision_count += 1

    stats = summarize_samples(per_waypoint_minima)
    stats["collision_count"] = collision_count
    return stats


def evaluate_path(path: Sequence[Dict[str, Any]], collision_radius: float, max_sensor_distance: float) -> Dict[str, Any]:
    return {
        "path_length": calculate_path_length(path),
        "path_irregularity": calculate_path_irregularity(path),
        "closest_pedestrian_distance": calculate_closest_pedestrian_distance(path, max_sensor_distance),
        "minimum_time_to_collision": calculate_min_ttc(path, collision_radius, max_sensor_distance),
    }


def format_metric(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def format_mean_std(stats: Dict[str, float]) -> str:
    return f"{format_metric(stats['mean'])}, \u00b1 {format_metric(stats['std'])}"


def format_mean_std_collision(stats: Dict[str, float]) -> str:
    return f"{format_metric(stats['mean'])}, \u00b1 {format_metric(stats['std'])}, ({int(stats['collision_count'])})"


def write_report(
    output_path: str,
    record_path: str,
    frame_count: int,
    collision_radius: float,
    max_sensor_distance: float,
    original_metrics: Dict[str, Any],
    offset_metrics: Dict[str, Any],
    social_force_offset_metrics: Dict[str, Any],
) -> None:
    lines = [
        f"record_path: {record_path}",
        f"frame_count: {frame_count}",
        f"collision_radius: {collision_radius:.6f}",
        f"max_sensor_distance: {max_sensor_distance:.6f}",
        "",
        "[original_path]",
        f"path_length: {format_metric(original_metrics['path_length'])}",
        f"path_irregularity: {format_metric(original_metrics['path_irregularity'])}",
        f"closest_pedestrian_distance: {format_mean_std(original_metrics['closest_pedestrian_distance'])}",
        f"minimum_time_to_collision: {format_mean_std_collision(original_metrics['minimum_time_to_collision'])}",
        "",
        "[offset_path]",
        f"path_length: {format_metric(offset_metrics['path_length'])}",
        f"path_irregularity: {format_metric(offset_metrics['path_irregularity'])}",
        f"closest_pedestrian_distance: {format_mean_std(offset_metrics['closest_pedestrian_distance'])}",
        f"minimum_time_to_collision: {format_mean_std_collision(offset_metrics['minimum_time_to_collision'])}",
        "",
        "[social_force_offset_path]",
        f"path_length: {format_metric(social_force_offset_metrics['path_length'])}",
        f"path_irregularity: {format_metric(social_force_offset_metrics['path_irregularity'])}",
        f"closest_pedestrian_distance: {format_mean_std(social_force_offset_metrics['closest_pedestrian_distance'])}",
        f"minimum_time_to_collision: {format_mean_std_collision(social_force_offset_metrics['minimum_time_to_collision'])}",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-path", type=str, default="record.json")
    ap.add_argument("--output-path", type=str, default="evaluation.txt")
    ap.add_argument("--collision-radius", type=float, default=1.0)
    ap.add_argument("--max-sensor-distance", dest="max_sensor_distance", type=float, default=50.0)
    ap.add_argument("--ekf", type=parse_bool_arg, nargs="?", const=True, default=True)
    ap.add_argument("--no-ekf", dest="ekf", action="store_false")
    args = ap.parse_args()

    frames = load_frames(args.record_path)
    original_path = extract_original_path(frames)
    offset_path = extract_offset_path(frames)
    if args.ekf:
        offset_path = filter_offset_path(offset_path)
    social_force_offset_path = extract_social_force_offset_path(frames)

    original_metrics = evaluate_path(original_path, args.collision_radius, args.max_sensor_distance)
    offset_metrics = evaluate_path(offset_path, args.collision_radius, args.max_sensor_distance)
    social_force_offset_metrics = evaluate_path(social_force_offset_path, args.collision_radius, args.max_sensor_distance)

    write_report(
        output_path=args.output_path,
        record_path=args.record_path,
        frame_count=len(frames),
        collision_radius=args.collision_radius,
        max_sensor_distance=args.max_sensor_distance,
        original_metrics=original_metrics,
        offset_metrics=offset_metrics,
        social_force_offset_metrics=social_force_offset_metrics,
    )

    print(f"Evaluation complete. Results written to '{args.output_path}'.")

# a standalone script that evaluates a recorded trajectory from a JSON file, computing metrics such as path length, irregularity, closest pedestrian distance, and minimum time-to-collision for both the original and offset-adjusted paths, and writes the results to a text report.
if __name__ == "__main__":
    main()
