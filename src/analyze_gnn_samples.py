#!/usr/bin/env python3
"""
Lightweight dataset / error analysis for the GNN regression task.

This file is intentionally standalone so it is easy to delete later.

What it does:
1. Summarizes label magnitude statistics for train/val splits.
2. Optionally loads a checkpoint and reports the highest-loss samples.

Example:
  python src/analyze_gnn_samples.py
  python src/analyze_gnn_samples.py --checkpoint src/gnn_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils import discover_npz_files, load_model, load_samples_from_npz


class Tee:
    """Mirror stdout to both the terminal and a text file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def extract_sequence_number(npz_path: str) -> int:
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    match = re.search(r"(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not extract sequence number from '{npz_path}'")
    return int(match.group(1))


def load_filtered_sequence_numbers(json_path: str) -> List[int]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        for key in ("filtered_files", "files", "sequence_numbers", "selected"):
            if key in payload:
                values = payload[key]
                break
        else:
            raise ValueError(f"Unsupported JSON structure in {json_path}")
    else:
        raise ValueError(f"Unsupported JSON structure in {json_path}")

    return [int(v) for v in values]


def split_npz_files(npz_files: Sequence[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    num_seqs = len(npz_files)
    num_val = max(1, int(round(num_seqs * val_ratio)))
    val_files = list(npz_files[-num_val:])
    train_files = list(npz_files[:-num_val])
    return train_files, val_files


def load_split_samples(npz_files: Sequence[str]) -> List[Tuple[str, object]]:
    samples: List[Tuple[str, object]] = []
    for npz_path in npz_files:
        for sample in load_samples_from_npz(npz_path):
            samples.append((npz_path, sample))
    return samples


def format_percentiles(values: np.ndarray, name: str) -> str:
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    return (
        f"{name}: min={values.min():.6f} mean={values.mean():.6f} max={values.max():.6f} "
        f"p50={p50:.6f} p90={p90:.6f} p95={p95:.6f} p99={p99:.6f}"
    )


def summarize_labels(samples: Sequence[Tuple[str, object]], split_name: str) -> None:
    ys = np.stack([sample.y.view(-1).cpu().numpy() for _, sample in samples], axis=0)
    norms = np.linalg.norm(ys, axis=1)

    print(f"\n[{split_name}] label summary")
    print(f"samples={len(samples)}")
    print(format_percentiles(ys[:, 0], "dx"))
    print(format_percentiles(ys[:, 1], "dy"))
    print(format_percentiles(norms, "||y||"))

    top_indices = np.argsort(norms)[-10:][::-1]
    print("top-10 largest ||y|| samples:")
    for rank, idx in enumerate(top_indices, start=1):
        npz_path, sample = samples[idx]
        meta = getattr(sample, "meta", {}) or {}
        print(
            f"{rank:02d}. ||y||={norms[idx]:.6f} y={ys[idx].tolist()} "
            f"seq={extract_sequence_number(npz_path)} frame={meta.get('frame')} "
            f"dmin={meta.get('dmin')} xml={meta.get('xml_path')}"
        )


def compute_sample_losses(
    model: torch.nn.Module,
    samples: Sequence[Tuple[str, object]],
    device: torch.device,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    model.eval()

    with torch.no_grad():
        for npz_path, sample in samples:
            sample_on_device = sample.to(device)
            pred = model(sample_on_device).view(-1, 2)
            target = sample_on_device.y.view(-1, 2)
            mse = F.mse_loss(pred, target, reduction="mean").item()
            mae = F.l1_loss(pred, target, reduction="mean").item()

            pred_cpu = pred.squeeze(0).cpu().numpy()
            target_cpu = target.squeeze(0).cpu().numpy()
            meta = getattr(sample, "meta", {}) or {}
            results.append(
                {
                    "npz_path": npz_path,
                    "sequence": extract_sequence_number(npz_path),
                    "frame": meta.get("frame"),
                    "xml_path": meta.get("xml_path"),
                    "dmin": meta.get("dmin"),
                    "mse": mse,
                    "mae": mae,
                    "pred": pred_cpu,
                    "target": target_cpu,
                    "target_norm": float(np.linalg.norm(target_cpu)),
                }
            )
    return results


def print_loss_summary(losses: Sequence[Dict[str, object]], split_name: str, top_k: int) -> None:
    mse_values = np.array([item["mse"] for item in losses], dtype=np.float64)
    mae_values = np.array([item["mae"] for item in losses], dtype=np.float64)

    print(f"\n[{split_name}] model error summary")
    print(format_percentiles(mse_values, "sample_mse"))
    print(format_percentiles(mae_values, "sample_mae"))

    ranked = sorted(losses, key=lambda item: float(item["mse"]), reverse=True)[:top_k]
    print(f"top-{min(top_k, len(ranked))} highest-MSE samples:")
    for rank, item in enumerate(ranked, start=1):
        pred = np.asarray(item["pred"]).tolist()
        target = np.asarray(item["target"]).tolist()
        print(
            f"{rank:02d}. mse={item['mse']:.6f} mae={item['mae']:.6f} "
            f"target_norm={item['target_norm']:.6f} pred={pred} target={target} "
            f"seq={item['sequence']} frame={item['frame']} dmin={item['dmin']} "
            f"xml={item['xml_path']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default="data/data_processed")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--filtered_files_json", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_txt", type=str, default="analyze_gnn_samples.txt")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    original_stdout = sys.stdout
    with open(args.output_txt, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            run_analysis(args)
        finally:
            sys.stdout = original_stdout


def run_analysis(args: argparse.Namespace) -> None:

    npz_files = discover_npz_files(args.npz_dir)
    if not npz_files:
        raise FileNotFoundError(f"No gnn_dataset_*.npz found in {args.npz_dir}")

    if args.filtered_files_json is not None:
        selected_numbers = set(load_filtered_sequence_numbers(args.filtered_files_json))
        npz_files = [
            path for path in npz_files
            if extract_sequence_number(path) in selected_numbers
        ]
        if not npz_files:
            raise FileNotFoundError(
                "No gnn_dataset_*.npz matched the sequence numbers from "
                f"{args.filtered_files_json}"
            )

    train_files, val_files = split_npz_files(npz_files, args.val_ratio)
    train_samples = load_split_samples(train_files)
    val_samples = load_split_samples(val_files)

    print(f"Found sequences: {len(npz_files)}")
    print(f"Train sequences: {len(train_files)}")
    print(f"Val sequences: {len(val_files)}")

    summarize_labels(train_samples, "train")
    summarize_labels(val_samples, "val")

    if args.checkpoint is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device)

    train_losses = compute_sample_losses(model, train_samples, device)
    val_losses = compute_sample_losses(model, val_samples, device)
    print_loss_summary(train_losses, "train", args.top_k)
    print_loss_summary(val_losses, "val", args.top_k)


if __name__ == "__main__":
    main()
