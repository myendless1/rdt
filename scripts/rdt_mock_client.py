#!/usr/bin/env python
# coding=utf-8

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.hdf5_vla_dataset import HDF5VLADataset
from scripts.action_mode_utils import get_active_indices


def encode_image_b64(image):
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def post_json(url, payload, timeout_sec):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def build_payload(sample):
    hist = sample["cam_high"].shape[0]
    cam_high = [encode_image_b64(sample["cam_high"][i]) for i in range(hist)]
    cam_right = [encode_image_b64(sample["cam_right_wrist"][i]) for i in range(hist)]
    cam_left = [encode_image_b64(sample["cam_left_wrist"][i]) for i in range(hist)]

    payload = {
        "instruction": sample["meta"]["instruction"],
        "state": sample["state"][0].astype(np.float32).tolist(),
        "images": {
            "cam_high": cam_high,
            "cam_right_wrist": cam_right,
            "cam_left_wrist": cam_left,
        },
        "norm_stats": {
            "state_mean": sample["state_mean"].astype(np.float32).tolist(),
            "state_std": np.maximum(sample["state_std"].astype(np.float32), 1e-6).tolist(),
            "action_mean": sample["action_mean"].astype(np.float32).tolist(),
            "action_std": np.maximum(sample["action_std"].astype(np.float32), 1e-6).tolist(),
        },
    }
    return payload


def parse_args():
    parser = argparse.ArgumentParser("Mock client for RDT REST server")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--dataset_path", type=str, default="/media/damoxing/datasets/myendless/random-rack2")
    parser.add_argument("--action_mode", type=str, default="eef_pose", choices=["joint", "eef_pose"])
    parser.add_argument("--action_target", type=str, default="delta", choices=["delta", "absolute"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--timeout_sec", type=float, default=60.0)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    ds = HDF5VLADataset(
        dataset_path=args.dataset_path,
        action_mode=args.action_mode,
        action_target=args.action_target,
    )
    active_indices = get_active_indices(args.action_mode)

    print("=== Mock Client Evaluation ===")
    print(f"server_url={args.server_url}")
    print(f"action_mode={args.action_mode}")
    print(f"action_target={args.action_target}")
    print(f"dataset_size={len(ds)}")
    print(f"active_indices({len(active_indices)}): {active_indices}")

    losses = []
    maes = []
    latencies = []
    sample_reports = []

    n = min(args.num_samples, len(ds))
    for i in range(n):
        ok, sample = ds.parse_hdf5_file(ds.file_paths[i])
        if not ok:
            continue

        payload = build_payload(sample)
        payload["action_target"] = args.action_target

        t0 = time.time()
        resp = post_json(f"{args.server_url}/infer", payload, args.timeout_sec)
        elapsed_ms = (time.time() - t0) * 1000.0

        if not resp.get("ok", False):
            raise RuntimeError(f"Server infer failed: {resp}")

        pred = np.array(resp["action"], dtype=np.float32)
        gt = sample["actions"].astype(np.float32)
        idx = np.array(active_indices, dtype=np.int64)

        pred_valid = pred[:, idx]
        gt_valid = gt[:, idx]

        mse = float(np.mean((pred_valid - gt_valid) ** 2))
        mae = float(np.mean(np.abs(pred_valid - gt_valid)))

        losses.append(mse)
        maes.append(mae)
        latencies.append(float(resp.get("latency_ms", elapsed_ms)))

        report = {
            "sample_index": i,
            "step_id": int(sample["meta"]["step_id"]),
            "mse_valid": mse,
            "mae_valid": mae,
            "latency_ms": float(resp.get("latency_ms", elapsed_ms)),
        }
        sample_reports.append(report)
        print(
            f"sample={i:03d} step_id={report['step_id']} "
            f"mse_valid={mse:.6f} mae_valid={mae:.6f} latency_ms={report['latency_ms']:.2f}"
        )

    if not losses:
        raise RuntimeError("No valid samples were evaluated")

    summary = {
        "action_mode": args.action_mode,
        "action_target": args.action_target,
        "num_evaluated": len(losses),
        "active_indices": active_indices,
        "mse_valid_mean": float(np.mean(losses)),
        "mse_valid_std": float(np.std(losses)),
        "mae_valid_mean": float(np.mean(maes)),
        "mae_valid_std": float(np.std(maes)),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        # "samples": sample_reports,
    }

    print("=== Summary ===")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    import debugpy; debugpy.listen(5678); print("Waiting for debugger to attach..."); debugpy.wait_for_client()
    main()
