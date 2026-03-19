#!/usr/bin/env python
# coding=utf-8

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from tqdm import trange, tqdm

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

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
    }
    return payload


def pick_grid(n_dims, max_cols=4):
    best = None
    for cols in range(1, min(max_cols, n_dims) + 1):
        rows = int(np.ceil(n_dims / float(cols)))
        empty = rows * cols - n_dims
        score = (empty, abs(rows - cols), cols)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    _, rows, cols = best
    return rows, cols


def load_episode_for_eval(ds, file_path):
    with h5py.File(file_path, "r") as f:
        parsed = ds._load_bimanual_from_hdf5(f)
        left_arm = parsed["left_arm"]
        right_arm = parsed["right_arm"]
        left_gripper = parsed["left_gripper"]
        right_gripper = parsed["right_gripper"]
        cmd_left_arm = parsed["cmd_left_arm"]
        cmd_right_arm = parsed["cmd_right_arm"]
        cmd_left_gripper = parsed["cmd_left_gripper"]
        cmd_right_gripper = parsed["cmd_right_gripper"]
        num_steps = parsed["num_steps"]

        if num_steps < 128:
            return False, None

        eps = 1e-3
        arm_state = np.concatenate([left_arm[:num_steps], right_arm[:num_steps]], axis=-1)
        arm_delta = np.abs(arm_state - arm_state[0:1])
        idx = np.where(np.any(arm_delta > eps, axis=1))[0]
        first_idx = int(idx[0]) if len(idx) > 0 else 0
        start = max(first_idx - 1, 0)

        state = ds._fill_bimanual_state(
            left_arm[start:num_steps],
            left_gripper[start:num_steps],
            right_arm[start:num_steps],
            right_gripper[start:num_steps],
        )
        command = ds._fill_bimanual_state(
            cmd_left_arm[start:num_steps],
            cmd_left_gripper[start:num_steps],
            cmd_right_arm[start:num_steps],
            cmd_right_gripper[start:num_steps],
        )
        action = ds._build_action_target(command, state)

        instruction = "Use right arm to perform rack manipulation while keeping left-arm joints fixed."

        return True, {
            "start": start,
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "instruction": instruction,
        }


def build_payload_from_step(ds, h5_file, abs_step, instruction, state_step):
    cam_high, _ = ds._parse_img(h5_file, "head", abs_step)
    cam_left_wrist, _ = ds._parse_img(h5_file, "left", abs_step)
    cam_right_wrist, _ = ds._parse_img(h5_file, "right", abs_step)

    hist = cam_high.shape[0]
    payload = {
        "instruction": instruction,
        "state": state_step.astype(np.float32).tolist(),
        "images": {
            "cam_high": [encode_image_b64(cam_high[i]) for i in range(hist)],
            "cam_right_wrist": [encode_image_b64(cam_right_wrist[i]) for i in range(hist)],
            "cam_left_wrist": [encode_image_b64(cam_left_wrist[i]) for i in range(hist)],
        },
    }
    return payload


def render_plot_frame(gt_all, pred_all, active_indices, t):
    n_dims = len(active_indices)
    rows, cols = pick_grid(n_dims)
    fig = Figure(figsize=(4.2 * cols, 2.6 * rows), dpi=100)
    canvas = FigureCanvas(fig)
    x = np.arange(t + 1, dtype=np.int64)

    for i, dim_idx in enumerate(active_indices):
        ax = fig.add_subplot(rows, cols, i + 1)

        gt = gt_all[: t + 1, dim_idx]
        pred = pred_all[: t + 1, dim_idx]

        ax.plot(x, gt, color="green", linewidth=1.0, label="gt")
        ax.plot(x, pred, color="red", linewidth=1.0, label="pred")
        ax.set_xlim(0, gt_all.shape[0] - 1)
        ax.set_title(f"dim_{dim_idx}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    total = rows * cols
    for i in range(n_dims, total):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.axis("off")

    fig.tight_layout()
    canvas.draw()
    w, h = canvas.get_width_height()
    img_rgb = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return img_rgb


def write_gt_vs_pred_video(gt_all, pred_all, active_indices, out_path, fps):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(
        out_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        macro_block_size=None,
    )
    try:
        for t in range(gt_all.shape[0]):
            frame = render_plot_frame(gt_all, pred_all, active_indices, t)
            writer.append_data(frame)
    finally:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser("Mock client for RDT REST server")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--dataset_path", type=str, default="/media/damoxing/datasets/myendless/random-rack2")
    parser.add_argument("--action_mode", type=str, default="eef_pose", choices=["joint", "eef_pose"])
    parser.add_argument("--action_target", type=str, default="delta", choices=["delta", "absolute"])
    parser.add_argument("--max_episodes", type=int, default=0, help="0 means evaluate all episodes")
    parser.add_argument("--timeout_sec", type=float, default=6000.0)
    parser.add_argument("--output_json", type=str, default="logs/rdt_eval_full_dataset.json")
    parser.add_argument("--output_video", type=str, default="logs/rdt_eval_full_dataset.mp4")
    parser.add_argument("--video_fps", type=int, default=20)
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

    losses_by_dim = []
    maes_by_dim = []
    latencies = []
    gt_all = []
    pred_all = []

    max_episodes = len(ds.file_paths) if args.max_episodes <= 0 else min(args.max_episodes, len(ds.file_paths))

    for ep_idx, file_path in enumerate(ds.file_paths[:max_episodes]):
        ok, episode = load_episode_for_eval(ds, file_path)
        if not ok:
            continue

        state = episode["state"]
        action = episode["action"]
        instruction = episode["instruction"]
        start = int(episode["start"])

        with h5py.File(file_path, "r") as h5_file:
            infer_indices = np.arange(state.shape[0])[::5] # Infer every 5 steps to reduce latency
            for local_t in tqdm(infer_indices, desc=f"Episode {ep_idx + 1}/{max_episodes}", unit="step"):
                abs_step = start + local_t

                payload = build_payload_from_step(
                    ds=ds,
                    h5_file=h5_file,
                    abs_step=abs_step,
                    instruction=instruction,
                    state_step=state[local_t],
                )
                payload["action_target"] = args.action_target

                t0 = time.time()
                resp = post_json(f"{args.server_url}/infer", payload, args.timeout_sec)
                elapsed_ms = (time.time() - t0) * 1000.0

                if not resp.get("ok", False):
                    raise RuntimeError(f"Server infer failed on episode={ep_idx} step={abs_step}: {resp}")

                pred = np.array(resp["action"], dtype=np.float32)
                pred_t0 = pred[0]
                gt_t0 = action[local_t].astype(np.float32)

                pred_all.append(pred_t0)
                gt_all.append(gt_t0)
                latencies.append(float(resp.get("latency_ms", elapsed_ms)))

        print(
            f"episode={ep_idx + 1:04d}/{max_episodes:04d} path={os.path.basename(file_path)} "
            f"frames={action.shape[0]}"
        )

    if not pred_all:
        raise RuntimeError("No valid frames were evaluated")

    pred_all = np.stack(pred_all, axis=0)
    gt_all = np.stack(gt_all, axis=0)

    idx = np.array(active_indices, dtype=np.int64)
    err = pred_all[:, idx] - gt_all[:, idx]
    mse_per_dim = np.mean(err ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(err), axis=0)
    losses_by_dim.extend(mse_per_dim.tolist())
    maes_by_dim.extend(mae_per_dim.tolist())

    write_gt_vs_pred_video(gt_all, pred_all, active_indices, args.output_video, args.video_fps)

    summary = {
        "action_mode": args.action_mode,
        "action_target": args.action_target,
        "num_episodes_evaluated": max_episodes,
        "num_frames_evaluated": int(pred_all.shape[0]),
        "active_indices": active_indices,
        "mse_valid_mean": float(np.mean(losses_by_dim)),
        "mse_valid_std": float(np.std(losses_by_dim)),
        "mae_valid_mean": float(np.mean(maes_by_dim)),
        "mae_valid_std": float(np.std(maes_by_dim)),
        "mse_valid_per_dim": [float(x) for x in mse_per_dim],
        "mae_valid_per_dim": [float(x) for x in mae_per_dim],
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        "output_video": args.output_video,
        "video_fps": args.video_fps,
    }

    print("=== Summary ===")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
