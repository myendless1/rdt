#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sys

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from configs.state_vec import STATE_VEC_IDX_MAPPING
from data.hdf5_vla_dataset import HDF5VLADataset
from scripts.action_mode_utils import get_active_indices


IDX_TO_STATE_KEY = {v: k for k, v in STATE_VEC_IDX_MAPPING.items()}
LEFT_GRIPPER_IDX = STATE_VEC_IDX_MAPPING["left_gripper_open"]
RIGHT_GRIPPER_IDX = STATE_VEC_IDX_MAPPING["right_gripper_open"]


def parse_args():
    parser = argparse.ArgumentParser("Sample and visualize HDF5 action trajectories")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/media/damoxing/datasets/myendless/random-rack2",
        help="Directory containing HDF5 episodes.",
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="eef_pose",
        choices=["joint", "eef_pose"],
        help="Action mode used to decode the dataset.",
    )
    parser.add_argument(
        "--action_target",
        type=str,
        default="delta",
        choices=["delta", "absolute"],
        help="Action target to visualize: delta action or absolute command action.",
    )
    parser.add_argument("--num_samples", type=int, default=4, help="Number of episodes to visualize.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode sampling.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=400,
        help="Maximum timesteps per exported video. <=0 means using full length.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--top_height", type=int, default=360, help="Height of top video panel.")
    parser.add_argument("--plot_height", type=int, default=300, help="Height of bottom plot panel.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/action_viz",
        help="Directory to save output videos.",
    )
    return parser.parse_args()


def decode_frame(rgb_bytes, rgb_sizes, frame_id):
    end = int(np.sum(rgb_sizes[: frame_id + 1]))
    start = end - int(rgb_sizes[frame_id])
    frame = rgb_bytes[start:end]
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)


def select_main_camera_key(file_obj):
    if "images_dict" not in file_obj:
        return None

    image_keys = list(file_obj["images_dict"].keys())
    if not image_keys:
        return None

    preferred = ["head", "cam_high", "main", "front", "rgb"]
    for key in preferred:
        if key in image_keys:
            return key

    return sorted(image_keys)[0]


def compute_start_index(parsed):
    eps = 1e-3
    arm_state = np.concatenate([parsed["left_arm"][: parsed["num_steps"]], parsed["right_arm"][: parsed["num_steps"]]], axis=-1)
    arm_delta = np.abs(arm_state - arm_state[0:1])
    idx = np.where(np.any(arm_delta > eps, axis=1))[0]
    first_idx = int(idx[0]) if len(idx) > 0 else 0
    return max(first_idx - 1, 0)


def load_episode_frames_and_actions(ds, file_path, max_steps):
    with h5py.File(file_path, "r") as f:
        parsed = ds._load_bimanual_from_hdf5(f)
        num_steps = parsed["num_steps"]
        if num_steps < 2:
            raise RuntimeError("Episode too short")

        state = ds._fill_bimanual_state(
            parsed["left_arm"][:num_steps],
            parsed["left_gripper"][:num_steps],
            parsed["right_arm"][:num_steps],
            parsed["right_gripper"][:num_steps],
        )
        command = ds._fill_bimanual_state(
            parsed["cmd_left_arm"][:num_steps],
            parsed["cmd_left_gripper"][:num_steps],
            parsed["cmd_right_arm"][:num_steps],
            parsed["cmd_right_gripper"][:num_steps],
        )
        action = ds._build_action_target(command, state)
        # Keep gripper channels as absolute commands when visualizing delta target.
        if ds.action_target == "delta":
            action[:, LEFT_GRIPPER_IDX] = command[:, LEFT_GRIPPER_IDX]
            action[:, RIGHT_GRIPPER_IDX] = command[:, RIGHT_GRIPPER_IDX]

        start = compute_start_index(parsed)
        action = action[start:num_steps]

        cam_key = select_main_camera_key(f)
        if cam_key is None:
            raise RuntimeError("No images_dict camera key found")

        rgb_bytes = f["images_dict"][cam_key]["rgb"][:]
        rgb_sizes = f["images_dict"][cam_key]["rgb_size"][:].astype(np.int64)

        available_img_steps = max(0, len(rgb_sizes) - start)
        seq_len = min(len(action), available_img_steps)
        if max_steps > 0:
            seq_len = min(seq_len, max_steps)

        if seq_len < 2:
            raise RuntimeError("Not enough aligned action/image steps")

        frames = []
        fallback_shape = None
        for step in range(start, start + seq_len):
            img = decode_frame(rgb_bytes, rgb_sizes, step)
            if img is None:
                if fallback_shape is None:
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                else:
                    img = np.zeros(fallback_shape, dtype=np.uint8)
            else:
                fallback_shape = img.shape
            frames.append(img)

        return frames, action[:seq_len], cam_key


def pick_grid(n_dims, max_cols=4):
    best = None
    for cols in range(1, min(max_cols, n_dims) + 1):
        rows = int(np.ceil(n_dims / float(cols)))
        empty = rows * cols - n_dims
        # Prefer fewer empty cells, then closer-to-square grids.
        score = (empty, abs(rows - cols), cols)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    _, rows, cols = best
    return rows, cols


def build_display_labels(action_mode, active_indices):
    """Build human-readable labels for each active dimension index."""
    labels = {}

    if action_mode == "joint":
        for dim_idx in active_indices:
            idx = int(dim_idx)
            labels[idx] = IDX_TO_STATE_KEY.get(idx, f"idx_{idx}")
        return labels

    # eef_pose is stored in unified EEF slots; show semantic labels.
    num_active = len(active_indices)
    if num_active < 4:
        for dim_idx in active_indices:
            idx = int(dim_idx)
            labels[idx] = IDX_TO_STATE_KEY.get(idx, f"idx_{idx}")
        return labels

    arm_dim = (num_active - 2) // 2
    if arm_dim == 7:
        per_arm_names = ["eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw"]
    elif arm_dim == 6:
        per_arm_names = ["eef_x", "eef_y", "eef_z", "eef_r0", "eef_r1", "eef_r2"]
    else:
        per_arm_names = [f"eef_{i}" for i in range(arm_dim)]

    left_indices = [int(x) for x in active_indices[:arm_dim]]
    right_indices = [int(x) for x in active_indices[arm_dim:2 * arm_dim]]
    left_gripper = int(active_indices[-2])
    right_gripper = int(active_indices[-1])

    for i, idx in enumerate(left_indices):
        labels[idx] = f"left_{per_arm_names[i]}"
    for i, idx in enumerate(right_indices):
        labels[idx] = f"right_{per_arm_names[i]}"
    labels[left_gripper] = "left_gripper_open"
    labels[right_gripper] = "right_gripper_open"

    return labels


def get_axis_limits(action_valid, active_indices, display_labels):
    """Build fixed y-limits per active dim, sharing limits for left/right pairs."""
    label_to_pos = {}
    for pos, dim_idx in enumerate(active_indices):
        idx = int(dim_idx)
        label_to_pos[display_labels.get(idx, "")] = pos

    limits = {}
    for pos, dim_idx in enumerate(active_indices):
        dim_idx = int(dim_idx)
        if dim_idx in limits:
            continue

        dim_key = display_labels.get(dim_idx, "")
        pair_pos = None
        if dim_key.startswith("left_"):
            pair_key = "right_" + dim_key[len("left_"):]
            pair_pos = label_to_pos.get(pair_key)
        elif dim_key.startswith("right_"):
            pair_key = "left_" + dim_key[len("right_"):]
            pair_pos = label_to_pos.get(pair_key)

        if pair_pos is not None:
            series = np.concatenate([action_valid[:, pos], action_valid[:, pair_pos]], axis=0)
            y_min = float(np.min(series))
            y_max = float(np.max(series))
            limits[int(active_indices[pos])] = (y_min, y_max)
            limits[int(active_indices[pair_pos])] = (y_min, y_max)
        else:
            series = action_valid[:, pos]
            limits[int(active_indices[pos])] = (float(np.min(series)), float(np.max(series)))

    return limits


def render_plot_image(action_valid, active_indices, display_labels, axis_limits, t, width, height):
    fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvas(fig)

    n_dims = action_valid.shape[1]
    rows, cols = pick_grid(n_dims)
    x = np.arange(t + 1)

    for dim in range(n_dims):
        ax = fig.add_subplot(rows, cols, dim + 1)
        y = action_valid[: t + 1, dim]
        ax.plot(x, y, color="#1f77b4", linewidth=1.2)
        ax.set_xlim(0, action_valid.shape[0] - 1)

        dim_idx = int(active_indices[dim])
        y_min, y_max = axis_limits.get(dim_idx, (-1.0, 1.0))
        y_abs = max(abs(y_min), abs(y_max))
        if y_abs < 1e-6:
            y_abs = 1.0
        ax.set_ylim(-y_abs * 1.05, y_abs * 1.05)

        dim_key = display_labels.get(dim_idx, f"idx_{dim_idx}")
        ax.set_title(dim_key, fontsize=6)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=5)
        ax.margins(x=0.0)

        # Reduce border clutter so each subplot uses maximum visual area.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if dim // cols == rows - 1:
            ax.set_xlabel("t", fontsize=6)
        if dim % cols == 0:
            ax.set_ylabel("a", fontsize=6)

    # Hide unused panels in the last row.
    total_axes = rows * cols
    for dim in range(n_dims, total_axes):
        ax = fig.add_subplot(rows, cols, dim + 1)
        ax.axis("off")

    fig.subplots_adjust(left=0.07, right=0.998, top=0.95, bottom=0.07, wspace=0.18, hspace=0.35)

    canvas.draw()
    w, h = canvas.get_width_height()
    img_rgb = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def write_visualization_video(out_path, frames, actions, active_indices, action_mode, action_target, fps, top_height, plot_height):
    action_valid = actions[:, active_indices]
    display_labels = build_display_labels(action_mode, active_indices)
    axis_limits = get_axis_limits(action_valid, active_indices, display_labels)

    h0, w0 = frames[0].shape[:2]
    out_w = int(round(w0 * (top_height / float(h0))))

    writer = imageio.get_writer(
        out_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        macro_block_size=None,
    )

    try:
        for t in range(len(frames)):
            top = cv2.resize(frames[t], (out_w, top_height), interpolation=cv2.INTER_AREA)
            cv2.putText(
                top,
                f"mode={action_mode} target={action_target} frame={t + 1}/{len(frames)}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            plot_img = render_plot_image(action_valid, active_indices, display_labels, axis_limits, t, out_w, plot_height)
            canvas = np.vstack([top, plot_img])
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ds = HDF5VLADataset(
        dataset_path=args.dataset_path,
        action_mode=args.action_mode,
        action_target=args.action_target,
    )
    active_indices = get_active_indices(args.action_mode)

    total = len(ds.file_paths)
    if total == 0:
        raise RuntimeError(f"No hdf5 files found under {args.dataset_path}")

    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, total)
    pick_ids = rng.choice(total, size=n, replace=False)

    print(
        f"Found {total} episodes, sampling {n} with mode={args.action_mode}, "
        f"target={args.action_target}"
    )
    for rank, ep_id in enumerate(pick_ids):
        file_path = ds.file_paths[int(ep_id)]
        try:
            frames, actions, cam_key = load_episode_frames_and_actions(ds, file_path, args.max_steps)
        except Exception as exc:
            print(f"[{rank + 1}/{n}] skip ep#{ep_id}: {exc}")
            continue

        out_name = (
            f"sample_{rank:02d}_ep{int(ep_id):05d}_{args.action_mode}_{args.action_target}.mp4"
        )
        out_path = os.path.join(args.output_dir, out_name)
        write_visualization_video(
            out_path=out_path,
            frames=frames,
            actions=actions,
            active_indices=active_indices,
            action_mode=args.action_mode,
            action_target=args.action_target,
            fps=args.fps,
            top_height=args.top_height,
            plot_height=args.plot_height,
        )
        print(
            f"[{rank + 1}/{n}] saved {out_path} | cam={cam_key} | steps={len(frames)} | valid_dims={len(active_indices)}"
        )


if __name__ == "__main__":
    # import debugpy; debugpy.listen(5678); print("Waiting for debugger attach..."); debugpy.wait_for_client()
    main()
