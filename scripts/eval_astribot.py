#!/usr/bin/env python
# coding=utf-8

"""Online eval for Astribot through RDT REST server.

流程：
1) 连接 Astribot SDK；
2) 获取当前机器人状态 + 三路相机观测（head/left_wrist/right_wrist）；
3) 上传 state + images 到 /infer；
4) 将返回动作转换为 Astribot 命令并执行。
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import cv2
import h5py
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "astribot_sdk"))

from configs.state_vec import STATE_VEC_IDX_MAPPING, STATE_VEC_LEN
from core.astribot_api.astribot_client import Astribot
from scripts.action_mode_utils import get_active_indices


def encode_image_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def post_json(url: str, payload: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
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


def gripper_joint_to_open(g_joint: float) -> float:
    # SDK: 0=open, 100=close; 模型: 1=open, 0=close
    return float(np.clip(1.0 - (g_joint / 100.0), 0.0, 1.0))


def gripper_open_to_joint(g_open: float) -> float:
    return float(np.clip((1.0 - g_open) * 100.0, 0.0, 100.0))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def build_state_vec(
    left_joint: List[float],
    right_joint: List[float],
    left_gripper_joint: float,
    right_gripper_joint: float,
    left_pose_xyzquat: List[float],
    right_pose_xyzquat: List[float],
) -> np.ndarray:
    """构造 128 维 unified state。"""
    state = np.zeros((STATE_VEC_LEN,), dtype=np.float32)

    for i in range(min(7, len(left_joint))):
        state[STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]] = left_joint[i]
    for i in range(min(7, len(right_joint))):
        state[STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]] = right_joint[i]

    state[STATE_VEC_IDX_MAPPING["left_gripper_open"]] = gripper_joint_to_open(left_gripper_joint)
    state[STATE_VEC_IDX_MAPPING["right_gripper_open"]] = gripper_joint_to_open(right_gripper_joint)

    if len(left_pose_xyzquat) >= 7:
        state[STATE_VEC_IDX_MAPPING["left_eef_pos_x"]] = left_pose_xyzquat[0]
        state[STATE_VEC_IDX_MAPPING["left_eef_pos_y"]] = left_pose_xyzquat[1]
        state[STATE_VEC_IDX_MAPPING["left_eef_pos_z"]] = left_pose_xyzquat[2]
        state[STATE_VEC_IDX_MAPPING["left_eef_angle_0"]] = left_pose_xyzquat[3]
        state[STATE_VEC_IDX_MAPPING["left_eef_angle_1"]] = left_pose_xyzquat[4]
        state[STATE_VEC_IDX_MAPPING["left_eef_angle_2"]] = left_pose_xyzquat[5]
        state[STATE_VEC_IDX_MAPPING["left_eef_angle_3"]] = left_pose_xyzquat[6]

    if len(right_pose_xyzquat) >= 7:
        state[STATE_VEC_IDX_MAPPING["right_eef_pos_x"]] = right_pose_xyzquat[0]
        state[STATE_VEC_IDX_MAPPING["right_eef_pos_y"]] = right_pose_xyzquat[1]
        state[STATE_VEC_IDX_MAPPING["right_eef_pos_z"]] = right_pose_xyzquat[2]
        state[STATE_VEC_IDX_MAPPING["right_eef_angle_0"]] = right_pose_xyzquat[3]
        state[STATE_VEC_IDX_MAPPING["right_eef_angle_1"]] = right_pose_xyzquat[4]
        state[STATE_VEC_IDX_MAPPING["right_eef_angle_2"]] = right_pose_xyzquat[5]
        state[STATE_VEC_IDX_MAPPING["right_eef_angle_3"]] = right_pose_xyzquat[6]

    return state


class CameraHistory:
    def __init__(self, history_size: int):
        self.history_size = int(history_size)
        self._lock = threading.Lock()
        self._buffers: Dict[str, Deque[np.ndarray]] = {
            "cam_high": deque(maxlen=self.history_size),
            "cam_right_wrist": deque(maxlen=self.history_size),
            "cam_left_wrist": deque(maxlen=self.history_size),
        }

    def put(self, camera_key: str, img_bgr: np.ndarray) -> None:
        with self._lock:
            self._buffers[camera_key].append(img_bgr.copy())

    def ready(self) -> bool:
        with self._lock:
            return all(len(v) > 0 for v in self._buffers.values())

    def get_history(self) -> Dict[str, List[np.ndarray]]:
        out: Dict[str, List[np.ndarray]] = {}
        with self._lock:
            for key, dq in self._buffers.items():
                if len(dq) == 0:
                    raise RuntimeError(f"No frame yet for {key}")
                seq = list(dq)
                if len(seq) < self.history_size:
                    pad = [seq[0].copy() for _ in range(self.history_size - len(seq))]
                    seq = pad + seq
                out[key] = seq
        return out


def make_image_callback(history: CameraHistory):
    camera_map = {
        "head_rgbd": "cam_high",
        "right_wrist_rgbd": "cam_right_wrist",
        "left_wrist_rgbd": "cam_left_wrist",
    }

    def _cb(topic_name, msg, width, height, array):
        try:
            # topic: /astribot_camera/<camera_name>/...
            parts = topic_name.split("/")
            camera_name = parts[2] if len(parts) > 2 else ""
            if msg.format.lower() != "jpeg":
                return
            if camera_name not in camera_map:
                return
            if not isinstance(array, np.ndarray) or array.ndim != 3:
                return
            history.put(camera_map[camera_name], array)
        except Exception:
            return

    return _cb


def get_robot_state(bot: Astribot) -> Tuple[np.ndarray, Dict[str, Any]]:
    # joints
    joint_names = [
        bot.arm_left_name,
        bot.arm_right_name,
        bot.effector_left_name,
        bot.effector_right_name,
    ]
    joints = bot.get_current_joints_position(names=joint_names)
    left_joint = joints[0]
    right_joint = joints[1]
    left_g_joint = float(joints[2][0])
    right_g_joint = float(joints[3][0])

    # cartesian pose [x,y,z,qx,qy,qz,qw]
    cart_names = [bot.arm_left_name, bot.arm_right_name]
    cart = bot.get_current_cartesian_pose(names=cart_names, frame=bot.chassis_frame_name)
    left_pose = cart[0]
    right_pose = cart[1]

    state_vec = build_state_vec(
        left_joint=left_joint,
        right_joint=right_joint,
        left_gripper_joint=left_g_joint,
        right_gripper_joint=right_g_joint,
        left_pose_xyzquat=left_pose,
        right_pose_xyzquat=right_pose,
    )

    raw = {
        "left_joint": np.asarray(left_joint, dtype=np.float32),
        "right_joint": np.asarray(right_joint, dtype=np.float32),
        "left_gripper_open": float(gripper_joint_to_open(left_g_joint)),
        "right_gripper_open": float(gripper_joint_to_open(right_g_joint)),
        "left_pose": np.asarray(left_pose, dtype=np.float32),
        "right_pose": np.asarray(right_pose, dtype=np.float32),
    }
    return state_vec, raw


def convert_action_to_robot_command(
    pred_action: np.ndarray,
    action_mode: str,
    action_target: str,
    active_indices: List[int],
    current_raw: Dict[str, Any],
    bot: Astribot,
) -> Dict[str, Any]:
    if pred_action.ndim == 2:
        pred0 = pred_action[0]
    else:
        pred0 = pred_action

    idx = np.asarray(active_indices, dtype=np.int64)
    act = pred0[idx].astype(np.float32)

    if action_mode == "joint":
        left = act[:7]
        right = act[7:14]
        g_left = float(act[14])
        g_right = float(act[15])

        if action_target == "delta":
            left = current_raw["left_joint"] + left
            right = current_raw["right_joint"] + right
            g_left = float(current_raw["left_gripper_open"] + g_left)
            g_right = float(current_raw["right_gripper_open"] + g_right)

        g_left_joint = gripper_open_to_joint(g_left)
        g_right_joint = gripper_open_to_joint(g_right)

        return {
            "cmd_type": "joint",
            "names": [
                bot.arm_left_name,
                bot.arm_right_name,
                bot.effector_left_name,
                bot.effector_right_name,
            ],
            "commands": [
                left.astype(float).tolist(),
                right.astype(float).tolist(),
                [g_left_joint],
                [g_right_joint],
            ],
            "meta": {
                "action_mode": action_mode,
                "action_target": action_target,
                "timestamp": time.time(),
            },
        }

    # eef_pose: [left(7), right(7), gL, gR]
    left_pose = act[:7]
    right_pose = act[7:14]
    g_left = float(act[14])
    g_right = float(act[15])

    if action_target == "delta":
        left_pose = current_raw["left_pose"] + left_pose
        right_pose = current_raw["right_pose"] + right_pose
        g_left = float(current_raw["left_gripper_open"] + g_left)
        g_right = float(current_raw["right_gripper_open"] + g_right)

    left_pose = left_pose.astype(np.float32)
    right_pose = right_pose.astype(np.float32)
    left_pose[3:7] = quat_normalize(left_pose[3:7])
    right_pose[3:7] = quat_normalize(right_pose[3:7])

    g_left_joint = gripper_open_to_joint(g_left)
    g_right_joint = gripper_open_to_joint(g_right)

    return {
        "cmd_type": "different_type",
        "names": [
            bot.arm_left_name,
            bot.arm_right_name,
            bot.effector_left_name,
            bot.effector_right_name,
        ],
        "types": ["cartesian", "cartesian", "joints", "joints"],
        "commands": [
            left_pose.astype(float).tolist(),
            right_pose.astype(float).tolist(),
            [g_left_joint],
            [g_right_joint],
        ],
        "meta": {
            "action_mode": action_mode,
            "action_target": action_target,
            "timestamp": time.time(),
        },
    }


def execute_robot_command(bot: Astribot, cmd: Dict[str, Any], control_way: str = "filter") -> None:
    if cmd["cmd_type"] == "joint":
        bot.set_joints_position(cmd["names"], cmd["commands"], control_way=control_way, use_wbc=False)
        return
    if cmd["cmd_type"] == "different_type":
        bot.set_different_type_command(
            cmd["names"],
            cmd["types"],
            cmd["commands"],
            control_way=control_way,
            use_wbc=False,
        )
        return
    raise ValueError(f"Unknown cmd_type={cmd.get('cmd_type')}")


def load_init_joint_target_from_hdf5(hdf5_path: str) -> Tuple[List[str], List[List[float]]]:
    """从 HDF5 中读取 t=0 的关节状态，并转换成 Astribot move_joints_position 输入。"""
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        if "joints_dict" not in f:
            raise KeyError("Missing group `joints_dict` in HDF5")

        source_key = None
        for key in ["joints_position_state", "joints_position_command"]:
            if key in f["joints_dict"]:
                source_key = key
                break
        if source_key is None:
            raise KeyError("`joints_dict` has neither `joints_position_state` nor `joints_position_command`")

        arr = np.asarray(f["joints_dict"][source_key][0], dtype=np.float32).reshape(-1)

    n = int(arr.shape[0])

    # 常见 Astribot 录制格式（参考 SDK 示例 208-traj_replay）
    # [chassis(3), torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2)] => 25
    if n >= 25:
        names = [
            "astribot_torso",
            "astribot_arm_left",
            "astribot_gripper_left",
            "astribot_arm_right",
            "astribot_gripper_right",
            "astribot_head",
        ]
        cmds = [
            arr[3:7].astype(float).tolist(),
            arr[7:14].astype(float).tolist(),
            arr[14:15].astype(float).tolist(),
            arr[15:22].astype(float).tolist(),
            arr[22:23].astype(float).tolist(),
            arr[23:25].astype(float).tolist(),
        ]
        return names, cmds

    # 无 head 的变体: 23 = 3+4+7+1+7+1
    if n >= 23:
        names = [
            "astribot_torso",
            "astribot_arm_left",
            "astribot_gripper_left",
            "astribot_arm_right",
            "astribot_gripper_right",
        ]
        cmds = [
            arr[3:7].astype(float).tolist(),
            arr[7:14].astype(float).tolist(),
            arr[14:15].astype(float).tolist(),
            arr[15:22].astype(float).tolist(),
            arr[22:23].astype(float).tolist(),
        ]
        return names, cmds

    # 训练管线常见精简格式: 14 = left_arm(7) + right_arm(7)
    if n == 14:
        names = ["astribot_arm_left", "astribot_arm_right"]
        cmds = [arr[0:7].astype(float).tolist(), arr[7:14].astype(float).tolist()]
        return names, cmds

    # 精简格式带 gripper: 16 = left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
    if n == 16:
        names = [
            "astribot_arm_left",
            "astribot_arm_right",
            "astribot_gripper_left",
            "astribot_gripper_right",
        ]
        cmds = [
            arr[0:7].astype(float).tolist(),
            arr[7:14].astype(float).tolist(),
            arr[14:15].astype(float).tolist(),
            arr[15:16].astype(float).tolist(),
        ]
        return names, cmds

    raise ValueError(f"Unsupported joints vector length at t=0: {n}")


def parse_args():
    parser = argparse.ArgumentParser("Online eval astribot via RDT REST server")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--action_mode", type=str, default="eef_pose", choices=["joint", "eef_pose"])
    parser.add_argument("--action_target", type=str, default="delta", choices=["delta", "absolute"])
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--control_hz", type=float, default=8.0)
    parser.add_argument("--ctrl_freq", type=int, default=25, help="发送给服务端的 ctrl_freq")
    parser.add_argument("--img_history_size", type=int, default=2)
    parser.add_argument("--timeout_sec", type=float, default=30.0)
    parser.add_argument("--warmup_sec", type=float, default=8.0)
    parser.add_argument("--high_control_rights", action="store_true")
    parser.add_argument("--activate_camera", action="store_true")
    parser.add_argument("--init_hdf5", type=str, default="", help="启动时读取该 hdf5 的 t=0 位姿并先移动到该位姿")
    parser.add_argument("--init_move_duration", type=float, default=3.0, help="启动初始化位姿移动时长（秒）")
    parser.add_argument("--execute", action="store_true", help="执行发送给 Astribot 的控制命令")
    parser.add_argument("--control_way", type=str, default="filter", choices=["filter", "direct"])
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    import debugpy; debugpy.listen(5678); print("Waiting for debugger attach..."); debugpy.wait_for_client()
    active_indices = get_active_indices(args.action_mode)

    print("=== Astribot Online Eval ===")
    print(f"server_url={args.server_url}")
    print(f"action_mode={args.action_mode}")
    print(f"action_target={args.action_target}")
    print(f"execute={args.execute}")

    bot = Astribot(freq=250.0, high_control_rights=args.high_control_rights)

    if args.init_hdf5:
        print(f"Loading init pose from hdf5: {args.init_hdf5}")
        init_names, init_cmds = load_init_joint_target_from_hdf5(args.init_hdf5)
        print(f"Moving robot to t=0 pose (duration={args.init_move_duration}s)...")
        bot.move_joints_position(init_names, init_cmds, duration=float(args.init_move_duration), use_wbc=False)
        time.sleep(0.1)

    if args.activate_camera:
        print("Activating cameras...")
        bot.activate_camera()

    cam_history = CameraHistory(history_size=args.img_history_size)
    image_cb = make_image_callback(cam_history)

    subs = []
    subs.append(bot.register_image_callback("head_rgbd", "color", image_cb, need_decode=True))
    subs.append(bot.register_image_callback("right_wrist_rgbd", "color", image_cb, need_decode=True))
    subs.append(bot.register_image_callback("left_wrist_rgbd", "color", image_cb, need_decode=True))
    _ = subs

    t0 = time.time()
    while not cam_history.ready() and (time.time() - t0) < args.warmup_sec:
        time.sleep(0.05)
    if not cam_history.ready():
        raise RuntimeError("Camera warmup timeout: failed to receive all required camera frames")

    results: List[Dict[str, Any]] = []
    dt = 1.0 / max(args.control_hz, 1e-6)

    for step in range(args.num_steps):
        loop_t0 = time.time()

        state_vec, current_raw = get_robot_state(bot)
        images = cam_history.get_history()

        payload = {
            "instruction": args.instruction,
            "state": state_vec.astype(np.float32).tolist(),
            "images": {
                "cam_high": [encode_image_b64(img) for img in images["cam_high"]],
                "cam_right_wrist": [encode_image_b64(img) for img in images["cam_right_wrist"]],
                "cam_left_wrist": [encode_image_b64(img) for img in images["cam_left_wrist"]],
            },
            "action_target": args.action_target,
            "ctrl_freq": int(args.ctrl_freq),
        }

        infer_t0 = time.time()
        resp = post_json(f"{args.server_url}/infer", payload, timeout_sec=args.timeout_sec)
        infer_latency_ms = float((time.time() - infer_t0) * 1000.0)

        if not resp.get("ok", False):
            raise RuntimeError(f"Server infer failed: {resp}")

        pred = np.asarray(resp["action"], dtype=np.float32)
        cmd = convert_action_to_robot_command(
            pred_action=pred,
            action_mode=args.action_mode,
            action_target=args.action_target,
            active_indices=active_indices,
            current_raw=current_raw,
            bot=bot,
        )

        if args.execute:
            execute_robot_command(bot, cmd, control_way=args.control_way)

        loop_ms = float((time.time() - loop_t0) * 1000.0)
        record = {
            "step": step,
            "latency_ms": float(resp.get("latency_ms", infer_latency_ms)),
            "loop_ms": loop_ms,
            "cmd": cmd,
        }
        results.append(record)

        print(
            f"step={step:04d} infer_latency_ms={record['latency_ms']:.2f} "
            f"loop_ms={record['loop_ms']:.2f} execute={args.execute}"
        )

        sleep_t = dt - (time.time() - loop_t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
