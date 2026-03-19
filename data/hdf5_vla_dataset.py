import os
import fnmatch

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self, dataset_path=None, action_mode="eef_pose", action_target="delta") -> None:
        # random-rack2: each HDF5 file is one episode
        HDF5_DIR = dataset_path if dataset_path is not None else "/media/damoxing/datasets/myendless/random-rack2"
        self.DATASET_NAME = "astribot"
        
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
        self.file_paths.sort()
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        self.action_mode = action_mode
        if self.action_mode not in {"joint", "eef_pose"}:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode}. "
                "Choose from {'joint', 'eef_pose'}."
            )
        self.action_target = action_target
        if self.action_target not in {"delta", "absolute"}:
            raise ValueError(
                f"Unsupported action_target={self.action_target}. "
                "Choose from {'delta', 'absolute'}."
            )

        # Slot definitions from configs/state_vec.py.
        self.LEFT_JOINT_SLOT_INDICES = [
            STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)
        ]
        self.RIGHT_JOINT_SLOT_INDICES = [
            STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
        ]
        self.LEFT_EEF_POS_SLOT_INDICES = [
            STATE_VEC_IDX_MAPPING["left_eef_pos_x"],
            STATE_VEC_IDX_MAPPING["left_eef_pos_y"],
            STATE_VEC_IDX_MAPPING["left_eef_pos_z"],
        ]
        self.RIGHT_EEF_POS_SLOT_INDICES = [
            STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
            STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
            STATE_VEC_IDX_MAPPING["right_eef_pos_z"],
        ]
        self.LEFT_EEF_ANGLE_SLOT_INDICES = [
            STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)
        ]
        self.RIGHT_EEF_ANGLE_SLOT_INDICES = [
            STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
        ]
        self.LEFT_GRIPPER_OPEN_SLOT_IDX = STATE_VEC_IDX_MAPPING["left_gripper_open"]
        self.RIGHT_GRIPPER_OPEN_SLOT_IDX = STATE_VEC_IDX_MAPPING["right_gripper_open"]
        self.ASTRIBOT_DATA_JOINT_LEFT_ARM_SLICE = slice(0, 7)
        self.ASTRIBOT_DATA_JOINT_RIGHT_ARM_SLICE = slice(7, 14)

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .hdf5 file found under {HDF5_DIR}")
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        episode_lens = np.array(episode_lens, dtype=np.float64)
        if episode_lens.sum() <= 0:
            self.episode_sample_weights = np.ones_like(episode_lens) / len(episode_lens)
        else:
            self.episode_sample_weights = episode_lens / episode_lens.sum()

    def _resolve_arm_slot_indices(self, arm_dim):
        if self.action_mode == "joint":
            if arm_dim > len(self.LEFT_JOINT_SLOT_INDICES):
                raise ValueError(f"Joint arm_dim={arm_dim} exceeds configured slots")
            return (
                self.LEFT_JOINT_SLOT_INDICES[:arm_dim],
                self.RIGHT_JOINT_SLOT_INDICES[:arm_dim],
            )

        # For eef mode, map position first then orientation terms.
        if arm_dim < 3:
            raise ValueError(f"EEF arm_dim={arm_dim} is too small")
        angle_dim = arm_dim - 3
        if angle_dim > len(self.LEFT_EEF_ANGLE_SLOT_INDICES):
            raise ValueError(
                f"EEF arm_dim={arm_dim} requires {angle_dim} angle slots, but only "
                f"{len(self.LEFT_EEF_ANGLE_SLOT_INDICES)} are defined"
            )
        left_indices = self.LEFT_EEF_POS_SLOT_INDICES + self.LEFT_EEF_ANGLE_SLOT_INDICES[:angle_dim]
        right_indices = self.RIGHT_EEF_POS_SLOT_INDICES + self.RIGHT_EEF_ANGLE_SLOT_INDICES[:angle_dim]
        return left_indices, right_indices

    def _fill_bimanual_state(self, left_arm, left_gripper, right_arm, right_gripper):
        """Fill only bimanual arm+gripper dimensions, keep others as zero."""
        if left_arm.shape[-1] != right_arm.shape[-1]:
            raise ValueError(
                f"Left/right arm dims mismatch: {left_arm.shape[-1]} vs {right_arm.shape[-1]}"
            )
        left_indices, right_indices = self._resolve_arm_slot_indices(left_arm.shape[-1])
        uni_vec = np.zeros(left_arm.shape[:-1] + (self.STATE_DIM,), dtype=np.float32)
        uni_vec[..., left_indices] = left_arm
        uni_vec[..., right_indices] = right_arm
        uni_vec[..., self.LEFT_GRIPPER_OPEN_SLOT_IDX] = left_gripper[..., 0]
        uni_vec[..., self.RIGHT_GRIPPER_OPEN_SLOT_IDX] = right_gripper[..., 0]
        return uni_vec

    def _build_state_indicator(self, arm_dim):
        left_indices, right_indices = self._resolve_arm_slot_indices(arm_dim)
        indicator = np.zeros((self.STATE_DIM,), dtype=np.float32)
        indicator[left_indices] = 1.0
        indicator[right_indices] = 1.0
        indicator[self.LEFT_GRIPPER_OPEN_SLOT_IDX] = 1.0
        indicator[self.RIGHT_GRIPPER_OPEN_SLOT_IDX] = 1.0
        return indicator

    @staticmethod
    def _normalize_gripper(state_gripper, cmd_gripper):
        """Normalize gripper widths to [0, 1] as recommended by RDT README."""
        g_min = 0
        g_max = 100
        if g_max - g_min < 1e-8:
            state_norm = np.zeros_like(state_gripper, dtype=np.float32)
            cmd_norm = np.zeros_like(cmd_gripper, dtype=np.float32)
        else:
            state_norm = (state_gripper - g_min) / (g_max - g_min)
            cmd_norm = (cmd_gripper - g_min) / (g_max - g_min)
        state_norm = np.clip(state_norm, 0.0, 1.0)
        cmd_norm = np.clip(cmd_norm, 0.0, 1.0)

        # Astribot source semantics: 1=closed, 0=open after min-max scaling.
        # RDT uses gripper_open where 1=open, 0=closed, so we invert here.
        return (1.0 - state_norm).astype(np.float32), (1.0 - cmd_norm).astype(np.float32)

    @staticmethod
    def _to_delta_action(command_vec, state_vec):
        """Convert command pose sequence to delta action sequence."""
        delta = np.zeros_like(command_vec, dtype=np.float32)
        if command_vec.shape[0] == 0:
            return delta
        # First step: delta from current state to current command.
        delta[0] = command_vec[0] - state_vec[0]
        # Later steps: temporal finite difference in command space.
        if command_vec.shape[0] > 1:
            delta[1:] = command_vec[1:] - command_vec[:-1]
        return delta

    def _build_action_target(self, command_full, state_full):
        if self.action_target == "absolute":
            return command_full.astype(np.float32)
        return self._to_delta_action(command_full, state_full)

    def _load_bimanual_from_hdf5(self, file_obj):
        """Load bimanual trajectories from HDF5 in the requested action mode."""
        if self.action_mode == "joint":
            left_arm = file_obj['joints_dict']['joints_position_state'][:, self.ASTRIBOT_DATA_JOINT_LEFT_ARM_SLICE].astype(np.float32)
            right_arm = file_obj['joints_dict']['joints_position_state'][:, self.ASTRIBOT_DATA_JOINT_RIGHT_ARM_SLICE].astype(np.float32)
            cmd_left_arm = file_obj['joints_dict']['joints_position_command'][:, self.ASTRIBOT_DATA_JOINT_LEFT_ARM_SLICE].astype(np.float32)
            cmd_right_arm = file_obj['joints_dict']['joints_position_command'][:, self.ASTRIBOT_DATA_JOINT_RIGHT_ARM_SLICE].astype(np.float32)
        else:
            left_arm = file_obj['poses_dict']['astribot_arm_left'][:].astype(np.float32)
            right_arm = file_obj['poses_dict']['astribot_arm_right'][:].astype(np.float32)
            cmd_left_arm = file_obj['command_poses_dict']['astribot_arm_left'][:].astype(np.float32)
            cmd_right_arm = file_obj['command_poses_dict']['astribot_arm_right'][:].astype(np.float32)

        left_gripper = file_obj['poses_dict']['astribot_gripper_left'][:].astype(np.float32)
        right_gripper = file_obj['poses_dict']['astribot_gripper_right'][:].astype(np.float32)
        cmd_left_gripper = file_obj['command_poses_dict']['astribot_gripper_left'][:].astype(np.float32)
        cmd_right_gripper = file_obj['command_poses_dict']['astribot_gripper_right'][:].astype(np.float32)

        left_gripper, cmd_left_gripper = self._normalize_gripper(left_gripper, cmd_left_gripper)
        right_gripper, cmd_right_gripper = self._normalize_gripper(right_gripper, cmd_right_gripper)

        num_steps = min(
            left_arm.shape[0], right_arm.shape[0], left_gripper.shape[0], right_gripper.shape[0],
            cmd_left_arm.shape[0], cmd_right_arm.shape[0], cmd_left_gripper.shape[0], cmd_right_gripper.shape[0]
        )
        return {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
            "cmd_left_arm": cmd_left_arm,
            "cmd_right_arm": cmd_right_arm,
            "cmd_left_gripper": cmd_left_gripper,
            "cmd_right_gripper": cmd_right_gripper,
            "num_steps": num_steps,
        }

    @staticmethod
    def _decode_packed_jpeg_frame(rgb_bytes, rgb_sizes, frame_id):
        end = int(np.sum(rgb_sizes[:frame_id + 1]))
        start = end - int(rgb_sizes[frame_id])
        frame = rgb_bytes[start:end]
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)

    def _parse_img(self, file_obj, cam_key, step_id):
        if 'images_dict' not in file_obj or cam_key not in file_obj['images_dict']:
            return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8), np.zeros(self.IMG_HISORY_SIZE, dtype=bool)

        rgb_bytes = file_obj['images_dict'][cam_key]['rgb'][:]
        rgb_sizes = file_obj['images_dict'][cam_key]['rgb_size'][:].astype(np.int64)
        num_steps = len(rgb_sizes)
        step_id = min(step_id, num_steps - 1)

        start = max(step_id - self.IMG_HISORY_SIZE + 1, 0)
        frames = []
        for i in range(start, step_id + 1):
            img = self._decode_packed_jpeg_frame(rgb_bytes, rgb_sizes, i)
            if img is None:
                # If a frame cannot be decoded, use a black placeholder with the same size as the last valid frame.
                if len(frames) > 0:
                    img = np.zeros_like(frames[-1])
                else:
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
            frames.append(img)

        imgs = np.stack(frames, axis=0)
        valid_len = imgs.shape[0]
        if valid_len < self.IMG_HISORY_SIZE:
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE - valid_len, 1, 1, 1)),
                imgs
            ], axis=0)

        img_mask = np.array(
            [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len,
            dtype=bool
        )
        return imgs, img_mask
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            parsed = self._load_bimanual_from_hdf5(f)
            left_arm = parsed["left_arm"]
            right_arm = parsed["right_arm"]
            left_gripper = parsed["left_gripper"]
            right_gripper = parsed["right_gripper"]
            cmd_left_arm = parsed["cmd_left_arm"]
            cmd_right_arm = parsed["cmd_right_arm"]
            cmd_left_gripper = parsed["cmd_left_gripper"]
            cmd_right_gripper = parsed["cmd_right_gripper"]
            num_steps = parsed["num_steps"]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None

            # Skip initial static segment using bimanual arm state.
            EPS = 1e-3
            arm_state = np.concatenate([left_arm[:num_steps], right_arm[:num_steps]], axis=-1)
            arm_delta = np.abs(arm_state - arm_state[0:1])
            indices = np.where(np.any(arm_delta > EPS, axis=1))[0]
            first_idx = int(indices[0]) if len(indices) > 0 else 0
            
            # We randomly sample a timestep
            min_step = max(first_idx - 1, 0)
            step_id = np.random.randint(min_step, num_steps)
            
            # random-rack2 does not provide per-episode language, use a stable instruction.
            instruction = "Use right arm to perform rack manipulation while keeping left-arm joints fixed."
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            # Parse the state and action
            state = self._fill_bimanual_state(
                left_arm[step_id:step_id+1],
                left_gripper[step_id:step_id+1],
                right_arm[step_id:step_id+1],
                right_gripper[step_id:step_id+1],
            )
            full_state = self._fill_bimanual_state(
                left_arm[:num_steps], left_gripper[:num_steps], right_arm[:num_steps], right_gripper[:num_steps]
            )
            state_std = np.std(full_state, axis=0)
            state_mean = np.mean(full_state, axis=0)
            state_norm = np.sqrt(np.mean(full_state**2, axis=0))

            command_full = self._fill_bimanual_state(
                cmd_left_arm[:num_steps], cmd_left_gripper[:num_steps], cmd_right_arm[:num_steps], cmd_right_gripper[:num_steps]
            )
            action_full = self._build_action_target(command_full, full_state)
            action_mean = np.mean(action_full, axis=0)
            action_std = np.std(action_full, axis=0)
            actions = action_full[step_id:step_id+self.CHUNK_SIZE]
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)

            state_indicator = self._build_state_indicator(left_arm.shape[-1])

            # Parse images from packed JPEG streams.
            cam_high, cam_high_mask = self._parse_img(f, 'head', step_id)
            cam_left_wrist, cam_left_wrist_mask = self._parse_img(f, 'left', step_id)
            cam_right_wrist, cam_right_wrist_mask = self._parse_img(f, 'right', step_id)
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "action_mean": action_mean,
                "action_std": action_std,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            parsed = self._load_bimanual_from_hdf5(f)
            left_arm = parsed["left_arm"]
            right_arm = parsed["right_arm"]
            left_gripper = parsed["left_gripper"]
            right_gripper = parsed["right_gripper"]
            cmd_left_arm = parsed["cmd_left_arm"]
            cmd_right_arm = parsed["cmd_right_arm"]
            cmd_left_gripper = parsed["cmd_left_gripper"]
            cmd_right_gripper = parsed["cmd_right_gripper"]
            num_steps = parsed["num_steps"]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None

            EPS = 1e-3
            arm_state = np.concatenate([left_arm[:num_steps], right_arm[:num_steps]], axis=-1)
            arm_delta = np.abs(arm_state - arm_state[0:1])
            indices = np.where(np.any(arm_delta > EPS, axis=1))[0]
            first_idx = int(indices[0]) if len(indices) > 0 else 0

            # Parse the state and action
            start = max(first_idx - 1, 0)
            state = self._fill_bimanual_state(
                left_arm[start:num_steps],
                left_gripper[start:num_steps],
                right_arm[start:num_steps],
                right_gripper[start:num_steps],
            )
            command = self._fill_bimanual_state(
                cmd_left_arm[start:num_steps],
                cmd_left_gripper[start:num_steps],
                cmd_right_arm[start:num_steps],
                cmd_right_gripper[start:num_steps],
            )
            action = self._build_action_target(command, state)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
