"""
This file will compute the min, max, mean, and standard deviation of each datasets 
in `pretrain_datasets.json` or `pretrain_datasets.json`.
"""

import json
import argparse

import h5py
import numpy as np
from tqdm import tqdm

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.hdf5_vla_dataset import HDF5VLADataset


def process_hdf5_dataset(vla_dataset):

    # Accumulators for State
    state_sum = None
    state_sum_sq = None
    state_cnt = None
    state_max = None
    state_min = None

    # Accumulators for Action
    action_sum = None
    action_sum_sq = None
    action_cnt = None
    action_max = None
    action_min = None

    # Helper to clean infinite values
    def clean_stat(val, fill_val=0.0):
        if val is None: return None
        val = np.array(val)
        val[np.isinf(val)] = fill_val
        return val.tolist()

    print(f"Computing stats for {len(vla_dataset)} episodes...")
    for i in tqdm(range(len(vla_dataset))):
        file_path = vla_dataset.file_paths[i]

        with h5py.File(file_path, 'r') as f:
            parsed = vla_dataset._load_bimanual_from_hdf5(f)
            left_arm = parsed["left_arm"]
            right_arm = parsed["right_arm"]
            left_gripper = parsed["left_gripper"]
            right_gripper = parsed["right_gripper"]
            cmd_left_arm = parsed["cmd_left_arm"]
            cmd_right_arm = parsed["cmd_right_arm"]
            cmd_left_gripper = parsed["cmd_left_gripper"]
            cmd_right_gripper = parsed["cmd_right_gripper"]
            num_steps = parsed["num_steps"]

            full_state = vla_dataset._fill_bimanual_state(
                left_arm[:num_steps], left_gripper[:num_steps], right_arm[:num_steps], right_gripper[:num_steps]
            )

            command_full = vla_dataset._fill_bimanual_state(
                cmd_left_arm[:num_steps], cmd_left_gripper[:num_steps], cmd_right_arm[:num_steps], cmd_right_gripper[:num_steps]
            )
            full_actions = vla_dataset._build_action_target(command_full, full_state)

            indicator = vla_dataset._build_state_indicator(left_arm.shape[-1])
            valid_mask = (indicator > 0.5)[None, :]

            if state_sum is None:
                dim = full_state.shape[1]
                state_sum = np.zeros(dim)
                state_sum_sq = np.zeros(dim)
                state_cnt = np.zeros(dim)
                state_max = np.full(dim, -np.inf)
                state_min = np.full(dim, np.inf)

                action_sum = np.zeros(dim)
                action_sum_sq = np.zeros(dim)
                action_cnt = np.zeros(dim)
                action_max = np.full(dim, -np.inf)
                action_min = np.full(dim, np.inf)

            state_sum += np.sum(full_state * valid_mask, axis=0)
            state_sum_sq += np.sum((full_state * valid_mask)**2, axis=0)
            state_cnt += np.sum(np.broadcast_to(valid_mask, full_state.shape), axis=0)

            curr_s_max = np.max(np.where(valid_mask, full_state, -np.inf), axis=0)
            curr_s_min = np.min(np.where(valid_mask, full_state, np.inf), axis=0)
            state_max = np.maximum(state_max, curr_s_max)
            state_min = np.minimum(state_min, curr_s_min)

            action_sum += np.sum(full_actions * valid_mask, axis=0)
            action_sum_sq += np.sum((full_actions * valid_mask)**2, axis=0)
            action_cnt += np.sum(np.broadcast_to(valid_mask, full_actions.shape), axis=0)

            curr_a_max = np.max(np.where(valid_mask, full_actions, -np.inf), axis=0)
            curr_a_min = np.min(np.where(valid_mask, full_actions, np.inf), axis=0)
            action_max = np.maximum(action_max, curr_a_max)
            action_min = np.minimum(action_min, curr_a_min)

    state_cnt = np.maximum(state_cnt, 1.0)
    action_cnt = np.maximum(action_cnt, 1.0)

    s_mean = state_sum / state_cnt
    s_var = (state_sum_sq / state_cnt) - (s_mean ** 2)
    s_std = np.sqrt(np.maximum(s_var, 0))

    a_mean = action_sum / action_cnt
    a_var = (action_sum_sq / action_cnt) - (a_mean ** 2)
    a_std = np.sqrt(np.maximum(a_var, 0))

    invalid_dims = (state_cnt <= 1.0)
    s_mean[invalid_dims] = 0.0
    s_std[invalid_dims] = 1.0

    a_mean[invalid_dims] = 0.0
    a_std[invalid_dims] = 1.0

    result = {
        "dataset_name": vla_dataset.get_dataset_name(),
        "state_mean": s_mean.tolist(),
        "state_std": s_std.tolist(),
        "state_min": clean_stat(state_min),
        "state_max": clean_stat(state_max),
        "action_mean": a_mean.tolist(),
        "action_std": a_std.tolist(),
        "action_min": clean_stat(action_min),
        "action_max": clean_stat(action_max),
    }

    mode = vla_dataset.action_mode
    target = vla_dataset.action_target
    result[f"state_mean_{mode}"] = s_mean.tolist()
    result[f"state_std_{mode}"] = s_std.tolist()
    result[f"action_mean_{target}_{mode}"] = a_mean.tolist()
    result[f"action_std_{target}_{mode}"] = a_std.tolist()

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, 
                        default="configs/dataset_stat.json", 
                        help="JSON file path to save the dataset statistics.")
    parser.add_argument('--dataset_path', type=str, default="/media/damoxing/datasets/myendless/random-rack2",
                        help="Path to the dataset to compute statistics for.")
    parser.add_argument('--skip_exist', action='store_true', 
                        help="Whether to skip the existing dataset statistics.")
    parser.add_argument('--action_mode', type=str, default="eef_pose", choices=["joint", "eef_pose"],
                        help="Action mode to determine which state/action dimensions to compute stats for.")
    parser.add_argument('--action_target', type=str, default="delta", choices=["delta", "absolute"],
                        help="Action target type to determine how to compute actions for statistics.")
    args = parser.parse_args()

    vla_dataset = HDF5VLADataset(dataset_path=args.dataset_path, action_mode=args.action_mode, action_target=args.action_target)
    dataset_name = vla_dataset.get_dataset_name()

    try:
        with open(args.save_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    if args.skip_exist and dataset_name in results:
        print(f"Skipping existed {dataset_name} dataset statistics")
    else:
        print(f"Processing {dataset_name} dataset")
        result = process_hdf5_dataset(vla_dataset)
        results[result["dataset_name"]] = result
        with open(args.save_path, 'w') as f:
            json.dump(results, f, indent=4)
    print("All datasets have been processed.")
