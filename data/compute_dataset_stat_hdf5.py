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
    episode_cnt = 0
    
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
        # We need full trajectory to compute delta actions, so we use internal parse logic
        # Access the private file path list if possible, or assume get_item order
        file_path = vla_dataset.file_paths[i]
        
        # We parse the file manually to get full sequence (get_item returns random chunks)
        # Re-using the logic from HDF5VLADataset.parse_hdf5_file but without random sampling
        with h5py.File(file_path, 'r') as f:
            # --- Load Raw Data ---
            left_arm = f['poses_dict']['astribot_arm_left'][:].astype(np.float32)
            right_arm = f['poses_dict']['astribot_arm_right'][:].astype(np.float32)
            left_gripper = f['poses_dict']['astribot_gripper_left'][:].astype(np.float32)
            right_gripper = f['poses_dict']['astribot_gripper_right'][:].astype(np.float32)

            cmd_left_arm = f['command_poses_dict']['astribot_arm_left'][:].astype(np.float32)
            cmd_right_arm = f['command_poses_dict']['astribot_arm_right'][:].astype(np.float32)
            cmd_left_gripper = f['command_poses_dict']['astribot_gripper_left'][:].astype(np.float32)
            cmd_right_gripper = f['command_poses_dict']['astribot_gripper_right'][:].astype(np.float32)

            # Normalize Gripper (Dataset specific logic re-used)
            left_gripper, cmd_left_gripper = vla_dataset._normalize_gripper(left_gripper, cmd_left_gripper)
            right_gripper, cmd_right_gripper = vla_dataset._normalize_gripper(right_gripper, cmd_right_gripper)

            # Get length
            num_steps = min(
                left_arm.shape[0], right_arm.shape[0], left_gripper.shape[0], right_gripper.shape[0],
                cmd_left_arm.shape[0], cmd_right_arm.shape[0], cmd_left_gripper.shape[0], cmd_right_gripper.shape[0]
            )
            
            # --- Construct Full State ---
            full_state = vla_dataset._fill_bimanual_state(
                left_arm[:num_steps], left_gripper[:num_steps], right_arm[:num_steps], right_gripper[:num_steps]
            )
            
            # --- Construct Full Action (Delta) ---
            command_full = vla_dataset._fill_bimanual_state(
                cmd_left_arm[:num_steps], cmd_left_gripper[:num_steps], cmd_right_arm[:num_steps], cmd_right_gripper[:num_steps]
            )
            full_actions = vla_dataset._to_delta_action(command_full, full_state)

            # --- Construct Indicator ---
            indicator = np.zeros((vla_dataset.STATE_DIM,), dtype=np.float32)
            indicator[vla_dataset.LEFT_ARM_INDICES] = 1.0
            indicator[vla_dataset.RIGHT_ARM_INDICES] = 1.0
            indicator[vla_dataset.LEFT_GRIPPER_IDX] = 1.0
            indicator[vla_dataset.RIGHT_GRIPPER_IDX] = 1.0
            
            # Broadcast indicator to (T, D)
            valid_mask = (indicator > 0.5)[None, :]
            
            # --- Aggregate Stats (State) ---
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

            # State Stats
            state_sum += np.sum(full_state * valid_mask, axis=0)
            state_sum_sq += np.sum((full_state * valid_mask)**2, axis=0)
            state_cnt += np.sum(np.broadcast_to(valid_mask, full_state.shape), axis=0)
            
            curr_s_max = np.max(np.where(valid_mask, full_state, -np.inf), axis=0)
            curr_s_min = np.min(np.where(valid_mask, full_state, np.inf), axis=0)
            state_max = np.maximum(state_max, curr_s_max)
            state_min = np.minimum(state_min, curr_s_min)
            
            # Action Stats
            action_sum += np.sum(full_actions * valid_mask, axis=0)
            action_sum_sq += np.sum((full_actions * valid_mask)**2, axis=0)
            action_cnt += np.sum(np.broadcast_to(valid_mask, full_actions.shape), axis=0)
            
            curr_a_max = np.max(np.where(valid_mask, full_actions, -np.inf), axis=0)
            curr_a_min = np.min(np.where(valid_mask, full_actions, np.inf), axis=0)
            action_max = np.maximum(action_max, curr_a_max)
            action_min = np.minimum(action_min, curr_a_min)

    # --- Compute Final Stats ---
    state_cnt = np.maximum(state_cnt, 1.0)
    action_cnt = np.maximum(action_cnt, 1.0)
    
    s_mean = state_sum / state_cnt
    s_var = (state_sum_sq / state_cnt) - (s_mean ** 2)
    s_std = np.sqrt(np.maximum(s_var, 0))
    
    a_mean = action_sum / action_cnt
    a_var = (action_sum_sq / action_cnt) - (a_mean ** 2)
    a_std = np.sqrt(np.maximum(a_var, 0))
    
    # Handle never-valid dimensions
    invalid_dims = (state_cnt <= 1.0)
    s_mean[invalid_dims] = 0.0
    s_std[invalid_dims] = 1.0
    
    # For actions, we want to normalize them too? 
    # Usually for delta actions, mean is close to 0. 
    # Can force mean=0 if desired, but let's trust data.
    a_mean[invalid_dims] = 0.0
    a_std[invalid_dims] = 1.0

    result = {
        "dataset_name": vla_dataset.get_dataset_name(),
        "state_mean": s_mean.tolist(),
        "state_std": s_std.tolist(),
        "state_min": clean_stat(state_max),
        "state_max": clean_stat(state_min),
        "action_mean": a_mean.tolist(),
        "action_std": a_std.tolist(),
        "action_min": clean_stat(action_max),
        "action_max": clean_stat(action_min),
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, 
                        default="configs/dataset_stat.json", 
                        help="JSON file path to save the dataset statistics.")
    parser.add_argument('--skip_exist', action='store_true', 
                        help="Whether to skip the existing dataset statistics.")
    args = parser.parse_args()
    
    vla_dataset = HDF5VLADataset()
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
