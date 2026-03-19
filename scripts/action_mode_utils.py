import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


def get_active_indices(action_mode: str):
    """Return active unified-action indices for each action mode.

    Note: In the current astribot HDF5 pipeline, both modes are stored in the
    same unified slots (arm_joint_0..6 for both arms + two grippers). The
    semantic meaning differs by mode.
    """
    if action_mode not in {"delta_joint", "delta_eef_pose"}:
        raise ValueError(f"Unsupported action_mode={action_mode}")

    if action_mode == "delta_joint":
        left_arm = [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)]
        right_arm = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]
    else:
        left_arm = [
            STATE_VEC_IDX_MAPPING["left_eef_pos_x"],
            STATE_VEC_IDX_MAPPING["left_eef_pos_y"],
            STATE_VEC_IDX_MAPPING["left_eef_pos_z"],
            STATE_VEC_IDX_MAPPING["left_eef_angle_0"],
            STATE_VEC_IDX_MAPPING["left_eef_angle_1"],
            STATE_VEC_IDX_MAPPING["left_eef_angle_2"],
            STATE_VEC_IDX_MAPPING["left_eef_angle_3"],
        ]
        right_arm = [
            STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
            STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
            STATE_VEC_IDX_MAPPING["right_eef_pos_z"],
            STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
            STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
            STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
            STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
        ]
    left_gripper = STATE_VEC_IDX_MAPPING["left_gripper_open"]
    right_gripper = STATE_VEC_IDX_MAPPING["right_gripper_open"]

    return left_arm + right_arm + [left_gripper, right_gripper]


def build_state_mask(state_dim: int, active_indices):
    mask = np.zeros((state_dim,), dtype=np.float32)
    mask[np.array(active_indices, dtype=np.int64)] = 1.0
    return mask
