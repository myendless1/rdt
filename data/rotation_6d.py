import numpy as np


def _normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    mag = np.maximum(mag, eps)
    return v / mag


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.cross(u, v, axis=-1)


def quaternion_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion in xyzw format to rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {q.shape}")

    q = _normalize_vector(q)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - zw)
    m02 = 2.0 * (xz + yw)
    m10 = 2.0 * (xy + zw)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - xw)
    m20 = 2.0 * (xz - yw)
    m21 = 2.0 * (yz + xw)
    m22 = 1.0 - 2.0 * (xx + yy)

    mat = np.stack(
        [
            np.stack([m00, m01, m02], axis=-1),
            np.stack([m10, m11, m12], axis=-1),
            np.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    )
    return mat.astype(np.float32)


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion in xyzw format."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matrix shape (..., 3, 3), got {m.shape}")

    m_flat = m.reshape(-1, 3, 3)
    q_flat = np.zeros((m_flat.shape[0], 4), dtype=np.float32)

    m00 = m_flat[:, 0, 0]
    m11 = m_flat[:, 1, 1]
    m22 = m_flat[:, 2, 2]
    trace = m00 + m11 + m22

    mask_t = trace > 0.0
    if np.any(mask_t):
        t = np.sqrt(np.maximum(trace[mask_t] + 1.0, 1e-8)) * 2.0
        q_flat[mask_t, 3] = 0.25 * t
        q_flat[mask_t, 0] = (m_flat[mask_t, 2, 1] - m_flat[mask_t, 1, 2]) / t
        q_flat[mask_t, 1] = (m_flat[mask_t, 0, 2] - m_flat[mask_t, 2, 0]) / t
        q_flat[mask_t, 2] = (m_flat[mask_t, 1, 0] - m_flat[mask_t, 0, 1]) / t

    mask_x = (~mask_t) & (m00 > m11) & (m00 > m22)
    if np.any(mask_x):
        t = np.sqrt(np.maximum(1.0 + m_flat[mask_x, 0, 0] - m_flat[mask_x, 1, 1] - m_flat[mask_x, 2, 2], 1e-8)) * 2.0
        q_flat[mask_x, 3] = (m_flat[mask_x, 2, 1] - m_flat[mask_x, 1, 2]) / t
        q_flat[mask_x, 0] = 0.25 * t
        q_flat[mask_x, 1] = (m_flat[mask_x, 0, 1] + m_flat[mask_x, 1, 0]) / t
        q_flat[mask_x, 2] = (m_flat[mask_x, 0, 2] + m_flat[mask_x, 2, 0]) / t

    mask_y = (~mask_t) & (~mask_x) & (m11 > m22)
    if np.any(mask_y):
        t = np.sqrt(np.maximum(1.0 + m_flat[mask_y, 1, 1] - m_flat[mask_y, 0, 0] - m_flat[mask_y, 2, 2], 1e-8)) * 2.0
        q_flat[mask_y, 3] = (m_flat[mask_y, 0, 2] - m_flat[mask_y, 2, 0]) / t
        q_flat[mask_y, 0] = (m_flat[mask_y, 0, 1] + m_flat[mask_y, 1, 0]) / t
        q_flat[mask_y, 1] = 0.25 * t
        q_flat[mask_y, 2] = (m_flat[mask_y, 1, 2] + m_flat[mask_y, 2, 1]) / t

    mask_z = (~mask_t) & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        t = np.sqrt(np.maximum(1.0 + m_flat[mask_z, 2, 2] - m_flat[mask_z, 0, 0] - m_flat[mask_z, 1, 1], 1e-8)) * 2.0
        q_flat[mask_z, 3] = (m_flat[mask_z, 1, 0] - m_flat[mask_z, 0, 1]) / t
        q_flat[mask_z, 0] = (m_flat[mask_z, 0, 2] + m_flat[mask_z, 2, 0]) / t
        q_flat[mask_z, 1] = (m_flat[mask_z, 1, 2] + m_flat[mask_z, 2, 1]) / t
        q_flat[mask_z, 2] = 0.25 * t

    q_flat = _normalize_vector(q_flat).astype(np.float32)
    return q_flat.reshape(m.shape[:-2] + (4,))


def rotation_6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matrix shape (..., 3, 3), got {m.shape}")
    col0 = m[..., :, 0]
    col1 = m[..., :, 1]
    return np.concatenate([col0, col1], axis=-1).astype(np.float32)


def matrix_from_rotation_6d(rot6d: np.ndarray) -> np.ndarray:
    r = np.asarray(rot6d, dtype=np.float32)
    if r.shape[-1] != 6:
        raise ValueError(f"Expected rot6d shape (..., 6), got {r.shape}")
    x_raw = r[..., 0:3]
    y_raw = r[..., 3:6]

    x = _normalize_vector(x_raw)
    z = _normalize_vector(_cross(x, y_raw))
    y = _cross(z, x)

    mat = np.stack([x, y, z], axis=-1)
    return mat.astype(np.float32)


def quaternion_xyzw_to_rotation_6d(quat_xyzw: np.ndarray) -> np.ndarray:
    return rotation_6d_from_matrix(quaternion_xyzw_to_matrix(quat_xyzw))


def rotation_6d_to_quaternion_xyzw(rot6d: np.ndarray) -> np.ndarray:
    return matrix_to_quaternion_xyzw(matrix_from_rotation_6d(rot6d))


def pose_xyzquat_to_xyzrot6d(pose_xyzquat: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_xyzquat, dtype=np.float32)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected pose shape (..., 7), got {pose.shape}")
    pos = pose[..., :3]
    quat = pose[..., 3:7]
    rot6d = quaternion_xyzw_to_rotation_6d(quat)
    return np.concatenate([pos, rot6d], axis=-1).astype(np.float32)


def pose_xyzrot6d_to_xyzquat(pose_xyzrot6d: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_xyzrot6d, dtype=np.float32)
    if pose.shape[-1] != 9:
        raise ValueError(f"Expected pose shape (..., 9), got {pose.shape}")
    pos = pose[..., :3]
    rot6d = pose[..., 3:9]
    quat = rotation_6d_to_quaternion_xyzw(rot6d)
    return np.concatenate([pos, quat], axis=-1).astype(np.float32)