from __future__ import annotations

import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot


Array = np.ndarray


def make_T(R: Array, t: Array) -> Array:
    """Create a 4x4 homogeneous transform from R and t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def split_T(T: Array) -> tuple[Array, Array]:
    """Split 4x4 transform into R and t."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3].copy(), T[:3, 3].copy()


def inv_T(T: Array) -> Array:
    """Inverse of a rigid-body transform."""
    R, t = split_T(T)
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def axis_angle_to_R(axis: Array, angle_rad: float) -> Array:
    """Rotation matrix from axis-angle."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    rvec = axis * float(angle_rad)
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64)


def euler_xyz_to_R(rx: float, ry: float, rz: float, degrees: bool = True) -> Array:
    """Rotation matrix from XYZ Euler angles."""
    return SciRot.from_euler("xyz", [rx, ry, rz], degrees=degrees).as_matrix().astype(np.float64)


def random_rotation(max_angle_deg: float, rng: np.random.Generator) -> Array:
    """Small random rotation with angle in [-max_angle_deg, max_angle_deg]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = math.radians(rng.uniform(-max_angle_deg, max_angle_deg))
    return axis_angle_to_R(axis, angle)


def add_pose_noise(T: Array, rot_sigma_deg: float, trans_sigma: float, rng: np.random.Generator) -> Array:
    """
    Add left-multiplicative SE(3) noise to a pose.

    T_noisy = dT * T

    rot_sigma_deg: standard deviation of rotation noise in degrees.
    trans_sigma: standard deviation of translation noise in meters.
    """
    rot_noise_vec = rng.normal(loc=0.0, scale=math.radians(rot_sigma_deg), size=3)
    R_noise, _ = cv2.Rodrigues(rot_noise_vec.astype(np.float64))
    t_noise = rng.normal(loc=0.0, scale=trans_sigma, size=3)
    dT = make_T(R_noise, t_noise)
    return dT @ T


def rotation_angle_error_deg(R_est: Array, R_gt: Array) -> float:
    """Geodesic rotation error in degrees between two rotation matrices."""
    R_delta = np.asarray(R_est) @ np.asarray(R_gt).T
    trace_val = float(np.trace(R_delta))
    cos_theta = (trace_val - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def translation_error(t_est: Array, t_gt: Array) -> float:
    """Euclidean translation error."""
    return float(np.linalg.norm(np.asarray(t_est).reshape(3) - np.asarray(t_gt).reshape(3)))


def format_T(name: str, T: Array) -> str:
    """Human-readable 4x4 matrix string."""
    return f"{name} =\n{np.array2string(T, precision=6, suppress_small=True)}"
