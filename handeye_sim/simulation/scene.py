from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from handeye_sim.core.se3 import (
    Array,
    make_T,
    inv_T,
    euler_xyz_to_R,
)


@dataclass(frozen=True)
class GroundTruthScene:
    """
    Coordinate definitions for Eye-in-Hand simulation.

    B: robot base frame
    H: robot hand / flange frame
    C: camera frame
    P: calibration pattern / board frame

    T_B_H[i]: transform from H_i to B, i.e. ^B T_H_i.
    T_H_C: fixed hand-eye transform from C to H, i.e. ^H T_C.
    T_B_P: fixed calibration board transform from P to B, i.e. ^B T_P.
    board_points_P: 3D board corner points expressed in P.
    """
    T_H_C: Array
    T_B_P: Array
    T_B_H_list: list[Array]
    board_points_P: Array


def create_board_points(cols: int = 7, rows: int = 5, square_size: float = 0.025) -> Array:
    """Create chessboard-like 3D points on Z=0 plane in pattern frame P."""
    points = []
    for r in range(rows):
        for c in range(cols):
            points.append([c * square_size, r * square_size, 0.0])
    return np.asarray(points, dtype=np.float64)


def create_ground_truth_scene(num_poses: int = 25) -> GroundTruthScene:
    """
    Create a deterministic Eye-in-Hand scene.

    The board is fixed in robot base frame. The robot hand moves to several
    different poses. The camera is rigidly mounted on the hand.
    """
    # Ground-truth hand-eye: ^H T_C
    # Camera is roughly 8 cm forward, 2 cm left, 6 cm above the hand frame,
    # with a non-trivial mounting rotation.
    R_H_C = euler_xyz_to_R(rx=20.0, ry=-10.0, rz=35.0, degrees=True)
    t_H_C = np.array([0.08, -0.02, 0.06], dtype=np.float64)
    T_H_C = make_T(R_H_C, t_H_C)

    # Fixed board pose in robot base: ^B T_P
    R_B_P = euler_xyz_to_R(rx=0.0, ry=0.0, rz=15.0, degrees=True)
    t_B_P = np.array([0.55, 0.05, 0.20], dtype=np.float64)
    T_B_P = make_T(R_B_P, t_B_P)

    # Robot hand poses: ^B T_H_i
    # We intentionally vary both rotation and translation so AX=XB is well-conditioned.
    T_B_H_list = []
    for i in range(num_poses):
        a = 2.0 * np.pi * i / max(1, num_poses)

        tx = 0.38 + 0.10 * np.cos(a)
        ty = 0.02 + 0.12 * np.sin(a)
        tz = 0.42 + 0.04 * np.sin(2.0 * a)

        rx = 165.0 + 15.0 * np.sin(a)
        ry = -8.0 + 18.0 * np.cos(1.3 * a)
        rz = 35.0 + 55.0 * np.sin(0.7 * a)

        R_B_H = euler_xyz_to_R(rx=rx, ry=ry, rz=rz, degrees=True)
        t_B_H = np.array([tx, ty, tz], dtype=np.float64)
        T_B_H_list.append(make_T(R_B_H, t_B_H))

    board_points_P = create_board_points()
    return GroundTruthScene(
        T_H_C=T_H_C,
        T_B_P=T_B_P,
        T_B_H_list=T_B_H_list,
        board_points_P=board_points_P,
    )


def compute_target_to_camera_observations(scene: GroundTruthScene) -> list[Array]:
    """
    Generate perfect visual observations ^C T_P for each robot pose.

    Chain:
        ^B T_C_i = ^B T_H_i * ^H T_C
        ^C T_P_i = inverse(^B T_C_i) * ^B T_P

    This is what a PnP/chessboard detector would estimate from images.
    """
    T_C_P_list = []
    for T_B_H in scene.T_B_H_list:
        T_B_C = T_B_H @ scene.T_H_C
        T_C_P = inv_T(T_B_C) @ scene.T_B_P
        T_C_P_list.append(T_C_P)
    return T_C_P_list
