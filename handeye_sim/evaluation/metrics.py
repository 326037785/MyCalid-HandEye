from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from handeye_sim.core.se3 import Array, split_T, inv_T, rotation_angle_error_deg, translation_error


@dataclass(frozen=True)
class HandEyeError:
    rotation_deg: float
    translation_m: float
    translation_mm: float


def handeye_error(T_H_C_est: Array, T_H_C_gt: Array) -> HandEyeError:
    R_est, t_est = split_T(T_H_C_est)
    R_gt, t_gt = split_T(T_H_C_gt)
    trans_m = translation_error(t_est, t_gt)
    return HandEyeError(
        rotation_deg=rotation_angle_error_deg(R_est, R_gt),
        translation_m=trans_m,
        translation_mm=1000.0 * trans_m,
    )


def board_pose_closure_errors(
    T_B_H_list: list[Array],
    T_C_P_list: list[Array],
    T_H_C: Array,
) -> tuple[float, float]:
    """
    Check consistency of the chain:
        ^B T_P_i = ^B T_H_i * ^H T_C * ^C T_P_i

    If calibration is good, all estimated ^B T_P_i should be almost identical.

    Returns:
        mean rotation spread in degrees relative to the first board pose,
        mean translation spread in meters relative to the first board pose.
    """
    T_B_P_list = []
    for i in range(len(T_B_H_list)):
        T_B_P_list.append(T_B_H_list[i] @ T_H_C @ T_C_P_list[i])

    T_ref = T_B_P_list[0]
    R_ref, t_ref = split_T(T_ref)

    rot_errors = []
    trans_errors = []
    for i in range(1, len(T_B_P_list)):
        R_i, t_i = split_T(T_B_P_list[i])
        rot_errors.append(rotation_angle_error_deg(R_i, R_ref))
        trans_errors.append(float(np.linalg.norm(t_i - t_ref)))

    return float(np.mean(rot_errors)), float(np.mean(trans_errors))
