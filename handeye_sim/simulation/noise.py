from __future__ import annotations

import numpy as np

from handeye_sim.core.se3 import Array, add_pose_noise


def add_noise_to_observations(
    T_C_P_list: list[Array],
    rot_sigma_deg: float,
    trans_sigma: float,
    seed: int = 7,
) -> list[Array]:
    """
    Add Gaussian pose noise to visual observations ^C T_P.

    In real calibration, this noise approximates corner localization noise,
    imperfect PnP, lens model residuals, board manufacturing error, etc.
    """
    rng = np.random.default_rng(seed)
    return [add_pose_noise(T, rot_sigma_deg, trans_sigma, rng) for T in T_C_P_list]


def add_noise_to_robot_poses(
    T_B_H_list: list[Array],
    rot_sigma_deg: float,
    trans_sigma: float,
    seed: int = 13,
) -> list[Array]:
    """
    Optional robot pose noise. Industrial robots are often much cleaner than
    vision observations, so default tests usually keep this very small or zero.
    """
    rng = np.random.default_rng(seed)
    return [add_pose_noise(T, rot_sigma_deg, trans_sigma, rng) for T in T_B_H_list]
