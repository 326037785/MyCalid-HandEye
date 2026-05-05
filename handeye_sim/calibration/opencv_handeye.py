from __future__ import annotations

import cv2

from handeye_sim.core.se3 import Array, split_T, make_T


def calibrate_eye_in_hand_opencv(
    T_B_H_list: list[Array],
    T_C_P_list: list[Array],
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> Array:
    """
    Estimate ^H T_C using OpenCV calibrateHandEye.

    OpenCV input convention:
        R_gripper2base, t_gripper2base: ^B T_H
        R_target2cam,  t_target2cam:    ^C T_P

    OpenCV output:
        R_cam2gripper, t_cam2gripper:   ^H T_C

    Therefore, this wrapper returns T_H_C_est.
    """
    if len(T_B_H_list) != len(T_C_P_list):
        raise ValueError("Robot pose count and visual observation count must match.")
    if len(T_B_H_list) < 3:
        raise ValueError("Hand-eye calibration needs at least several distinct poses.")

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(len(T_B_H_list)):
        R_B_H, t_B_H = split_T(T_B_H_list[i])
        R_C_P, t_C_P = split_T(T_C_P_list[i])
        R_gripper2base.append(R_B_H)
        t_gripper2base.append(t_B_H.reshape(3, 1))
        R_target2cam.append(R_C_P)
        t_target2cam.append(t_C_P.reshape(3, 1))

    R_H_C, t_H_C = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method,
    )

    return make_T(R_H_C, t_H_C.reshape(3))
