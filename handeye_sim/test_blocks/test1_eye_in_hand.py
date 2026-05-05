from __future__ import annotations

import cv2
import numpy as np

from handeye_sim.core.se3 import format_T
from handeye_sim.simulation.scene import (
    create_ground_truth_scene,
    compute_target_to_camera_observations,
)
from handeye_sim.simulation.noise import add_noise_to_observations, add_noise_to_robot_poses
from handeye_sim.calibration.opencv_handeye import calibrate_eye_in_hand_opencv
from handeye_sim.evaluation.metrics import handeye_error, board_pose_closure_errors


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def test1() -> None:
    """
    Test block 1: Eye-in-Hand simulation.

    Goal:
        Estimate ^H T_C, the camera-to-hand transform.

    Coordinate frames:
        B: robot base
        H: robot hand / flange
        C: camera
        P: calibration board / pattern

    Simulated measurements:
        Robot provides ^B T_H_i.
        Vision/PnP provides ^C T_P_i.

    OpenCV calibrateHandEye returns:
        ^H T_C.
    """
    np.set_printoptions(precision=6, suppress=True)

    print("\n========== test1: Eye-in-Hand hand-eye calibration simulation ==========")

    # 1. Ground truth generation.
    scene = create_ground_truth_scene(num_poses=25)
    T_C_P_clean = compute_target_to_camera_observations(scene)

    print("\n[Ground Truth Coordinate Setup]")
    print("B: robot base")
    print("H: robot hand/flange")
    print("C: camera")
    print("P: calibration board")
    print(format_T("GT ^H T_C  (camera -> hand, target hand-eye)", scene.T_H_C))
    print(format_T("GT ^B T_P  (pattern -> base, fixed board)", scene.T_B_P))
    print(f"Board points in P: shape = {scene.board_points_P.shape}")
    print("First five board points in P:")
    print(scene.board_points_P[:5])

    # 2. Add Gaussian noise to observations.
    # Visual pose noise is intentionally small but nonzero.
    # Units: translation in meters, rotation in degrees.
    T_C_P_noisy = add_noise_to_observations(
        T_C_P_clean,
        rot_sigma_deg=0.15,
        trans_sigma=0.0015,
        seed=7,
    )

    # Robot poses can also be noisy. Here we keep it tiny to represent a good robot.
    T_B_H_noisy = add_noise_to_robot_poses(
        scene.T_B_H_list,
        rot_sigma_deg=0.02,
        trans_sigma=0.0002,
        seed=13,
    )

    print("\n[Noisy Observations]")
    print("Robot observations: noisy ^B T_H_i")
    print("Vision observations: noisy ^C T_P_i")
    print(format_T("Example noisy ^B T_H_0", T_B_H_noisy[0]))
    print(format_T("Example noisy ^C T_P_0", T_C_P_noisy[0]))

    # 3. Estimate hand-eye transform with OpenCV.
    print("\n[Estimate ^H T_C with OpenCV calibrateHandEye]")
    for method_name, method_id in METHODS.items():
        try:
            T_H_C_est = calibrate_eye_in_hand_opencv(
                T_B_H_list=T_B_H_noisy,
                T_C_P_list=T_C_P_noisy,
                method=method_id,
            )
            err = handeye_error(T_H_C_est, scene.T_H_C)
            closure_rot_deg, closure_trans_m = board_pose_closure_errors(
                T_B_H_list=T_B_H_noisy,
                T_C_P_list=T_C_P_noisy,
                T_H_C=T_H_C_est,
            )

            print(f"\n--- Method: {method_name} ---")
            print(format_T("Estimated ^H T_C", T_H_C_est))
            print(f"Rotation error    : {err.rotation_deg:.6f} deg")
            print(f"Translation error : {err.translation_mm:.6f} mm")
            print(f"Board closure mean rotation spread    : {closure_rot_deg:.6f} deg")
            print(f"Board closure mean translation spread : {1000.0 * closure_trans_m:.6f} mm")
        except cv2.error as e:
            print(f"\n--- Method: {method_name} failed ---")
            print(str(e))

    print("\n[Core chain used in simulation]")
    print("^B T_C_i = ^B T_H_i * ^H T_C")
    print("^C T_P_i = inverse(^B T_C_i) * ^B T_P")
    print("OpenCV input : ^B T_H_i and ^C T_P_i")
    print("OpenCV output: ^H T_C")
    print("====================================================================\n")
