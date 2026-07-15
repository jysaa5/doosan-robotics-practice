import time
import numpy as np
import rclpy
import DR_init
from robot import Robot
from camera import RealSenseD435
from ultralytics import YOLO
from test_yolo import detect_cubes_once

from config import ROBOT_ID, ROBOT_MODEL

def main(args=None):
    rclpy.init(args=args)
    
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node("example_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import (
        set_robot_mode,
        ROBOT_MODE_AUTONOMOUS,
        get_current_pose,
        task_compliance_ctrl,
        release_compliance_ctrl,
        amovel,
        check_force_condition,
        DR_AXIS_Z,
        DR_TOOL,
    )

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    robot = Robot(node)
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")
    yolo_model = YOLO("/home/jooyeon/ros2_ws/src/doosan-robot2/runs/segment/train-6/weights/best.pt")
    cube_list = detect_cubes_once(camera, yolo_model)

    extrinsic_matrix = np.array([
    [-9.913940669321062993e-01, 1.305577585829072740e-01, -9.616429983419437152e-03, -5.324902337851710188e-01],
    [1.186279247674202064e-01, 8.648774877739129341e-01, -4.877646426374731559e-01, 2.981576563494563459e-01],
    [-5.536442465332774149e-02, -4.847077499026544167e-01, -8.729222059651777776e-01, 3.296726497672328171e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ], dtype=np.float32)

    task_home_joint = [-180.00, 0.00, 90.00, 0.00, 90.00, 60.00]
    stiffness = [3000, 3000, 150, 200, 200, 200]

    for i in range(len(cube_list)):
        robot.move_j(task_home_joint)
        robot.release()

        center_x = cube_list[i][0]
        center_y = cube_list[i][1]
        cam_z = cube_list[i][2]
        yaw = cube_list[i][3]

        if yaw > 90:
            yaw = yaw - 180

        cam_x = np.multiply(
            center_x - camera._color_intrinsics_mat[0][2],
            cam_z / camera._color_intrinsics_mat[0][0],
        )
        cam_y = np.multiply(
            center_y - camera._color_intrinsics_mat[1][2],
            cam_z / camera._color_intrinsics_mat[1][1],
        )

        cam_coordinate = [cam_x, cam_y, cam_z, yaw]
        world_coordinate = extrinsic_matrix[0:3, 0:3] @ cam_coordinate[0:3]
        world_coordinate += extrinsic_matrix[0:3, 3:].flatten()

        print(f"world_coord\n{world_coordinate}")

        robot_x = world_coordinate[0] * 1000
        robot_y = world_coordinate[1] * 1000
        robot_z = -160

        prepare_gripper_yaw = list(get_current_pose())
        prepare_gripper_yaw[3] = prepare_gripper_yaw[3] - yaw
        robot.move_j(prepare_gripper_yaw)

        grasp_ready_pose = list(get_current_pose()[1])
        grasp_ready_pose[0] = robot_x
        grasp_ready_pose[1] = robot_y
        grasp_ready_pose[2] = robot_z
        robot.move_l(grasp_ready_pose)

        grasp_pose = grasp_ready_pose.copy()
        grasp_pose[2] = -220

        task_compliance_ctrl(stx=stiffness)

        collision_detected = False
        start_z = int(grasp_ready_pose[2])
        target_z = int(grasp_pose[2])

        for current_z in range(start_z - 5, target_z - 1, -5):
            step_pose = grasp_ready_pose.copy()
            step_pose[2] = float(current_z)

            amovel(step_pose, vel=10, acc=10)

            t_start = time.time()
            while time.time() - t_start < 0.2:
                if check_force_condition(axis=DR_AXIS_Z, min=5, ref=DR_TOOL) == 0:
                    print("Collision Detection!!!")
                    collision_detected = True
                    break

            if collision_detected:
                break

        if collision_detected:
            robot.move_l(grasp_ready_pose)
            release_compliance_ctrl()
        else:
            release_compliance_ctrl()
            robot.grasp()
            robot.move_l(grasp_ready_pose)

            place_pose = grasp_ready_pose.copy()
            place_pose[0] = -425
            place_pose[1] = -300

            robot.move_l(place_pose)
            robot.release()
            robot.move_j(task_home_joint)

if __name__ == "__main__":
    main()