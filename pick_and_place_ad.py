import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import DR_init

from robot import Robot
from camera import RealSenseD435
from ultralytics import YOLO
from test_yolo import detect_cubes_once
from test_rd import RDInspector, crop_cube_by_bbox

from config import ROBOT_ID, ROBOT_MODEL


# ============================================================
# 경로 및 설정
# ============================================================

PROJECT_ROOT = Path(
    "/home/jooyeon/ros2_ws/src/doosan-robot2"
)

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "train-6"
    / "weights"
    / "best.pt"
)

DEBUG_CROP_DIR = (
    PROJECT_ROOT
    / "rd_results"
    / "robot_crops"
)

# 정상 큐브 배치 위치
GOOD_PLACE_X = -548.0
GOOD_PLACE_Y = -315.0

# 불량 의심 큐브 배치 위치
BAD_PLACE_X = -400.0
BAD_PLACE_Y = -315.0

# 학습 데이터 생성 시 사용한 값과 같아야 합니다.
CROP_PADDING_RATIO = 0.15


# ============================================================
# 좌표 변환 행렬
# ============================================================

EXTRINSIC_MATRIX = np.array(
    [
        [
            -9.913940669321062993e-01,
            1.305577585829072740e-01,
            -9.616429983419437152e-03,
            -5.324902337851710188e-01,
        ],
        [
            1.186279247674202064e-01,
            8.648774877739129341e-01,
            -4.877646426374731559e-01,
            2.981576563494563459e-01,
        ],
        [
            -5.536442465332774149e-02,
            -4.847077499026544167e-01,
            -8.729222059651777776e-01,
            3.296726497672328171e-01,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ],
    dtype=np.float32,
)


TASK_HOME_JOINT = [
    -180.00,
    0.00,
    90.00,
    0.00,
    90.00,
    60.00,
]

STIFFNESS = [
    3000,
    3000,
    150,
    200,
    200,
    200,
]


# ============================================================
# 확인용 Crop 이미지 저장
# ============================================================

def save_debug_crop(
    cube_crop,
    cube_index,
    rd_result,
):
    DEBUG_CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_name = (
        "bad"
        if rd_result["is_defective"]
        else "good"
    )

    output_path = (
        DEBUG_CROP_DIR
        / (
            f"cube_{cube_index:02d}_"
            f"{result_name}_"
            f"{rd_result['score']:.5f}.jpg"
        )
    )

    success = cv2.imwrite(
        str(output_path),
        cube_crop,
    )

    if success:
        print(
            "Crop 이미지 저장:",
            output_path,
        )
    else:
        print(
            "Crop 이미지 저장 실패:",
            output_path,
        )


# ============================================================
# 카메라 좌표 → 로봇 좌표
# ============================================================

def convert_camera_to_robot_coordinate(
    camera,
    center_x,
    center_y,
    cam_z,
):
    color_intrinsics = (
        camera._color_intrinsics_mat
    )

    cam_x = (
        center_x
        - color_intrinsics[0][2]
    ) * (
        cam_z
        / color_intrinsics[0][0]
    )

    cam_y = (
        center_y
        - color_intrinsics[1][2]
    ) * (
        cam_z
        / color_intrinsics[1][1]
    )

    camera_coordinate = np.array(
        [
            cam_x,
            cam_y,
            cam_z,
        ],
        dtype=np.float32,
    )

    world_coordinate = (
        EXTRINSIC_MATRIX[0:3, 0:3]
        @ camera_coordinate
    )

    world_coordinate += (
        EXTRINSIC_MATRIX[0:3, 3]
    )

    return world_coordinate


# ============================================================
# 메인
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node(
        "cube_inspection_robot",
        namespace=ROBOT_ID,
    )

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

    set_robot_mode(
        ROBOT_MODE_AUTONOMOUS
    )

    robot = None
    camera = None
    compliance_enabled = False

    try:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                "YOLO 모델이 없습니다:\n"
                f"{YOLO_MODEL_PATH}"
            )

        robot = Robot(node)

        camera = RealSenseD435(
            color_resolution=720,
            depth_mode="720P",
        )

        print("YOLO 모델 로드")

        yolo_model = YOLO(
            str(YOLO_MODEL_PATH)
        )

        print("RD 모델 로드")

        # 시작할 때 모델과 threshold를 한 번만 준비합니다.
        rd_inspector = RDInspector()

        # 새 test_yolo.py 반환 형식:
        # cube_list: dict 리스트
        # color_image: 원본 BGR 이미지
        cube_list, color_image = (
            detect_cubes_once(
                camera=camera,
                model=yolo_model,
                confidence_threshold=0.4,
                depth_radius=3,
                show_result=False,
            )
        )

        if not cube_list:
            print("탐지된 큐브가 없습니다.")
            return

        print(
            "탐지된 큐브 수:",
            len(cube_list),
        )

        robot.move_j(
            TASK_HOME_JOINT
        )

        for index, cube in enumerate(
            cube_list,
            start=1,
        ):
            print(
                f"\n===== 큐브 {index} 검사 ====="
            )

            # 새 test_yolo.py는 딕셔너리로 반환합니다.
            center_x = cube["center_x"]
            center_y = cube["center_y"]
            cam_z = cube["depth"]
            yaw = cube["yaw"]
            bbox = cube["bbox"]
            confidence = cube["confidence"]

            print(
                f"center=({center_x}, {center_y})"
            )
            print(
                f"depth={cam_z:.4f}"
            )
            print(
                f"yaw={yaw:.2f}"
            )
            print(
                f"bbox={bbox}"
            )
            print(
                f"confidence={confidence:.3f}"
            )

            # ------------------------------------------------
            # RD 불량 검사
            # ------------------------------------------------

            # 학습 데이터와 동일한 YOLO bbox Crop 방식
            cube_crop = crop_cube_by_bbox(
                image=color_image,
                bbox=bbox,
                padding_ratio=CROP_PADDING_RATIO,
            )

            rd_result = (
                rd_inspector.predict_cv_image(
                    cube_crop
                )
            )

            print(
                f"큐브 {index} RD 결과:",
                rd_result["prediction"],
            )

            print(
                f"score="
                f"{rd_result['score']:.6f}, "
                f"threshold="
                f"{rd_result['threshold']:.6f}"
            )

            save_debug_crop(
                cube_crop=cube_crop,
                cube_index=index,
                rd_result=rd_result,
            )

            # 판정에 따라 배치 위치 결정
            if rd_result["is_defective"]:
                place_x = BAD_PLACE_X
                place_y = BAD_PLACE_Y

                print(
                    "불량 의심 위치로 분류합니다."
                )
            else:
                place_x = GOOD_PLACE_X
                place_y = GOOD_PLACE_Y

                print(
                    "정상 위치로 분류합니다."
                )

            # ------------------------------------------------
            # 로봇 초기 위치 및 그리퍼 열기
            # ------------------------------------------------

            robot.move_j(
                TASK_HOME_JOINT
            )

            robot.release()

            # ------------------------------------------------
            # 좌표 변환
            # ------------------------------------------------

            if yaw > 90:
                yaw -= 180

            if yaw < -90:
                yaw += 180

            world_coordinate = (
                convert_camera_to_robot_coordinate(
                    camera=camera,
                    center_x=center_x,
                    center_y=center_y,
                    cam_z=cam_z,
                )
            )

            print(
                "world_coordinate:\n",
                world_coordinate,
            )

            robot_x = float(
                world_coordinate[0] * 1000
            )

            robot_y = float(
                world_coordinate[1] * 1000
            )

            robot_z = -160.0

            print(
                f"robot coordinate: "
                f"x={robot_x:.2f}, "
                f"y={robot_y:.2f}, "
                f"z={robot_z:.2f}"
            )

            # ------------------------------------------------
            # 그리퍼 Yaw 맞추기
            # ------------------------------------------------

            prepare_gripper_yaw = list(
                get_current_pose(0)
            )

            prepare_gripper_yaw[5] = (
                prepare_gripper_yaw[5]
                - yaw
            )

            robot.move_j(
                prepare_gripper_yaw
            )

            # ------------------------------------------------
            # 큐브 위로 이동
            # ------------------------------------------------

            grasp_ready_pose = list(
                get_current_pose(1)
            )

            grasp_ready_pose[0] = robot_x
            grasp_ready_pose[1] = robot_y
            grasp_ready_pose[2] = robot_z

            # TODO
            robot.move_l(grasp_ready_pose)
            robot.move_l(
                grasp_ready_pose
            )

            grasp_pose = (
                grasp_ready_pose.copy()
            )

            grasp_pose[2] = -226.0

            # ------------------------------------------------
            # 힘 감지 하강
            # ------------------------------------------------

            task_compliance_ctrl(
                stx=STIFFNESS
            )

            compliance_enabled = True
            collision_detected = False

            start_z = int(
                grasp_ready_pose[2]
            )

            target_z = int(
                grasp_pose[2]
            )

            for current_z in range(
                start_z - 5,
                target_z - 1,
                -10,
            ):
                step_pose = (
                    grasp_ready_pose.copy()
                )

                step_pose[2] = float(
                    current_z
                )

                amovel(
                    step_pose,
                    vel=10,
                    acc=10,
                )

                check_start_time = time.time()

                while (
                    time.time()
                    - check_start_time
                    < 0.2
                ):
                    force_result = (
                        check_force_condition(
                            axis=DR_AXIS_Z,
                            min=10,
                            ref=DR_TOOL,
                        )
                    )

                    if force_result == 0:
                        print(
                            "Collision Detection!"
                        )

                        collision_detected = True
                        break

                if collision_detected:
                    break

            release_compliance_ctrl()
            compliance_enabled = False

            # 기존 로직 유지:
            # 충돌이 감지되면 집지 않고 건너뜁니다.
            if collision_detected:
                robot.move_l(
                    grasp_ready_pose
                )

                print(
                    "충돌로 인해 이 큐브를 "
                    "건너뜁니다."
                )

                continue

            # ------------------------------------------------
            # 큐브 집기
            # ------------------------------------------------

            robot.grasp()

            robot.move_l(
                grasp_ready_pose
            )

            # ------------------------------------------------
            # 정상 또는 불량 위치로 이동
            # ------------------------------------------------

            place_pose = (
                grasp_ready_pose.copy()
            )

            place_pose[0] = place_x
            place_pose[1] = place_y

            robot.move_l(
                place_pose
            )

            robot.release()

            print(
                f"큐브 {index} 분류 완료:",
                rd_result["prediction"],
            )

        robot.move_j(
            TASK_HOME_JOINT
        )

    except Exception as error:
        print(
            "\n프로그램 실행 중 오류:",
            error,
        )

        raise

    finally:
        # 예외 중에도 순응 제어가 켜져 있으면 해제
        if compliance_enabled:
            try:
                release_compliance_ctrl()
            except Exception as error:
                print(
                    "순응 제어 해제 실패:",
                    error,
                )

        if camera is not None:
            try:
                if hasattr(camera, "stop"):
                    camera.stop()
            except Exception as error:
                print(
                    "카메라 종료 실패:",
                    error,
                )

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()