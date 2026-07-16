from pathlib import Path

import cv2
from ultralytics import YOLO

from camera import RealSenseD435
from test_yolo import detect_cubes_once
from test_rd import (
    RDInspector,
    crop_cube_by_bbox,
)


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

RESULT_DIR = (
    PROJECT_ROOT
    / "rd_results"
    / "inspection"
)

CROP_DIR = (
    RESULT_DIR
    / "crops"
)


def draw_result(
    image,
    bbox,
    prediction,
    score,
    threshold,
):
    x1, y1, x2, y2 = [
        int(value)
        for value in bbox
    ]

    if prediction == "불량 의심":
        color = (0, 0, 255)
        text_prediction = "BAD_SUSPECT"
    else:
        color = (0, 255, 0)
        text_prediction = "GOOD"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    text = (
        f"{text_prediction} "
        f"{score:.5f}/{threshold:.5f}"
    )

    text_y = max(
        25,
        y1 - 10,
    )

    cv2.putText(
        image,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = None

    try:
        print("YOLO 모델 로드")

        yolo_model = YOLO(
            str(YOLO_MODEL_PATH)
        )

        print("RD 모델 로드")

        rd_inspector = RDInspector()

        print("카메라 시작")

        camera = RealSenseD435(
            color_resolution=720,
            depth_mode="720P",
        )

        cube_list, color_image = (
            detect_cubes_once(
                camera,
                yolo_model,
            )
        )

        if not cube_list:
            print(
                "탐지된 큐브가 없습니다."
            )
            return

        display_image = (
            color_image.copy()
        )

        for index, cube in enumerate(
            cube_list,
            start=1,
        ):
            bbox = cube["bbox"]

            cube_crop = (
                crop_cube_by_bbox(
                    image=color_image,
                    bbox=bbox,
                    padding_ratio=0.15,
                )
            )

            result = (
                rd_inspector.predict_cv_image(
                    cube_crop
                )
            )

            print(
                f"\nCube {index}"
            )

            print(
                "판정:",
                result["prediction"],
            )

            print(
                f"score="
                f"{result['score']:.6f}"
            )

            print(
                f"threshold="
                f"{result['threshold']:.6f}"
            )

            crop_name = (
                f"cube_{index:02d}_"
                f"{'bad' if result['is_defective'] else 'good'}"
                f".jpg"
            )

            cv2.imwrite(
                str(
                    CROP_DIR
                    / crop_name
                ),
                cube_crop,
            )

            draw_result(
                image=display_image,
                bbox=bbox,
                prediction=result[
                    "prediction"
                ],
                score=result["score"],
                threshold=result[
                    "threshold"
                ],
            )

        result_path = (
            RESULT_DIR
            / "inspection_result.jpg"
        )

        cv2.imwrite(
            str(result_path),
            display_image,
        )

        print(
            "\n결과 저장:",
            result_path,
        )

        cv2.imshow(
            "Cube RD Inspection",
            display_image,
        )

        cv2.waitKey(0)

    finally:
        if camera is not None:
            if hasattr(camera, "stop"):
                camera.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()