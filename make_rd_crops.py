from pathlib import Path

import cv2
from ultralytics import YOLO


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

SOURCE_DATASET_ROOT = (
    PROJECT_ROOT
    / "rd_dataset"
)

OUTPUT_DATASET_ROOT = (
    PROJECT_ROOT
    / "rd_dataset_crop"
)

IMAGE_SIZE = 256

# bbox 주변 배경을 15% 포함
PADDING_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def crop_cube_by_bbox(
    image,
    bbox,
    padding_ratio=0.15,
):
    x1, y1, x2, y2 = bbox

    image_height, image_width = (
        image.shape[:2]
    )

    box_width = x2 - x1
    box_height = y2 - y1

    padding_x = int(
        box_width * padding_ratio
    )

    padding_y = int(
        box_height * padding_ratio
    )

    crop_x1 = max(
        0,
        int(x1) - padding_x,
    )

    crop_y1 = max(
        0,
        int(y1) - padding_y,
    )

    crop_x2 = min(
        image_width,
        int(x2) + padding_x,
    )

    crop_y2 = min(
        image_height,
        int(y2) + padding_y,
    )

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()

    if crop.size == 0:
        raise RuntimeError(
            "Crop 이미지 크기가 0입니다."
        )

    crop = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    return crop


def find_source_image_directory(
    split_name,
):
    candidates = [
        (
            SOURCE_DATASET_ROOT
            / split_name
            / "images"
            / "good"
        ),
        (
            SOURCE_DATASET_ROOT
            / split_name
            / "images"
        ),
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        image_count = sum(
            1
            for path in candidate.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        if image_count > 0:
            return candidate

    raise FileNotFoundError(
        f"{split_name} 이미지 폴더를 찾지 못했습니다."
    )


def process_split(
    model,
    split_name,
):
    source_directory = (
        find_source_image_directory(
            split_name
        )
    )

    output_directory = (
        OUTPUT_DATASET_ROOT
        / split_name
        / "images"
        / "good"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        path
        for path in source_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    print(
        f"\n===== {split_name} 처리 ====="
    )
    print(
        "원본 경로:",
        source_directory,
    )
    print(
        "원본 이미지 수:",
        len(image_paths),
    )

    saved_count = 0
    failed_count = 0

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                "이미지 읽기 실패:",
                image_path.name,
            )
            failed_count += 1
            continue

        results = model(
            image,
            conf=0.4,
            verbose=False,
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            print(
                "YOLO 탐지 실패:",
                image_path.name,
            )
            failed_count += 1
            continue

        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        best_index = int(
            confidences.argmax()
        )

        bbox = (
            boxes.xyxy[best_index]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        try:
            crop = crop_cube_by_bbox(
                image=image,
                bbox=bbox,
                padding_ratio=PADDING_RATIO,
            )

        except Exception as error:
            print(
                "Crop 실패:",
                image_path.name,
                error,
            )
            failed_count += 1
            continue

        output_path = (
            output_directory
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            crop,
        )

        if success:
            saved_count += 1
        else:
            failed_count += 1

    print(
        f"{split_name} 저장 완료:",
        saved_count,
    )

    print(
        f"{split_name} 실패:",
        failed_count,
    )


def main():
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO 모델이 없습니다:\n"
            f"{YOLO_MODEL_PATH}"
        )

    print(
        "YOLO 모델:",
        YOLO_MODEL_PATH,
    )

    model = YOLO(
        str(YOLO_MODEL_PATH)
    )

    for split_name in [
        "train",
        "valid",
        "test",
    ]:
        process_split(
            model=model,
            split_name=split_name,
        )

    print(
        "\n===== RD Crop 데이터 생성 완료 ====="
    )

    print(
        "저장 경로:",
        OUTPUT_DATASET_ROOT,
    )


if __name__ == "__main__":
    main()