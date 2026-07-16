import cv2
import numpy as np


def calculate_yaw_from_segment(segment):
    """
    YOLO segmentation polygon으로 큐브 회전 각도를 계산합니다.

    반환 범위:
        -90도 ~ 90도
    """

    if segment is None or len(segment) < 3:
        return 0.0

    contour = np.asarray(
        segment,
        dtype=np.float32,
    )

    rectangle = cv2.minAreaRect(contour)

    (_, _), (width, height), angle = rectangle

    if width <= 0 or height <= 0:
        return 0.0

    if width < height:
        yaw = angle
    else:
        yaw = angle + 90.0

    if yaw > 90.0:
        yaw -= 180.0

    if yaw < -90.0:
        yaw += 180.0

    return float(yaw)


def clamp_bbox(
    bbox,
    image_width,
    image_height,
):
    """
    Bounding box가 이미지 범위를 벗어나지 않도록 보정합니다.
    """

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(int(x1), image_width - 1),
    )

    y1 = max(
        0,
        min(int(y1), image_height - 1),
    )

    x2 = max(
        0,
        min(int(x2), image_width),
    )

    y2 = max(
        0,
        min(int(y2), image_height),
    )

    return x1, y1, x2, y2


def draw_detection_result(
    image,
    bbox,
    center_x,
    center_y,
    confidence,
    depth,
    yaw,
):
    """
    탐지 결과를 확인하기 위해 이미지에 bbox와 정보를 그립니다.
    """

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.circle(
        image,
        (center_x, center_y),
        5,
        (0, 0, 255),
        -1,
    )

    text = (
        f"conf={confidence:.2f} "
        f"depth={depth:.3f} "
        f"yaw={yaw:.1f}"
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
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def detect_cubes_once(
    camera,
    model,
    confidence_threshold=0.4,
    depth_radius=3,
    show_result=False,
):
    """
    RealSense 카메라에서 한 장을 받아 YOLO로 큐브를 탐지합니다.

    반환값:
        detected_cubes:
            [
                {
                    "center_x": int,
                    "center_y": int,
                    "depth": float,
                    "yaw": float,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": int,
                    "class_name": str,
                }
            ]

        color_image:
            원본 OpenCV BGR 이미지

    사용 예:
        cube_list, color_image = detect_cubes_once(
            camera,
            yolo_model,
        )
    """

    print("\n=== YOLO 큐브 탐지 시작 ===")

    color_image, depth_image = camera.get_image()

    if color_image is None:
        raise RuntimeError(
            "RealSense 컬러 이미지를 가져오지 못했습니다."
        )

    if depth_image is None:
        raise RuntimeError(
            "RealSense depth 이미지를 가져오지 못했습니다."
        )

    if color_image.ndim != 3:
        raise RuntimeError(
            "컬러 이미지 형식이 올바르지 않습니다."
        )

    if depth_image.ndim != 2:
        raise RuntimeError(
            "Depth 이미지 형식이 올바르지 않습니다."
        )

    color_height, color_width = color_image.shape[:2]
    depth_height, depth_width = depth_image.shape

    if (
        color_width != depth_width
        or color_height != depth_height
    ):
        print(
            "경고: Color와 Depth 해상도가 다릅니다."
        )
        print(
            "Color:",
            color_width,
            "x",
            color_height,
        )
        print(
            "Depth:",
            depth_width,
            "x",
            depth_height,
        )

    results = model(
        color_image,
        conf=confidence_threshold,
        verbose=False,
    )

    detected_cubes = []
    display_image = color_image.copy()

    if not results:
        print("YOLO 결과가 없습니다.")
        return detected_cubes, color_image

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("탐지된 큐브가 없습니다.")

        if show_result:
            cv2.imshow(
                "YOLO Cube Detection",
                display_image,
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detected_cubes, color_image

    names = result.names

    for index, box in enumerate(result.boxes):
        raw_bbox = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = clamp_bbox(
            raw_bbox,
            color_width,
            color_height,
        )

        if x2 <= x1 or y2 <= y1:
            print(
                f"Cube {index + 1}: "
                "유효하지 않은 bbox입니다."
            )
            continue

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        if not (
            0 <= center_x < depth_width
            and 0 <= center_y < depth_height
        ):
            print(
                f"Cube {index + 1}: "
                f"중심점이 depth 이미지 범위를 벗어났습니다. "
                f"center=({center_x}, {center_y})"
            )
            continue

        cam_z = camera.get_valid_depth(
            depth_image,
            center_x,
            center_y,
            radius=depth_radius,
        )

        if cam_z is None:
            print(
                f"Cube {index + 1}: "
                "유효한 depth 값을 찾지 못했습니다."
            )
            continue

        if not np.isfinite(cam_z) or cam_z <= 0:
            print(
                f"Cube {index + 1}: "
                f"잘못된 depth 값입니다: {cam_z}"
            )
            continue

        confidence = float(
            box.conf[0]
            .detach()
            .cpu()
            .item()
        )

        class_id = int(
            box.cls[0]
            .detach()
            .cpu()
            .item()
        )

        if isinstance(names, dict):
            class_name = names.get(
                class_id,
                str(class_id),
            )
        else:
            class_name = str(class_id)

        yaw = 0.0

        if result.masks is not None:
            mask_segments = result.masks.xy

            if index < len(mask_segments):
                yaw = calculate_yaw_from_segment(
                    mask_segments[index]
                )

        cube_data = {
            "center_x": center_x,
            "center_y": center_y,
            "depth": float(cam_z),
            "yaw": float(yaw),
            "bbox": [
                x1,
                y1,
                x2,
                y2,
            ],
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_name,
        }

        detected_cubes.append(
            cube_data
        )

        print(
            f"Cube {len(detected_cubes)} | "
            f"class={class_name} | "
            f"center=({center_x}, {center_y}) | "
            f"depth={cam_z:.4f} | "
            f"yaw={yaw:.2f} | "
            f"bbox={[x1, y1, x2, y2]} | "
            f"confidence={confidence:.3f}"
        )

        draw_detection_result(
            image=display_image,
            bbox=(x1, y1, x2, y2),
            center_x=center_x,
            center_y=center_y,
            confidence=confidence,
            depth=cam_z,
            yaw=yaw,
        )

    print(
        "최종 탐지 큐브 수:",
        len(detected_cubes),
    )

    if show_result:
        cv2.imshow(
            "YOLO Cube Detection",
            display_image,
        )

        print(
            "아무 키나 누르면 창이 닫힙니다."
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_cubes, color_image