import cv2
import numpy as np
from ultralytics import YOLO
from camera import RealSenseD435

def detect_cubes_once(camera, model):
    print("\n=== Start testing the one-time cube detection function ===")

    color_image, depth_image = camera.get_image()

    results = model(color_image, conf=0.6, verbose=False)

    # Create image for visualization
    annotated_frame = results[0].plot()

    # List of cube data to return last
    detected_cubes = []

    # Calculate center point, distance, and rotation angle based on segmentation mask information
    if results[0].masks is not None:
        for box, segments in zip(results[0].boxes, results[0].masks.xy):
            xyxy = box.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)

            # Check if pixels are valid within the 1280x720 image range
            if 0 <= cx < 1280 and 0 <= cy < 720:
                distance_m = depth_image[cy, cx]

                if distance_m > 0:
                    # --- [Calculation of rotation angle] ---
                    contours = segments.astype(np.int32)
                    rect = cv2.minAreaRect(contours)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int0(box_points)

                    angle = rect[2]

                    (width, height) = rect[1]
                    if width < height:
                        angle = angle + 90.0

                    # [pixel X, pixel Y, Distance Z (m), Rotation Angle (degree)]
                    detected_cubes.append([cx, cy, float(distance_m), float(angle)])

                    # For visualization
                    cv2.drawContours(annotated_frame, [box_points], 0, (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                    text = f"Z:{distance_m:.3f}m R:{angle:.1f}deg"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

    cv2.imshow("Cube Detection Result (Single Frame)", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"\n[Detected] Length of returned list: {len(detected_cubes)}")
    print("------------------------------------------")

    for idx, cube in enumerate(detected_cubes):
        px, py, pz, rot = cube
        print(
            f"Cube_list[{idx}] -> Pixel X: {px}, Pixel Y: {py}, "
            f"Distance Z: {pz:.3f}m, Rotation Angle: {rot:.1f}°"
        )

    return detected_cubes

if __name__ == "__main__":
    model_path = "/home/hyu/runs/segment/train-8/weights/best.pt"
    yolo_model = YOLO(model_path)
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")

    try:
        cube_list = detect_cubes_once(camera, yolo_model)
    finally:
        if hasattr(camera, "_pipeline"):
            camera._pipeline.stop()
