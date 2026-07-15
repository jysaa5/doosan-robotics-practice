import os
import cv2
import numpy as np
from camera import RealSenseD435

output_dir = "/home/jooyeon/ros2_ws/src/doosan-robot2/cube_dataset"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/color", exist_ok=True)
os.makedirs(f"{output_dir}/depth", exist_ok=True)

camera = RealSenseD435(color_resolution=720, depth_mode="720P")
count = 0

print("=== RealSense Data Collect ===")
print(" - [Spacebar] : Save Image (RGB & Depth)")
print(" - [ESC]      : Exit")

try:
    while True:
        color_image, depth_meters = camera.get_image()
        depth_image = (depth_meters * 1000).astype(np.uint16)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        images = np.hstack((color_image, depth_colormap))
        cv2.putText(
            images,
            f"Saved: {count} images",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("RealSense Data Collection", images)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Exit...")
            break
        elif key == ord(" "):
            color_path = f"{output_dir}/color/cube_{count:03d}.png"
            depth_path = f"{output_dir}/depth/cube_{count:03d}.png"

            cv2.imwrite(color_path, color_image)
            cv2.imwrite(depth_path, depth_image.astype(np.uint16))

            print(f"Save complete {color_path} & {depth_path}")
            count += 1

finally:
    if hasattr(camera, "_pipeline"):
        camera._pipeline.stop()
    cv2.destroyAllWindows()