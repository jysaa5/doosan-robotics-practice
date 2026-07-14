import cv2
from camera import RealSenseD435

def main():
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")
    color_img, depth_img = camera.get_image()

    cv2.imshow("Original Color image", color_img)
    cv2.imshow("Original Depth image", depth_img)

    cv2.imwrite("/home/hyu/ros2_ws/src/doosan-robot2/color_img.png", color_img)
    cv2.imwrite("/home/hyu/ros2_ws/src/doosan-robot2/depth_img.png", depth_img)

    print(f"Color image's shape: {color_img.shape}")
    print(f"Color pixel value: {color_img[360][640]}")
    print(f"Depth image's shape: {depth_img.shape}")
    print(f"Depth image's Y: {depth_img.shape[0]}")
    print(f"Depth image's X: {depth_img.shape[1]}")
    print(f"Depth value: {depth_img[360][640]}")

if __name__ == "__main__":
    main()