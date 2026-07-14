import os
import numpy as np
import cv2

from camera import RealSenseD435

extrinsic_matrix = np.array([
    [-9.983442881914330602e-01, 5.484553023173414921e-02, -1.733926322370462833e-02, -5.252487976502869804e-01],
[5.702310027855138352e-02, 9.040819869576690593e-01, -4.235376333849169006e-01, 3.571823023143709253e-01],
[-7.553030528420674912e-03, -4.238251156695102551e-01, -9.057125499061843277e-01, 3.183161848612462985e-01],
[0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
], dtype=np.float32)

workspace_limits = [[-0.621, -0.421], [0.0, 0.200]]

# workspace_limits = [[0.425, 0.725], [-0.085, 0.215]] # x min max, y min max
# workspace_limits = [[-0.766, -0.466], [-0.140, 0.160]] # x min max, y min max
# workspace_limits = [[-0.628, -0.428], [-0.068, 0.132]] # x min max, y min max (200x200)
# workspace_limits_mm = np.array([[277.0, 577.0], [-80.0, 215.0]])
# workspace_limits_mm = [[0.425, 0.725], [-0.085, 0.215]] # x min max, y min max

# workspace_limits_t = [[0.425, 0.575], [-0.082, 0.068]] # x min max, y min max (15 15)

# 픽셀당 해상도 (0.002 = 1픽셀당 2mm 해상도의 격자)
heightmap_resolution = 0.001

def mouse_move_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        bgr_value = img[y, x]
        
        print(f"마우스 위치: ({x:4d}, {y:4d}) | BGR 색상: {bgr_value}", end="\r")

def main():
    cam = RealSenseD435(
        color_resolution=720, 
        depth_mode="720P",
        extrinsic_matrix=extrinsic_matrix,
    )
    while True:
        try:
            color_img, raw_depth_img = cam.get_image()
            print(f"@@@{raw_depth_img.shape}")
            color_heightmap, depth_heightmap = cam.get_heightmap(workspace_limits, heightmap_resolution)

            # if color_heightmap is not None or color_heightmap.size > 0:
                # continue
            view_color = color_heightmap.copy()
            view_depth = depth_heightmap.copy()
            view_depth_1 = view_depth.copy()

            gray_heightmap = cv2.cvtColor(view_color, cv2.COLOR_RGB2GRAY)
            hole_mask = (gray_heightmap == 0).astype(np.uint8) * 255

            clean_color_heightmap = cv2.inpaint(view_color, hole_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

            # valid_depth_mask = view_depth > 0
            # depth_visual = np.zeros_like(view_depth, dtype=np.uint8)

            # if np.any(valid_depth_mask):
            #     d_min = view_depth[valid_depth_mask].min()
            #     d_max = view_depth[valid_depth_mask].max()
            # if d_max - d_min > 0:
            #     depth_visual[valid_depth_mask] = ((view_depth[valid_depth_mask] - d_min) / (d_max - d_min) * 255).astype(np.uint8)

            # depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)


            # view_color_bgr = cv2.cvtColor(clean_color_heightmap, cv2.COLOR_RGB2BGR)
            
            # # 0630
            # view_color = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR) # RGB -> BGR 변환


            # valid_depth_mask = depth_heightmap > 0
            # depth_visual = np.zeros_like(depth_heightmap, dtype=np.uint8)
            
            # if np.any(valid_depth_mask):
            #     d_min = depth_heightmap[valid_depth_mask].min()
            #     d_max = depth_heightmap[valid_depth_mask].max()
            #     if d_max - d_min > 0:
            #         depth_visual[valid_depth_mask] = ((depth_heightmap[valid_depth_mask] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            
            # depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

            # cv2.imshow("Original Color image", color_img)
            # cv2.imwrite('./output.png', color_img)
            # cv2.imshow("Original color_heightmap", clean_color_heightmap)
            color_img_rotated = cv2.rotate(clean_color_heightmap, cv2.ROTATE_90_CLOCKWISE)
            depth_img_rotated = cv2.rotate(view_depth_1, cv2.ROTATE_90_CLOCKWISE)
            depth_img_rotated = depth_img_rotated * 1000
            cv2.imshow("Orthographic Color111 Heightmap", color_img_rotated)
            # cv2.imshow("Orthographic Color Heightmap", depth_heightmap)
            # cv2.imwrite('./output.png', color_img_rotated)
            # cv2.imshow("Orthographic Color Heightmap", color_heightmap)
            # cv2.imshow("Orthographic Depth Heightmap", depth_heightmap)

            # cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"error: {e}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()