import numpy as np
from camera import RealSenseD435

extrinsic_matrix = np.array([
    [-9.913940669321062993e-01, 1.305577585829072740e-01, -9.616429983419437152e-03, -5.324902337851710188e-01],
[1.186279247674202064e-01, 8.648774877739129341e-01, -4.877646426374731559e-01, 2.981576563494563459e-01],
[-5.536442465332774149e-02, -4.847077499026544167e-01, -8.729222059651777776e-01, 3.296726497672328171e-01],
[0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
], dtype=np.float32)

def main(args=None):
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")
    color_image, depth_image = camera.get_image()

    center_x = 623
    center_y = 253 
    cam_z = depth_image[center_y, center_x]
    yaw = 0.0


    cam_z = camera.get_valid_depth(
        depth_image,
        center_x,
        center_y,
        radius=3,
    )

    if cam_z is None:
        print("Depth 측정 실패")
        return

    cam_x = np.multiply(
        center_x - camera._color_intrinsics_mat[0][2],
        cam_z / camera._color_intrinsics_mat[0][0],
    )
    cam_y = np.multiply(
        center_y - camera._color_intrinsics_mat[1][2],
        cam_z / camera._color_intrinsics_mat[1][1],
    )

    cam_coordinate = [cam_x, cam_y, cam_z, yaw]
    print(f"cam_coord\n{cam_coordinate}")

    world_coordinate = extrinsic_matrix[0:3, 0:3] @ cam_coordinate[0:3]
    world_coordinate += extrinsic_matrix[0:3, 3:].flatten()

    print(f"world_coord\n{world_coordinate}")

if __name__ == "__main__":
    main()