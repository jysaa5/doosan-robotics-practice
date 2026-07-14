import numpy as np
from camera import RealSenseD435

extrinsic_matrix = np.array([
    
    [-9.994466382660912585e-01, 3.148521209594088224e-03,3.311350287725785269e-02, -5.524691072389610325e-01],
    [-6.226084404233151945e-03, 9.602076644630809232e-01, -2.792176158113003348e-01, 2.309949336075064752e-01],
    [-3.267496184543744464e-02, -2.792692749311108114e-01, -9.596567193261614781e-01, 3.711441810516031281e-01],
    [0.0000000000000000000e+00, 0.0000000000000000000e+00, 0.0000000000000000000e+00,1.0000000000000000000e+00, ],
], dtype=np.float32)

def main(args=None):
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")
    color_image, depth_image = camera.get_image()

    center_x = 633 
    center_y = 427 
    cam_z = depth_image[center_y, center_x]
    yaw = 0.0

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