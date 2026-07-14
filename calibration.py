import rclpy
import DR_init
import sys
import numpy as np
import time
import math
import cv2

from robot import Robot
from scipy import optimize
from camera import RealSenseD435
from config import ROBOT_ID, ROBOT_MODEL


# workspace_limits = np.asarray([[-650, -450], [25, 100], [-100, 0]]) # [[X min, X max], [Y min, Y max], [Z min, Z max]] & unit: mm
workspace_limits = np.asarray([[-650, -450], [50, 100], [-100, 0]]) # [[X min, X max], [Y min, Y max], [Z min, Z max]] & unit: mm

joint_start = [-180.00, 60.00, 90.00, 90.00, -90.00, -150.00] # unit: mm
tool_orientation = [85, 90, -120]

calib_grid_step = 75

measured_pts = np.array([])
observed_pts = np.array([])
observed_pix = np.array([])
world2camera = np.eye(4)
camera = None

# Camera calibaration
checkerboard_size = (3, 3)
checkerboard_offset_from_tool = [0, -30, 0]

def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1)) # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB) # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t

def get_rigid_transform_error(z_scale, camera_in, measured_pts_in, observed_pts_in, observed_pix_in, world2camera_container):
    # Apply z offset and compute new observed points using camera intrinsics
    observed_z = observed_pts_in[:,2:] * z_scale
    observed_x = np.multiply(observed_pix_in[:,[0]]-camera_in._color_intrinsics_mat[0][2],observed_z/camera_in._color_intrinsics_mat[0][0])
    observed_y = np.multiply(observed_pix_in[:,[1]]-camera_in._color_intrinsics_mat[1][2],observed_z/camera_in._color_intrinsics_mat[1][1])
    new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

    # Estimate rigid transform between measured points and new observed points
    R, t = get_rigid_transform(np.asarray(measured_pts_in), np.asarray(new_observed_pts))
    t.shape = (3, 1)
    world2camera_container[:] = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)

    # Compute rigid transform error
    registered_pts = np.dot(R,np.transpose(measured_pts_in)) + np.tile(t,(1,measured_pts_in.shape[0]))
    error = np.transpose(registered_pts) - new_observed_pts
    error = np.sum(np.multiply(error,error))
    rmse = np.sqrt(error/measured_pts_in.shape[0])
    return rmse

def main(args=None):
    rclpy.init(args=args)

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node('example_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movej, movel, posj, set_robot_mode, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    camera = RealSenseD435(color_resolution=720, depth_mode="720P")
    robot = Robot(node)

    measured_pts = np.array([])
    observed_pts = np.array([])
    observed_pix = np.array([])
    world2camera = np.eye(4)

    # Construct 3D calibration grid across workspace
    gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], math.ceil((workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step) + 1)
    gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], math.ceil((workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step) + 1)
    gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], math.ceil((workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step) + 1)
    calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
    num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]

    calib_grid_x.shape = (num_calib_grid_pts, 1)
    calib_grid_y.shape = (num_calib_grid_pts, 1)
    calib_grid_z.shape = (num_calib_grid_pts, 1)
    calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

    _measured_pts = []
    _observed_pts = []
    _observed_pix = []

    # Move robot to home pose
    print(f"\nMove robot to start pose...\n")
    robot.move_j(joint_start)
    

    # Move robot to each calibration point in workspace
    print(f"Collecting data...\n")
    for calib_pt_idx in range(num_calib_grid_pts):
        tool_position = calib_grid_pts[calib_pt_idx, :]
        target_pose = list(tool_position) + tool_orientation
        print(f"{calib_pt_idx}'s target pose : {target_pose}")
        movel(target_pose, 20, 20)
        time.sleep(0.5)

        # Find checkerboard center
        color_img, depth_img = camera.get_image()
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
        checkerboard_found, corners = cv2.findChessboardCorners(gray_img, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
        if checkerboard_found:
            print(f"{calib_pt_idx}'s checkerboard found!")
            corners_refined = cv2.cornerSubPix(gray_img, corners, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Get observed checkerboard center 3D point in camera space
            checkerboard_pix = np.round(corners_refined[4,0,:]).astype(int)
            checkerboard_z = depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
            print(f"Checkerboard_pix: {checkerboard_pix}, Checkerboard_z: {checkerboard_z}")
            
            if checkerboard_z == 0:
                print(f"Warning: Checkerboard depth is 0. (Skipping this point)")
                continue

            checkerboard_x = np.multiply(checkerboard_pix[0]-camera._color_intrinsics_mat[0][2],checkerboard_z/camera._color_intrinsics_mat[0][0])
            checkerboard_y = np.multiply(checkerboard_pix[1]-camera._color_intrinsics_mat[1][2],checkerboard_z/camera._color_intrinsics_mat[1][1])

            # Save calibration point and observed checkerboard center (for Doosan)
            _observed_pts.append([checkerboard_x,checkerboard_y,checkerboard_z])
            tool_position = (tool_position + checkerboard_offset_from_tool)/1000

            _measured_pts.append(tool_position)
            _observed_pix.append(checkerboard_pix)

            # Draw and display the corners
            vis = cv2.drawChessboardCorners(color_img, checkerboard_size, corners_refined, checkerboard_found)
            cv2.imwrite(f'/home/jooyeon/ros2_ws/src/doosan-robot2/images/{len(_measured_pts):03}.png', vis) # need to edit (check your path)
            cv2.imshow('Calibration', vis)
            cv2.waitKey(1000)
            print(f"Draw chessboard corners finish!")

        else:
            print(f"{calib_pt_idx}'s checkerboard not found..................")

        if len(_measured_pts) > 0:
            measured_pts = np.asarray(_measured_pts)
            observed_pts = np.asarray(_observed_pts)
            observed_pix = np.asarray(_observed_pix)
            world2camera = np.eye(4)

            # Optimize z scale w.r.t. rigid transform error
            print(f"Calibrating...")
            z_scale_init = 1
            
            optim_result = optimize.minimize(
                get_rigid_transform_error, 
                np.asarray(z_scale_init), 
                args=(camera, measured_pts, observed_pts, observed_pix, world2camera), 
                method='Nelder-Mead'
            )
            camera_depth_offset = optim_result.x

            # Save camera optimized offset and camera pose
            print(f"Saving...")
            np.savetxt('/home/jooyeon/ros2_ws/src/doosan-robot2/images/camera_depth_scale.txt', camera_depth_offset, delimiter=' ') # need to edit (check your path)
            get_rigid_transform_error(camera_depth_offset, camera, measured_pts, observed_pts, observed_pix, world2camera)
            camera_pose = np.linalg.inv(world2camera)
            np.savetxt('/home/jooyeon/ros2_ws/src/doosan-robot2/images/camera_pose.txt', camera_pose, delimiter=' ') # need to edit (check your path)

            print(f"Done!\n")
        else:
            print(f"No valid data collected yet. Skipping calibration optimization for this step.\n")

        time.sleep(1)

    print("Example complete")
    rclpy.shutdown()

if __name__ == '__main__':
    main()