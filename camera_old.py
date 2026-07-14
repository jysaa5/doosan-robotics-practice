import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseD435(object):
    def __init__(
        self,
        color_resolution,
        depth_mode,
        extrinsic_matrix,
    ):
        self._color_resolution = color_resolution
        self._depth_mode = depth_mode
        self._fps = 30
        self.extrinsic_matrix = extrinsic_matrix

        self._init_realsense()
        self._color_intrinsics_mat, self._depth_intrinsics_mat = self._init_intrinsics()

    def _init_realsense(self):
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        color_config = self._color_resolution
        if color_config == "OFF":
            color_w, color_h = 0, 0
        elif color_config == 480:
            color_w, color_h = 640, 480
        elif color_config == 540:
            color_w, color_h = 960, 540
        elif color_config == 720:
            color_w, color_h = 1280, 720
        elif color_config == 1080:
            color_w, color_h = 1920, 1080
        else:
            raise Exception("Unknown color resolution in config")

        if color_config != "OFF":
            self._config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, self._fps)

        depth_config = self._depth_mode
        if depth_config == "OFF":
            depth_w, depth_h = 0, 0
        elif depth_config == "480P":
            depth_w, depth_h = 640, 480
        elif depth_config == "720P":
            depth_w, depth_h = 1280, 720
        elif depth_config == "Z_HALF":
            depth_w, depth_h = 848, 480
        else:
            raise Exception("Unknown depth mode in config")

        if depth_config != "OFF":
            self._config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, self._fps)

        self._profile = self._pipeline.start(self._config)
        self._align = rs.align(rs.stream.color)

    def _init_intrinsics(self):
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._color_intrinsics = color_stream.get_intrinsics()
        
        _color_intrinsics_mat = np.array([
            [self._color_intrinsics.fx, 0, self._color_intrinsics.ppx],
            [0, self._color_intrinsics.fy, self._color_intrinsics.ppy],
            [0, 0, 1]
        ])

        depth_stream = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self._depth_intrinsics = depth_stream.get_intrinsics()
        
        _depth_intrinsics_mat = np.array([
            [self._depth_intrinsics.fx, 0, self._depth_intrinsics.ppx],
            [0, self._depth_intrinsics.fy, self._depth_intrinsics.ppy],
            [0, 0, 1]
        ])

        return _color_intrinsics_mat, _depth_intrinsics_mat

    def get_image(self):
        while True:
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue
            break

        color_image = np.asanyarray(color_frame.get_data())
        transformed_depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, transformed_depth_image/1000

    def get_point_cloud(self):
        color_image, depth_image = self.get_image()
        
        mask = np.where(depth_image > 0)
        x = mask[1]
        y = mask[0]
        
        normalized_x = (x.astype(np.float32) - self._color_intrinsics_mat[0,2])/self._color_intrinsics_mat[0,0]
        normalized_y = (y.astype(np.float32) - self._color_intrinsics_mat[1,2])/self._color_intrinsics_mat[1,1]
        world_x = normalized_x * depth_image[y, x]
        world_y = normalized_y * depth_image[y, x]
        world_z = depth_image[y, x]
        
        point_cloud = np.vstack((world_x, world_y, world_z)).T

        colors = color_image[y, x, :]

        return point_cloud, colors
    
    def get_heightmap(self, workspace_limits, heightmap_resolution):
        # Compute heightmap size
        heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)
        
        # Get 3D point cloud from RGB-D images
        surface_pts, color_pts = self.get_point_cloud()

        # Transform 3D point cloud from camera coordinates {c} to robot coordinates {w}
        cam_pose = self.extrinsic_matrix
        surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))
        
        # Sort surface points by z value
        sort_z_ind = np.argsort(surface_pts[:,2])
        surface_pts = surface_pts[sort_z_ind]
        color_pts = color_pts[sort_z_ind]

        # Filter out surface points outside heightmap boundaries
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1])
        surface_pts = surface_pts[heightmap_valid_ind]
        color_pts = color_pts[heightmap_valid_ind]
        
        # Create orthographic top-down-view RGB-D heightmaps
        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        depth_heightmap = np.zeros(heightmap_size)
        heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
        heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
        color_heightmap_r[heightmap_pix_x,heightmap_pix_y] = color_pts[:,[0]]
        color_heightmap_g[heightmap_pix_x,heightmap_pix_y] = color_pts[:,[1]]
        color_heightmap_b[heightmap_pix_x,heightmap_pix_y] = color_pts[:,[2]]
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        depth_heightmap[heightmap_pix_x,heightmap_pix_y] = surface_pts[:,2]
        depth_heightmap[depth_heightmap < 0] = 0

        return color_heightmap, depth_heightmap
