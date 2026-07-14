import pyrealsense2 as rs
import numpy as np

class RealSenseD435(object):
    def __init__(
        self,
        color_resolution,
        depth_mode,
    ):
        self._color_resolution = color_resolution
        self._depth_mode = depth_mode
        self._fps = 30

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
            [0, 0, 1],
        ])

        depth_stream = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self._depth_intrinsics = depth_stream.get_intrinsics()
        _depth_intrinsics_mat = np.array([
            [self._depth_intrinsics.fx, 0, self._depth_intrinsics.ppx],
            [0, self._depth_intrinsics.fy, self._depth_intrinsics.ppy],
            [0, 0, 1],
        ])

        return _color_intrinsics_mat, _depth_intrinsics_mat

    def get_image(self):
        # update
        for _ in range(30):
            frames = self._pipeline.wait_for_frames()
        while True:
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue
            break

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image / 1000

    # update
    def get_valid_depth(self, depth_image, x, y, radius=3):
        h, w = depth_image.shape

        x1 = max(0, x - radius) 
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)

        region = depth_image[y1:y2, x1:x2]
        valid_depths = region[region > 0]

        if valid_depths.size == 0:
            return None

        return float(np.median(valid_depths))