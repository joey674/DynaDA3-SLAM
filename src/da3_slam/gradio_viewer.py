import trimesh
import numpy as np
from pathlib import Path


class TrimeshViewer:
    def __init__(self):
        self.scene = trimesh.Scene()
        self.frame_size = 0.1  # Scale of camera frame axes

    def add_camera_pose(self, cam2world: np.ndarray):
        """Add camera coordinate frame (as axis) to scene."""
        T = cam2world

        # Trimesh's built-in coordinate frame
        cam_frame = trimesh.creation.axis(origin_size=0.01, axis_length=self.frame_size)
        cam_frame.apply_transform(T)

        self.scene.add_geometry(cam_frame)

    def add_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        dynamic_mask: np.ndarray = None,
        transparent_dynamic: bool = False,
    ):
        """
        points: (N, 3)
        colors: (N, 3) uint8 or float
        """
        if colors is not None and colors.size > 0 and transparent_dynamic and dynamic_mask is not None:
            if colors.max() <= 1.0:
                colors_uint8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                colors_uint8 = np.clip(colors, 0.0, 255.0).astype(np.uint8)

            alpha = np.full((colors_uint8.shape[0], 1), 255, dtype=np.uint8)
            alpha[np.asarray(dynamic_mask).reshape(-1).astype(bool)] = 0
            colors = np.concatenate([colors_uint8, alpha], axis=1)

        point_cloud = trimesh.points.PointCloud(points, colors=colors)
        self.scene.add_geometry(point_cloud)

    def export(self, out_path="output.glb") -> str:
        """Export the scene as a GLB file (for Hugging Face display)."""
        out_path = Path(out_path)
        self.scene.export(out_path)
        return str(out_path)
