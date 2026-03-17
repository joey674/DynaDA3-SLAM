import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from src.da3_slam.slam_utils import apply_point_visualization

class GraphMap:
    def __init__(self):
        self.submaps = dict()
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
    
    def get_largest_key(self):
        if len(self.submaps) == 0:
            return -1
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self):
        return self.get_submap(self.get_largest_key())
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        for submap_key in self.submaps.keys():
            if submap_key == current_submap_id:
                continue

            if ignore_last_submap and (submap_key == current_submap_id-1):
                continue

            else:
                submap = self.submaps[submap_key]
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for embedding in submap_embeddings:
                    score = torch.linalg.norm(embedding-query_vector)
                    scores.append(score.item())
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        
        return frames
    
    def update_submap_homographies(self, graph):
        for submap_key in self.submaps.keys():
            submap = self.submaps[submap_key]
            submap.set_reference_homography(graph.get_homography(submap_key).matrix())
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]

    def write_poses_to_file(self, file_name):
        with open(file_name, "w") as f:
            for submap in self.ordered_submaps_by_key():
                poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
                frame_ids = submap.get_frame_ids()
                assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
                for frame_id, pose in zip(frame_ids, poses):
                    x, y, z = pose[0:3, 3]
                    rotation_matrix = pose[0:3, 0:3]
                    quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                    output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def save_framewise_pointclouds(self, file_name):
        os.makedirs(file_name, exist_ok=True)
        for submap in self.ordered_submaps_by_key():
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(ignore_loop_closure_frames=True)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)
                
    def _collect_points_and_colors(self):
        pcd_all = []
        colors_all = []
        dynamic_masks_all = []
        low_conf_masks_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame(include_low_conf=True).reshape(-1, 3)
            colors = submap.get_points_colors(include_low_conf=True).reshape(-1, 3)
            dynamic_mask = submap.get_points_dynamic_mask(include_low_conf=True).reshape(-1)
            low_conf_mask = submap.get_points_low_conf_mask(include_low_conf=True).reshape(-1)
            pcd_all.append(pcd)
            colors_all.append(colors)
            dynamic_masks_all.append(dynamic_mask)
            low_conf_masks_all.append(low_conf_mask)

        if len(pcd_all) == 0:
            raise RuntimeError("No submaps available. Run reconstruction before exporting point cloud.")

        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        dynamic_masks_all = np.concatenate(dynamic_masks_all, axis=0).astype(bool)
        low_conf_masks_all = np.concatenate(low_conf_masks_all, axis=0).astype(bool)
        return pcd_all, colors_all, dynamic_masks_all, low_conf_masks_all

    def _collect_camera_poses(self):
        pose_list = []
        for submap in self.ordered_submaps_by_key():
            poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
            if poses is None or len(poses) == 0:
                continue
            pose_list.append(poses)

        if len(pose_list) == 0:
            return np.empty((0, 4, 4), dtype=np.float32)
        return np.concatenate(pose_list, axis=0).astype(np.float32)

    def _create_camera_frustum_wireframe(self, trimesh_module, marker_size=0.05):
        depth = marker_size
        half_w = marker_size * 0.45
        half_h = marker_size * 0.30

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [-half_w, -half_h, depth],
                [half_w, -half_h, depth],
                [half_w, half_h, depth],
                [-half_w, half_h, depth],
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (2, 3), (3, 4), (4, 1),
        ]
        line_radius = max(marker_size * 0.03, 1e-4)
        line_meshes = []
        for start_idx, end_idx in edges:
            segment = np.array([vertices[start_idx], vertices[end_idx]], dtype=np.float32)
            edge_mesh = trimesh_module.creation.cylinder(radius=line_radius, segment=segment, sections=10)
            edge_mesh.visual.face_colors = np.array([40, 170, 255, 255], dtype=np.uint8)
            line_meshes.append(edge_mesh)
        return trimesh_module.util.concatenate(line_meshes)

    def _export_glb(self, file_name, points, colors, camera_poses=None, camera_marker_size=0.05):
        import trimesh

        if colors.max() <= 1.0:
            colors_uint8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            colors_uint8 = np.clip(colors, 0.0, 255.0).astype(np.uint8)

        scene = trimesh.Scene()
        point_cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors_uint8)
        scene.add_geometry(point_cloud, geom_name="reconstructed_pointcloud")

        if camera_poses is not None and len(camera_poses) > 0:
            base_frustum = self._create_camera_frustum_wireframe(trimesh, marker_size=camera_marker_size)
            for idx, pose in enumerate(camera_poses):
                cam_frustum = base_frustum.copy()
                cam_frustum.apply_transform(pose)
                scene.add_geometry(cam_frustum, geom_name=f"camera_pose_{idx}")

        scene.export(file_name)

    def _export_open3d(self, file_name, points, colors):
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd_all.colors = o3d.utility.Vector3dVector(colors)
        success = o3d.io.write_point_cloud(file_name, pcd_all)
        if not success:
            raise RuntimeError(f"Failed to write point cloud to {file_name}.")
        return file_name


    def write_points_to_file(self, file_name, vis_uncertainty="red", vis_low_conf="transparent"):
        output_dir = os.path.dirname(file_name)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        points, colors, dynamic_mask, low_conf_mask = self._collect_points_and_colors()
        colors, transparent_mask = apply_point_visualization(
            colors,
            dynamic_mask=dynamic_mask,
            low_conf_mask=low_conf_mask,
            vis_uncertainty=vis_uncertainty,
            vis_low_conf=vis_low_conf,
        )
        ext = os.path.splitext(file_name)[1].lower()

        if ext == ".glb":
            if colors.max() <= 1.0:
                colors_uint8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                colors_uint8 = np.clip(colors, 0.0, 255.0).astype(np.uint8)
            alpha = np.full((colors_uint8.shape[0], 1), 255, dtype=np.uint8)
            alpha[transparent_mask] = 0
            colors_rgba = np.concatenate([colors_uint8, alpha], axis=1)
            camera_poses = self._collect_camera_poses()
            try:
                self._export_glb(file_name, points, colors_rgba, camera_poses=camera_poses)
                return file_name
            except ModuleNotFoundError as exc:
                fallback_file_name = os.path.splitext(file_name)[0] + ".ply"
                print(f"[Warning] {exc}. Falling back to {fallback_file_name}")
                return self._export_open3d(
                    fallback_file_name,
                    points[~transparent_mask],
                    colors[~transparent_mask],
                )

        if ext == ".npz":
            np.savez(
                file_name,
                pointcloud=points,
                colors=colors,
                dynamic_mask=dynamic_mask,
                low_conf_mask=low_conf_mask,
                transparent_mask=transparent_mask,
            )
            return file_name

        return self._export_open3d(file_name, points[~transparent_mask], colors[~transparent_mask])
