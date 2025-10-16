from __future__ import annotations

import copy
from .color_utils import hsvF_to_rgb255, rgb255_to_bgr255, HSVSpace
import cv2
from ..scene_graph.graph_node import GraphNode
import hashlib
from ..logger import logger
import numpy as np
from open3d.geometry import OrientedBoundingBox
from ..params.data_params import ImgDataParams
import random
import rerun as rr
import rerun.blueprint as rrb
from .rerun_wrapper_window import RerunWrapperWindow
from roman.object.segment import Segment
from roman.params.fastsam_params import FastSAMParams
from roman.scene_graph.word_net_wrapper import WordListWrapper
from scipy.spatial.transform import Rotation as R
import trimesh


class RerunWrapperWindowMap(RerunWrapperWindow):
    def __init__(self, enable: bool, robot_name: str, fastsam_params: FastSAMParams):
        super().__init__(enable)
        self.robot_name: str = robot_name
        self.fastsam_params: FastSAMParams = fastsam_params
        self.id_to_color_mapping: dict = dict()
        self.node_statuses_to_show: list = [GraphNode.SegmentStatus.SEGMENT,
                                            GraphNode.SegmentStatus.INACTIVE]

    # ===================== Methods to Override =====================
    def _get_blueprint_part(self) -> rrb.BlueprintPart:
        # Create the views
        graph_view = rrb.GraphView(name="Graph", origin=f'/graph/{self.robot_name}')
        world_view = rrb.Spatial3DView(name="World", origin=f'/world/{self.robot_name}')
        log_view = rrb.TextLogView(name="Text Logs", origin=f"/logs")
        image_view = rrb.Spatial2DView(name="Image", origin=f'/world/{self.robot_name}/camera/image')
        depth_view = rrb.Spatial2DView(name="Depth", origin=f'/world/{self.robot_name}/camera/depth')
        seg_view = rrb.Spatial2DView(name="Segmentation Mask", origin=f'/world/{self.robot_name}/camera/segmentation')

        # Create the tab
        world_graph_log_horiz = rrb.Horizontal(graph_view, world_view, log_view)
        img_depth_seg_horiz = rrb.Horizontal(image_view, depth_view, seg_view)
        tab_name: str = f"Map {self.robot_name}"
        return rrb.Vertical(world_graph_log_horiz, img_depth_seg_horiz, name=tab_name)
    
    def _get_curr_robot_name(self) -> str:
        return self.robot_name
    
    # ===================== Data Loggers =====================
    def update_graph(self, root_node: GraphNode):
        self.id_to_color_mapping = self._assign_colors_graph(self.id_to_color_mapping, root_node, HSVSpace((0.0, 1.0), (0.2, 1.0), (0.2, 1.0)))
        self._update_graph_general(root_node, self.id_to_color_mapping, self.node_statuses_to_show)
    
    def update_img(self, img: np.ndarray) -> None:
        if not self.enable: return
        self._update_frame_tick()
        rr.log(f"/world/{self._get_curr_robot_name()}/camera/image", rr.Image(img, color_model="BGR"))

    def update_depth_img(self, depth_img: np.ndarray) -> None:
        if not self.enable: return
        self._update_frame_tick()

        # Scale image and threshold values past max_detph
        depth_img_vis = copy.deepcopy(depth_img).astype(np.float32)
        depth_img_vis /= self.fastsam_params.depth_scale
        depth_img_vis[depth_img_vis > self.fastsam_params.max_depth] = self.fastsam_params.max_depth
        rr.log(f"/world/{self._get_curr_robot_name()}/camera/depth", rr.DepthImage(depth_img_vis))

    def update_camera_pose(self, camera_pose: np.ndarray) -> None:
        if not self.enable: return
        self._update_frame_tick()

        rot = R.from_matrix(camera_pose[:3,:3])
        rr.log(f"/world/{self._get_curr_robot_name()}/camera", rr.Transform3D(translation=camera_pose[:3,3],
            quaternion=rot.as_quat(), relation=rr.TransformRelation.ParentFromChild, clear=False), strict=True)
        
    def update_camera_intrinsics(self, img_data_params: ImgDataParams) -> None:
        if not self.enable: return
        self._update_frame_tick()

        rr.log(f"/world/{self._get_curr_robot_name()}/camera/image", 
                rr.Pinhole(resolution=[img_data_params.width, img_data_params.height],
                            focal_length=[img_data_params.K[0], img_data_params.K[4]],
                            principal_point=[img_data_params.K[2], img_data_params.K[5]],
                            image_plane_distance=1.0))
        
    def update_seg_img(self, seg_img: np.ndarray, img: np.ndarray, associations: list[tuple[int, int]], node_to_obs_mapping: dict) -> None:
        if not self.enable: return
        self._update_frame_tick()

        # Create mapping from observation to color
        num_obs: int = seg_img.shape[0]
        colormap: np.ndarray = np.full((num_obs+1, 3), 128, dtype=np.uint8)
        colormap[0] = [0, 0, 0]
        for pair in associations:
            colormap[node_to_obs_mapping[pair[0]]+1] = rgb255_to_bgr255(self.id_to_color_mapping[pair[1]])
        
        # Calculate image of masks with bools instead of per number
        bool_img = np.zeros_like(seg_img)
        bool_img[seg_img > 0] = 1

        # Get array representing number of observations in each pixel
        obs_per_pixel = np.sum(bool_img, axis=0) + np.full(bool_img.shape[1:], 0.00001)
        obs_per_pixel = obs_per_pixel[:,:,np.newaxis]
        obs_per_pixel = np.repeat(obs_per_pixel, 3, axis=2)

        # Calculate the color as the average of the colors divided by the number of observations
        color_mask = np.divide(np.sum(colormap[seg_img], axis=0), obs_per_pixel).astype(np.uint8)
        
        # Overlay color onto the normal image
        overlay = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
        rr.log(f"/world/{self._get_curr_robot_name()}/camera/segmentation", rr.Image(overlay, color_model="BGR"))
