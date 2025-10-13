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


class RerunWrapperWindowMapComparison(RerunWrapperWindow):
    def __init__(self, enable: bool, robot_0_name: str, robot_1_name: str):
        super().__init__(enable)
        self.robot_0_name = robot_0_name
        self.robot_1_name = robot_1_name
        assert robot_0_name != robot_1_name, "Robot names cannot match!"
        self.curr_robot = None

        self.robot_0_H_range: tuple = (0, 0.05)
        self.robot_1_H_range: tuple = (0.5, 0.55)

        self.node_statuses_to_show = [GraphNode.SegmentStatus.NURSERY,
                                      GraphNode.SegmentStatus.SEGMENT,
                                      GraphNode.SegmentStatus.INACTIVE,
                                      GraphNode.SegmentStatus.GRAVEYARD]

    # ===================== Methods to Override =====================
    def _get_blueprint_part(self) -> rrb.BlueprintPart:
        # Create the views
        graph_0_view = rrb.GraphView(name="Graph", origin=f'/graph/{self.robot_0_name}')
        graph_1_view = rrb.GraphView(name="Graph", origin=f'/graph/{self.robot_1_name}')
        world_view = rrb.Spatial3DView(name="World", origin=f'/world')

        # Create the tab
        graphs_vert = rrb.Vertical(graph_0_view, graph_1_view)
        return rrb.Horizontal(world_view, graphs_vert, name=f"Comparison {self.robot_0_name}-{self.robot_1_name}")
    
    def _get_curr_robot_name(self) -> str:
        return self.curr_robot
    
    # ===================== Robot Assignment =====================
    def set_curr_robot(self, name: str) -> None:
        if name == self.robot_0_name or name == self.robot_1_name:
            self.curr_robot = name
        else:
            raise ValueError(f"Provided name {name} does not match one of our two robots ({self.robot_0_name},{self.robot_1_name})")
    
    # ===================== Data Loggers =====================
    def update_graph(self, root_node: GraphNode):
        id_to_color_mapping = self._assign_colors_graph({}, root_node, self.get_hsv_space_for_curr_robot())
        self._update_graph_general(root_node, id_to_color_mapping, self.node_statuses_to_show)
    
    def update_segments(self, segments: list[Segment]):
        if not self.enable: return
        self._update_frame_tick()

        # Assign colors to segments
        seg_colors = self._assign_colors_segments(segments, self.get_hsv_space_for_curr_robot())

        # Extract Point Data
        seg_ids: list[int] = []
        colors_rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        points: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        point_to_node_ids: np.ndarray = np.zeros((0), dtype=np.uint32)
        point_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        for j, seg in enumerate(segments):
            seg_ids.append(seg.id)
            colors_rgb = np.concatenate((colors_rgb, seg_colors[j]), dtype=np.uint8)
            points = np.concatenate((points, seg.points), dtype=float)
            point_to_node_ids = np.concatenate((point_to_node_ids, np.full((seg.points.shape[0]), seg.id, dtype=np.uint32)))
            point_colors = np.concatenate((point_colors, np.full((seg.points.shape[0], 3), seg_colors[j])))

        # Extract Bounding Boxes
        box_centers, box_half_sizes = [], []
        box_quats = []
        box_colors = []
        box_ids = []
        for j, seg in enumerate(segments):
            _ = seg.volume # Will cause OBB to be created
            self.extract_data_from_obb(seg._obb, box_centers, box_half_sizes, box_quats)
            box_colors.append(seg_colors[j])
            box_ids.append(seg.id)

        # Send the data to Rerun
        rr.log(f"/world/{self._get_curr_robot_name()}/points", rr.Points3D(positions=points, colors=point_colors))
        rr.log(f"/world/{self._get_curr_robot_name()}/boxes", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0.01, fill_mode="line",
                            labels=None))
        rr.log(f"/world/{self._get_curr_robot_name()}/labels", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_ids))
        
    # ===================== Color Assignment =====================
    def get_hsv_space_for_curr_robot(self) -> HSVSpace:
        if self._get_curr_robot_name() == self.robot_0_name: H_range = self.robot_0_H_range
        else: H_range = self.robot_1_H_range
        return HSVSpace(H_range, (0.6, 1.0), (0.6, 1.0))
        
    def _assign_colors_segments(self, segments: list[Segment], hsv_space: HSVSpace) -> list[tuple[float, float, float]]:
        num_segments = len(segments)
        
        # Split the HSV space along the hue axis
        hue_subspaces = hsv_space.split(num_segments, axis=0)

        seg_colors = []
        for i, seg in enumerate(segments):
            # Deterministic RNG seeded from seg.id
            seed_bytes = hashlib.sha256(str(seg.id).encode()).digest()
            seed_int = int.from_bytes(seed_bytes[:8], "big")
            rng = random.Random(seed_int)

            # Pick hue in the segment's hue range
            h_min, h_max = hue_subspaces[i].h_range
            h = rng.uniform(h_min, h_max) % 1.0

            # Pick saturation and value within hsv_space ranges
            s_min, s_max = hsv_space.s_range
            v_min, v_max = hsv_space.v_range
            s = rng.uniform(s_min, s_max)
            v = rng.uniform(v_min, v_max)

            seg_colors.append(hsvF_to_rgb255(np.array([h, s, v], dtype=float)))

        return seg_colors