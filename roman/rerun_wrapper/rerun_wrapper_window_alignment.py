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
from roman.object.segment import Segment, SegmentMinimalData
from roman.params.fastsam_params import FastSAMParams
from roman.scene_graph.word_net_wrapper import WordListWrapper
from scipy.spatial.transform import Rotation as R
import trimesh


class RerunWrapperWindowAlignment(RerunWrapperWindow):
    def __init__(self, enable: bool, robot_0_name: str, robot_1_name: str):
        super().__init__(enable)
        self.robot_0_name = robot_0_name + "[0]"
        self.robot_1_name = robot_1_name + "[1]"
        self.curr_robot_index = None
        self.same_name: bool = robot_0_name == robot_1_name

        self.robot_0_H_range: tuple = (0, 0.05)
        self.robot_1_H_range: tuple = (0.5, 0.55)

    # ===================== Methods to Override =====================
    def _get_blueprint_part(self) -> rrb.BlueprintPart:
        return rrb.Horizontal(rrb.Spatial3DView(name="World", origin=f'/world/{self.robot_0_name}-{self.robot_1_name}'), name=f"Alignment {self.robot_0_name}-{self.robot_1_name}")
    
    def _get_curr_robot_name(self) -> str:
        if self.curr_robot_index == 0:
            return self.robot_0_name
        else:
            return self.robot_1_name
    
    # ===================== Timeline Handlers =====================
    def _update_frame_tick(self) -> None:
        rr.set_time(f"robot_{self.robot_0_name}-{self.robot_1_name}_update_frame_tick", sequence=self.update_frame)
        self.update_frame += 1

    def update_curr_time(self, curr_time: float) -> None:
        rr.set_time(f"robot_{self.robot_0_name}-{self.robot_1_name}_camera_frame_time", timestamp=curr_time)
    
    # ===================== Robot Assignment =====================
    def set_curr_robot_index(self, index: int) -> None:
        if index == 0 or index == 1:
            self.curr_robot_index = index
        else: raise ValueError(f"Provided index {index} not 0 or 1")
    
    # ===================== Data Loggers =====================
    def _update_segments(self, segments: list[Segment], seg_colors: dict | None = None, disable_tick: bool = False):
        if not self.enable: return
        if not disable_tick: self._update_frame_tick()

        # Assign colors to segments
        if seg_colors is None:
            seg_colors = self._assign_colors_segments(segments, self.get_hsv_space())

        # Extract Point Data
        seg_ids: list[int] = []
        colors_rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        points: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        point_to_node_ids: np.ndarray = np.zeros((0), dtype=np.uint32)
        point_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        for j, seg in enumerate(segments):
            seg_ids.append(seg.id)
            colors_rgb = np.concatenate((colors_rgb, [seg_colors[j]]), dtype=np.uint8)
            points = np.concatenate((points, seg.points), dtype=float)
            point_to_node_ids = np.concatenate((point_to_node_ids, np.full((seg.points.shape[0]), seg.id, dtype=np.uint32)))
            point_colors = np.concatenate((point_colors, np.full((seg.points.shape[0], 3), [seg_colors[j]])))

        # Extract Bounding Boxes
        box_centers, box_half_sizes = [], []
        box_quats = []
        box_colors = []
        box_ids = []
        for j, seg in enumerate(segments):
            _ = seg.volume # Will cause OBB to be created
            obb = seg._obb
            if obb is None: continue
            self.extract_data_from_obb(seg._obb, box_centers, box_half_sizes, box_quats)
            box_colors.append([seg_colors[j]])
            box_ids.append(seg.id)

        # Send the data to Rerun
        rr.log(f"/world/{self.robot_0_name}-{self.robot_1_name}/{self._get_curr_robot_name()}/points", rr.Points3D(positions=points, colors=point_colors))
        rr.log(f"/world/{self.robot_0_name}-{self.robot_1_name}/{self._get_curr_robot_name()}/boxes", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0.01, fill_mode="line",
                            labels=None))
        rr.log(f"/world/{self.robot_0_name}-{self.robot_1_name}/{self._get_curr_robot_name()}/labels", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_ids))
        
    def update_associations(self, i_segs: list[Segment], j_segs: list[Segment], associations: np.ndarray) -> None:
        if not self.enable: return
        self._update_frame_tick()

        # Update segments with a green color if they are associated
        i_seg_colors = self._assign_colors_segments(i_segs, self.get_hsv_space(0))
        j_seg_colors = self._assign_colors_segments(j_segs, self.get_hsv_space(1))

        for idx_i, idx_j in associations:
            i_seg_colors[idx_i] = np.array([0, 255, 0], dtype=np.uint8)
            j_seg_colors[idx_j] = np.array([0, 255, 0], dtype=np.uint8)

        self.set_curr_robot_index(0)
        self._update_segments(i_segs, i_seg_colors, True)
        self.set_curr_robot_index(1)
        self._update_segments(j_segs, j_seg_colors, True)

        # Draw lines between associated segments
        if len(associations) == 0: 
            rr.log(f"/world/{self.robot_0_name}-{self.robot_1_name}/associations",
                    rr.LineStrips3D([],colors=[], radii=np.full((0,), 0.02)))
            return
        
        i_centroids = np.array([seg.center for seg in i_segs])
        j_centroids = np.array([seg.center for seg in j_segs])

        lines = []
        colors = []
        for idx_i, idx_j in associations:
            p1 = i_centroids[idx_i]
            p2 = j_centroids[idx_j]
            lines.append(np.stack([p1, p2], axis=0))
            colors.append([0, 255, 0])
        lines = np.stack(lines)
        colors = np.array(colors, dtype=np.uint8)

        rr.log(f"/world/{self.robot_0_name}-{self.robot_1_name}/associations",
            rr.LineStrips3D(lines,colors=colors, radii=np.full((len(lines),), 0.02)))

    # ===================== Color Assignment =====================
    def get_hsv_space(self, robot_index: int | None = None) -> HSVSpace:
        if robot_index is None:
            if self._get_curr_robot_name() == self.robot_0_name: H_range = self.robot_0_H_range
            else: H_range = self.robot_1_H_range
        else:
            assert robot_index == 1 or robot_index == 0, "Robot Index must be 0 or 1!"
            if robot_index == 0: H_range = self.robot_0_H_range
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