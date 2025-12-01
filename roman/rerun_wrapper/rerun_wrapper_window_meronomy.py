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
from roman.scene_graph.word_net_wrapper import WordListWrapper
from scipy.spatial.transform import Rotation as R

class RerunWrapperWindowMeronomy(RerunWrapperWindow):
    def __init__(self, enable: bool, robot_name: str):
        super().__init__(enable)
        self.robot_name: str = robot_name
        self.id_to_color_mapping: dict = dict()

    # ===================== Methods to Override =====================
    def _get_blueprint_part(self) -> rrb.BlueprintPart:
        # Create the views
        graph_view = rrb.GraphView(name="Graph", origin=f'/graph/{self.robot_name}')
        world_view = rrb.Spatial3DView(name="World", origin=f'/world/{self.robot_name}')
        log_view = rrb.TextLogView(name="Text Logs", origin=f"/logs")

        # Create the tab
        horiz_graph_world = rrb.Horizontal(graph_view, world_view)
        tab_name: str = f"MeronomyGraph {self.robot_name}"
        return rrb.Vertical(horiz_graph_world, log_view, name=tab_name)
    
    def _get_curr_robot_name(self) -> str:
        return self.robot_name
    
    # ===================== Data Loggers =====================
    def update_graph(self, root_node: GraphNode):
        self.id_to_color_mapping = self._assign_colors_meronomy(self.id_to_color_mapping, root_node)
        self._update_graph_general(root_node, self.id_to_color_mapping)

    # ===================== Color Assignment =====================
    def _assign_colors_meronomy(self, id_to_color_mapping: dict[int, np.ndarray], root_node: GraphNode) -> dict[int, np.ndarray]:
        """ Nodes identical to their segment counterparts stay yellow, while meronym altered nodes are purple """

        for node in root_node:
            if node.is_meronomy_created_or_altered:
                # Purple for meronomy created/altered nodes
                id_to_color_mapping[node.id] = np.array([128, 0, 128], dtype=np.uint8)
            elif node.is_descendent_of_meronomy_created_or_altered_node():
                # Cyan for descendents of meronomy created/altered nodes
                id_to_color_mapping[node.id] = np.array([0, 255, 255], dtype=np.uint8)
            else:
                # Yellow for original segments
                id_to_color_mapping[node.id] = np.array([255, 255, 0], dtype=np.uint8)
        return id_to_color_mapping
