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
        self.node_statuses_to_show: list = [GraphNode.SegmentStatus.NURSERY,
                                            GraphNode.SegmentStatus.SEGMENT,
                                            GraphNode.SegmentStatus.INACTIVE, 
                                            GraphNode.SegmentStatus.GRAVEYARD]

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
        self.id_to_color_mapping = self._assign_colors_graph(self.id_to_color_mapping, root_node, HSVSpace((0.0, 1.0), (0.2, 1.0), (0.2, 1.0)))
        self._update_graph_general(root_node, self.id_to_color_mapping, self.node_statuses_to_show)
