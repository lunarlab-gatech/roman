from __future__ import annotations

from .data_params import DataParams, PoseDataGTParams
from .fastsam_params import FastSAMParams
from .mapper_params import MapperParams
from .offline_rpgo_params import OfflineRPGOParams
import os
from pathlib import Path
from pydantic import BaseModel, model_validator
from robotdatapy.data.pose_data import PoseData
from roman.utils import expandvars_recursive
from .scene_graph_3D_params import SceneGraph3DParams, GraphNodeParams
from .submap_align_params import SubmapAlignParams
import yaml

class SystemParams(BaseModel):
    """
    Wrapper class that holds sub-classes for each of the sub-system parameters.
    
    Args:
        data_params (DataParams): Parameters for data loading.
        fastsam_params (FastSAMParams): Parameters for FastSAM & YOLO.
        graph_node_params (GraphNodeParams): Parameters for graph node point cloud processing.
        mapper_params (MapperParams): Parameters for ROMAN's mapper.
        offline_rpgo_params (OfflineRPGOParams): Parameters for Kimera-RPGO.
        scene_graph_3D_params (SceneGraph3DParams): Parameters for MeronomyGraph Mapper.
        submap_align_params (SubmapAlignParams): Parameters for Submap alignment via CLIPPER.
        gt_file (Path | None): Optional path to ground truth pose file.
        num_req_assoc (int): Number of required associations for merging nodes.
    """

    data_params: DataParams
    pose_data_gt_params: PoseDataGTParams
    fastsam_params: FastSAMParams
    graph_node_params: GraphNodeParams
    mapper_params: MapperParams
    offline_rpgo_params: OfflineRPGOParams
    scene_graph_3D_params: SceneGraph3DParams
    submap_align_params: SubmapAlignParams
    num_req_assoc: int
    use_scene_graph: bool
    use_roman_map_for_alignment: bool
    random_seed: int

    @classmethod
    def from_param_dir(cls, path: str) -> SystemParams:
        params_path = Path(path)

        data_params = DataParams.from_yaml(params_path / "data.yaml")
        pose_data_gt_params = PoseDataGTParams.from_yaml(params_path / "gt_pose.yaml")
        fastsam_params = FastSAMParams.from_yaml(params_path / "fastsam.yaml")
        graph_node_params = GraphNodeParams.from_yaml(params_path / "graph_node.yaml")
        mapper_params = MapperParams.from_yaml(params_path / "mapper.yaml")
        offline_rpgo_params = OfflineRPGOParams.from_yaml(params_path / "offline_rpgo.yaml") 
        scene_graph_3D_params = SceneGraph3DParams.from_yaml(params_path / "scene_graph_3D.yaml")
        submap_align_params = SubmapAlignParams.from_yaml(params_path / "submap_align.yaml")
        
        with open(params_path / "system_params.yaml") as f:
            data = yaml.safe_load(f)
        num_req_assoc = data['num_req_assoc']
        use_scene_graph = data['use_scene_graph']
        use_roman_map_for_alignment = data['use_roman_map_for_alignment']
        random_seed = data['random_seed']

        if not use_roman_map_for_alignment and not use_scene_graph:
            raise ValueError("Cannot set 'use_roman_map_for_alignment' to False when 'use_scene_graph' is also False.")
        
        if scene_graph_3D_params.use_convex_hull_for_iou and not graph_node_params.require_valid_convex_hull:
            raise ValueError("If Scene Graph uses Convex Hull IOU, then GraphNode must require Convex Hull.")

        return cls(data_params=data_params, 
                   pose_data_gt_params=pose_data_gt_params,
                   fastsam_params=fastsam_params, 
                   graph_node_params=graph_node_params, 
                   mapper_params=mapper_params,
                   offline_rpgo_params=offline_rpgo_params, 
                   scene_graph_3D_params=scene_graph_3D_params, 
                   submap_align_params=submap_align_params, 
                   num_req_assoc=num_req_assoc,
                   use_scene_graph=use_scene_graph,
                   use_roman_map_for_alignment=use_roman_map_for_alignment,
                   random_seed=random_seed)