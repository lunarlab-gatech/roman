from __future__ import annotations

from .data_params import DataParams
from .fastsam_params import FastSAMParams
from .mapper_params import MapperParams
from .offline_rpgo_params import OfflineRPGOParams
from pathlib import Path
from pydantic import BaseModel
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
    fastsam_params: FastSAMParams
    graph_node_params: GraphNodeParams
    mapper_params: MapperParams
    offline_rpgo_params: OfflineRPGOParams
    scene_graph_3D_params: SceneGraph3DParams
    submap_align_params: SubmapAlignParams
    gt_file: Path | None
    num_req_assoc: int

    @classmethod
    def from_param_dir(cls, path: str) -> SystemParams:
        params_path = Path(path)

        data_params = DataParams.from_yaml(params_path / "data.yaml")
        fastsam_params = FastSAMParams.from_yaml(params_path / "fastsam.yaml")
        graph_node_params = GraphNodeParams.from_yaml(params_path / "graph_node.yaml")
        mapper_params = MapperParams.from_yaml(params_path / "mapper.yaml")
        offline_rpgo_params = OfflineRPGOParams.from_yaml(params_path / "offline_rpgo.yaml") 
        scene_graph_3D_params = SceneGraph3DParams.from_yaml(params_path / "scene_graph_3D.yaml")
        submap_align_params = SubmapAlignParams.from_yaml(params_path / "submap_align.yaml")
        
        gt_file_path = params_path / "gt_pose.yaml"
        gt_file = None
        if gt_file_path.exists(): gt_file = gt_file_path

        with open(params_path / "system_params.yaml") as f:
            data = yaml.safe_load(f)
        num_req_assoc = data['num_req_assoc']

        return cls(data_params=data_params, 
                   fastsam_params=fastsam_params, 
                   graph_node_params=graph_node_params, 
                   mapper_params=mapper_params,
                   offline_rpgo_params=offline_rpgo_params, 
                   scene_graph_3D_params=scene_graph_3D_params, 
                   submap_align_params=submap_align_params, 
                   gt_file=gt_file, 
                   num_req_assoc=num_req_assoc)


