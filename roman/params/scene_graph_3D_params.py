from __future__ import annotations

from pydantic import BaseModel
import yaml

class SceneGraph3DParams(BaseModel):
    """
    Params for 3D Scene Graph association, merging, and relationship inference.
    """

    # ===== Node Association =====
    min_iou_3d: float
    use_convex_hull_for_iou: bool
    voxel_size_for_voxel_grid_iou: float

    # ===== Node Status =====
    max_t_no_sightings: float 

    # ===== Association Merges =====
    min_iou_2d_for_merging: float

    # ===== Parameters to mimic ROMAN functionality =====
    downsample_and_remove_outliers_after_hungarian_for_new_nodes: bool
    enable_dbscan_on_node_inactivation: bool
    update_voxel_grid_on_inactivation_dbscan: bool

    @classmethod
    def from_yaml(cls, path: str) -> SceneGraph3DParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)

class GraphNodeParams(BaseModel):

    # ===== Convex Hull ===== 
    require_valid_convex_hull: bool
    use_convex_hull_for_volume: bool

    @classmethod
    def from_yaml(cls, path: str) -> GraphNodeParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        new_params = cls(**raw_cfg)
        if new_params.use_convex_hull_for_volume and not new_params.require_valid_convex_hull:
            raise ValueError("'use_convex_hull_for_volume' is True but `require_valid_convex_hull` is False.")

        return new_params

    @classmethod
    def get_default_for_test_cases(cls) -> GraphNodeParams:
        return cls(
            require_valid_convex_hull=False,
            use_convex_hull_for_volume=False)