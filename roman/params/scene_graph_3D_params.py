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

    # ===== Initialization ===== 
    dbscan_and_remove_outliers_on_node_creation: bool
    downsample_on_node_creation: bool
    check_minimum_node_size_on_node_creation: bool

    # ===== Convex Hull ===== 
    require_valid_convex_hull: bool
    use_convex_hull_for_volume: bool
    convex_hull_outward_offset: float

    # ===== Voxel Downsampling ===== 
    enable_variable_voxel_size: bool
    voxel_size_not_variable: float
    voxel_size_variable_ratio_to_length: float

    # ===== DBSCAN Parameters ===== 
    dbscan_min_points: int
    enable_roman_dbscan: bool
    enable_variable_epsilon: bool
    epsilon_not_variable: float
    epsilon_variable_ratio_to_length: float
    min_cluster_percentage: float

    # ===== Statistical Outlier Removal Parameters =====
    enable_remove_statistical_outliers: bool
    stat_out_num_neighbors: int
    stat_out_std_ratio: float

    # ===== Semantic Descriptor =====
    calculate_descriptor_incrementally: bool
    use_weighted_average_for_descriptor: bool
    ignore_descriptors_from_observation: bool

    # ===== Data inheritance from child nodes =====
    parent_node_inherits_data_from_children: bool
    parent_node_inherits_descriptors_from_children: bool

    # ===== Parameters to mimic ROMAN functionality =====
    merge_with_node_use_first_seen_time_from_self: bool

    @classmethod
    def from_yaml(cls, path: str) -> GraphNodeParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        new_params = cls(**raw_cfg)
        if new_params.calculate_descriptor_incrementally and new_params.use_weighted_average_for_descriptor:
            raise ValueError("Only one of 'calculate_descriptor_incrementally' or "
                "'use_weighted_average_for_descriptor' can be True.")
        if new_params.calculate_descriptor_incrementally and new_params.parent_node_inherits_data_from_children:
            raise ValueError("Only one of 'calculate_descriptor_incrementally' or "
                "'parent_node_includes_child_node_for_data' can be True.")
        if new_params.use_convex_hull_for_volume and not new_params.require_valid_convex_hull:
            raise ValueError("'use_convex_hull_for_volume' is True but `require_valid_convex_hull` is False.")

        return new_params