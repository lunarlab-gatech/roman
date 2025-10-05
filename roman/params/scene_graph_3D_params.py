from __future__ import annotations

from pydantic import BaseModel
import yaml

class SceneGraph3DParams(BaseModel):
    """
    Params for 3D Scene Graph association, merging, and relationship inference.
    """

    # ======================= Visualization =======================
    enable_rerun_viz: bool

    # ======================= Node Association =======================
    use_convex_hull_for_iou: bool
    min_iou_for_association: float
    min_iou_2d_for_merging: float
    voxel_size_for_voxel_grid_iou: float

    downsample_and_remove_outliers_after_hungarian_for_new_nodes: bool

    # ======================= Semantic Merges =======================
    enable_synonym_merges: bool
    min_sem_con_for_association: float
    ratio_dist2length_threshold_nearby_node_semantic_merge: float

    # ======================= Resolve Overlapping Nodes =======================
    enable_resolve_overlapping_nodes: bool
    iou_threshold_overlapping_obj: float
    enc_threshold_overlapping_obj: float
    
    # ======================= Meronomy Relationship Inference =======================
    enable_meronomy_relationship_inference: bool
    ratio_dist2length_threshold_shared_holonym: float
    ratio_dist2length_threshold_holonym_meronym: float
    ratio_relationship_weight_2_total_weight: float

    # ======================= Node Retirement =======================
    max_t_active_for_node: float # seconds
    max_dist_active_for_node: float # meters
    run_dbscan_when_retiring_node: bool
    delete_nodes_only_seen_once: bool

    max_t_no_sightings: float # seconds

    @classmethod
    def from_yaml(cls, path: str) -> SceneGraph3DParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)

class GraphNodeParams(BaseModel):
    # If using wordnet for semantic merging, number of words to simulatneously consider ourselves as
    num_words_to_consider_ourselves: int

    require_valid_convex_hull: bool
    check_minimum_node_size_during_creation: bool

    # ===== Voxel Downsampling ===== 
    downsample_on_node_creation: bool
    enable_variable_voxel_size: bool
    voxel_size_not_variable: float
    voxel_size_variable_ratio_to_length: float

    # ===== DBSCAN Parameters ===== 
    enable_roman_dbscan: bool
    run_dbscan_when_creating_node: bool
    enable_variable_epsilon: bool
    epsilon_not_variable: float
    epsilon_variable_ratio_to_length: float
    min_points: int

    # Required percentage size of cluster relative to full cloud to consider keeping node
    cluster_percentage_of_full: float

    # ===== Statistical Outlier Removal Parameters =====
    # Number of neighbors to calculate average distance for a point
    enable_remove_statistical_outliers: bool
    stat_out_num_neighbors: int
    
    # STD ratio for thresholding
    std_ratio: float

    # ===== Semantic Descriptor =====
    use_weighted_average_for_descriptor: bool
    ignore_descriptors_from_observation: bool
    calculate_descriptor_incrementally: bool

    # ===== Data inheritance from child nodes =====
    parent_node_includes_child_node_for_data: bool

    use_oriented_bbox_for_volume: bool

    merge_with_node_keep_first_seen_of_self: bool

    @classmethod
    def from_yaml(cls, path: str) -> GraphNodeParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)