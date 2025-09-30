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
    voxel_size_for_voxel_grid_iou: float

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

    @classmethod
    def from_yaml(cls, path: str) -> SceneGraph3DParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)

class GraphNodeParams(BaseModel):
    # If using wordnet for semantic merging, number of words to simulatneously consider ourselves as
    num_words_to_consider_ourselves: int

    # ===== Voxel Downsampling ===== 
    enable_variable_voxel_size: bool
    voxel_size_not_variable: float
    voxel_size_variable_ratio_to_length: float

    # ===== DBSCAN Parameters ===== 
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
    get_semantic_descriptors_includes_children: bool

    # ===== Point Clouds =====
    get_point_cloud_includes_children: bool

    @classmethod
    def from_yaml(cls, path: str) -> GraphNodeParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)