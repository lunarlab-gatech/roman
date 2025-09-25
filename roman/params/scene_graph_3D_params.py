from __future__ import annotations

from pydantic import BaseModel
import yaml

class SceneGraph3DParams(BaseModel):
    # ======================= Node Association =======================
    # Requirement for an observation to be associated with a current graph node or for two nodes to be merged.
    min_iou_for_association: float

    # ======================= Semantic Merges =======================
    enable_semantic_merges: bool

    # Minimum Cosine similarity for semantic merges
    min_sem_con_for_association: float
    
    # Threshold maximum of ratio of distance/length
    ratio_dist2length_threshold_nearby_node_semantic_merge: float

    # ======================= Resolve Overlapping Nodes =======================
    enable_resolve_overlapping_nodes: bool

    # Threshold to determine if two convex hulls are overlapping in resolve_overlapping_convex_hulls()
    iou_threshold_overlapping_obj: float
    enc_threshold_overlapping_obj: float
    
    # ======================= Meronomy Relationship Inference =======================
    enable_meronomy_relationship_inference: bool

    # Ratio of distance to object volume thresholds
    ratio_dist2length_threshold_shared_holonym: float
    ratio_dist2length_threshold_holonym_meronym: float

    # Ratio of detected relationship weight vs. previous total weight
    ratio_relationship_weight_2_total_weight: float

    # ======================= Node Retirement =======================
    # If a high-level node goes this long since it was first seen, inactivate
    max_t_active_for_node: float # seconds

    # If travel this dist from first place this object was seen, inactivate
    max_dist_active_for_node: float # meters

    @classmethod
    def from_yaml(cls, path: str) -> SceneGraph3DParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)

class GraphNodeParams(BaseModel):
    # If using wordnet for semantic merging, number of words to simulatneously consider ourselves as
    num_words_to_consider_ourselves: int

    # Voxel size set for downsampling relative to the longest line of node
    sample_voxel_size_to_longest_line_ratio: float

    # ===== DBSCAN Parameters ===== 
    # Epsilon set relative to longest line of point cloud
    sample_epsilon_to_longest_line_ratio: float

    # Required percentage size of cluster relative to full cloud to consider keeping node
    cluster_percentage_of_full: float

    # ===== Statistical Outlier Removal Parameters =====
    # Number of neighbors to calculate average distance for a point
    stat_out_num_neighbors: int
    
    # STD ratio for thresholding
    std_ratio: float

    @classmethod
    def from_yaml(cls, path: str) -> GraphNodeParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)