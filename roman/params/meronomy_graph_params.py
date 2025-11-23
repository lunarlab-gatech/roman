from __future__ import annotations

from pydantic import BaseModel
import yaml

class MeronomyGraphParams(BaseModel):
    """
    Params for Meronomy Graph generation and processing.
    """

    initial_word_dict: list[str]

    # ===== Synonym Merges =====
    enable_synonym_merges: bool
    min_cos_sim_for_synonym: float
    min_enc_for_synonym: float

    # ===== Resolve Overlapping Nodes =====
    enable_resolve_overlapping_nodes: bool
    iou_threshold_overlapping_obj: float
    enc_threshold_overlapping_obj: float
    overlapping_nodes_throw_away_overlap: bool

    # ===== Meronomy Relationship Inference =====
    enable_shared_holonym_inference: bool
    enable_holonym_meronym_inference: bool
    ratio_dist2length_threshold_shared_holonym: float
    ratio_dist2length_threshold_holonym_meronym: float
    ratio_relationship_weight_2_total_weight: float

    @classmethod
    def from_yaml(cls, path: str) -> MeronomyGraphParams:
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls(**raw_cfg)