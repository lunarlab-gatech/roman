import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum

import clipperpy

from roman.align.object_registration import ObjectRegistration
from roman.object.object import Object
from roman.scene_graph.graph_node import GraphNode
from roman.scene_graph.scene_graph_3D import SceneGraph3D

class FusionMethod(Enum):
    GEOMETRIC_MEAN = clipperpy.invariants.ROMAN.GEOMETRIC_MEAN
    ARITHMETIC_MEAN = clipperpy.invariants.ROMAN.ARITHMETIC_MEAN
    PRODUCT = clipperpy.invariants.ROMAN.PRODUCT

@dataclass
class ROMANParams:
    
    point_dim: int = 3
    fusion_method = FusionMethod.GEOMETRIC_MEAN

    sigma: float = 0.4
    epsilon: float = 0.6
    mindist: float = 0.2

    gravity: bool = False
    volume: bool = False
    pca: bool = False
    extent: bool = False
    semantics_dim = 0

    cos_min: float = 0.85
    cos_max: float = 0.95
    epsilon_shape: float = None


class ROMANRegistration(ObjectRegistration):
    def __init__(self, params: ROMANParams):
        super().__init__(dim=params.point_dim)

        ratio_feature_dim = 0
        self.volume = params.volume
        self.extent = params.extent
        self.pca = params.pca
        self.semantics = params.semantics_dim > 0
        
        if self.pca:
            ratio_feature_dim += 3
        if self.volume:
            ratio_feature_dim += 1
        if self.extent:
            ratio_feature_dim += 3

        self.iparams = clipperpy.invariants.ROMANParams()
        self.iparams.point_dim = params.point_dim
        self.iparams.ratio_feature_dim = ratio_feature_dim
        self.iparams.cos_feature_dim = params.semantics_dim

        self.iparams.sigma = params.sigma
        self.iparams.epsilon = params.epsilon
        self.iparams.mindist = params.mindist

        self.iparams.distance_weight = 1.0
        self.iparams.ratio_weight = 1.0
        self.iparams.cosine_weight = 1.0

        self.iparams.ratio_epsilon = np.zeros(ratio_feature_dim) \
            if params.epsilon_shape is None \
            else np.ones(ratio_feature_dim) *  params.epsilon_shape 
        self.iparams.cosine_min = params.cos_min
        self.iparams.cosine_max = params.cos_max

        self.iparams.gravity_guided = params.gravity
        self.iparams.drift_aware = False
        
        return
    
    def _setup_clipper(self):
        """ Initializes classes """
        
        # Initialize the ROMAN clipper method (clipper::invariants::ROMAN)
        invariant = clipperpy.invariants.ROMAN(self.iparams)

        # Initialize parameters for CLIPPER (clipper::Params)
        params = clipperpy.Params()

        # Wrap with convenience class (clipper:CLIPPERPairwiseAndSingle)
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)
        return clipper
    
    def _clipper_score_all_to_all(self, clipper, map1: List[GraphNode], map2: List[GraphNode]):

        # Detect putative shared holonyms, holonym-meronym relationships, and synonymys
        putative_shared_holonyms: list[tuple[int, int, list[str]]] = SceneGraph3D.find_putative_relationships(
                                                     map1, map2, SceneGraph3D.NodeRelationship.SHARED_HOLONYM)
        putative_holonym_meronyms: list[tuple[int, int, bool]] = SceneGraph3D.find_putative_relationships(
                                                     map1, map2, SceneGraph3D.NodeRelationship.HOLONYM_MERONYM)
        putative_synonymys: list[tuple[int, int, None]] =  SceneGraph3D.find_putative_relationships(
                                                     map1, map2, SceneGraph3D.NodeRelationship.SYNONYMY)
           
        # Calculate number of putative meronyms in each map
        num_putative_meronyms_map1: int = sum(1 for _, _, final_bool in putative_holonym_meronyms if final_bool)
        num_putative_meronyms_map2: int = sum(1 for _, _, final_bool in putative_holonym_meronyms if not final_bool)

        # Calculate the total number of nodes in each map (real & putative)
        total_nodes_map_1: int = len(map1) + len(putative_shared_holonyms) + num_putative_meronyms_map1
        total_nodes_map_2: int = len(map2) + len(putative_shared_holonyms) + num_putative_meronyms_map2

        # Create an all-to-all association matrix (n1*n2 x 2)
        A_init: np.ndarray = clipperpy.utils.create_all_to_all(total_nodes_map_1, total_nodes_map_2)

        # Get list of necessary data for each object (N x Num Data Values)
        map1_cl = np.array([self._object_to_clipper_list(p) for p in map1])
        map2_cl = np.array([self._object_to_clipper_list(p) for p in map2])
        self._check_clipper_arrays(map1_cl, map2_cl)

        # Calculate scores for each association in A_init(clipper::CLIPPERPairwiseAndSingle::scorePairwiseAndSingleConsistency)
        clipper.score_pairwise_and_single_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init

    def _object_to_clipper_list(self, object: Object) -> list:       
        """ Convert object into list of values we need for comparison/scoring. """ 

        # Extract center of object (could be centroid or bottom-middle)
        object_as_list: list = object.center.reshape(-1).tolist()[:self.dim]
        
        # Extract attributes of point cloud via PCA
        if self.pca:
            e = object.normalized_eigenvalues()
            object_as_list += [object.linearity(e), object.planarity(e), object.scattering(e)]

        # Extract object volume
        if self.volume:
            object_as_list.append(object.volume)

        # Extract extent of OBB
        if self.extent:
            object_as_list += sorted(object.extent)

        # Extract semantic descriptor
        if self.semantics:
            object_as_list += np.array(object.semantic_descriptor).tolist()
        return object_as_list 
    
    def _check_clipper_arrays(self, map1_cl, map2_cl):
        assert map1_cl.shape[1] == map2_cl.shape[1]
        # TODO: check that the number of point elements + feature elements is correct
        # if self.use_gravity:
        #     assert map1_cl.shape[1] == 3 + 2, f"map1_cl.shape[1] = {map1_cl.shape[1]}"
        return