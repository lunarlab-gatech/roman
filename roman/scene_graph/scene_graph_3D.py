from __future__ import annotations

import copy
from enum import Enum
from .graph_node import GraphNode, ParentGraphNode, LeafGraphNode
from .id_manager import IDManager
from ..map.observation import Observation
import numpy as np
from ..object.segment import Segment
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
import trimesh
from typeguard import typechecked


class SceneGraph3D():
    root_node: ParentGraphNode
    id_manager: IDManager

    min_iou_for_association = 0.8
    min_sem_con_for_association = 0.8
    iou_threshold_overlapping_obj = 0.2

    @typechecked
    def __init__(self, root_node: ParentGraphNode):
        self.root_node = root_node
        self.id_manager = IDManager()
        self.assert_root_node_has_no_parent()

    def assert_root_node_has_no_parent(self):
        if self.root_node.noParent() is False:
            raise RuntimeError("Top-level nodes list of SceneGraph3D should only have root nodes!")

    @typechecked
    def add_dummy_nodes(self, node_list: list[GraphNode]):
        raise NotImplementedError("NEEDS REWORKING")
        # Iterate through each node
        for node in node_list:

            # If there is no parent, add one as a dummy node
            if node.noParent():
                node.setParent(GraphNode.create_dummy_node(self.id_manager.acquire(), None))
            else: # Otherwise, add a dummy node between parent and child (to represent a missing hierarchical layer)
                intermediate_node = GraphNode.create_dummy_node(self.id_manager.acquire(), node.parent_node)
                node.setDummyParent(intermediate_node)
                node.parent_node.addChild(intermediate_node)

            # Add a dummy node regardless if it has children or not
            node.addChild(GraphNode.create_dummy_node(self.id_manager.acquire(), None))

            # Recursively add dummy nodes to children as well
            if isinstance(node, ParentGraphNode):
                self.add_dummy_nodes(node.child_nodes)

    @typechecked
    def len(self) -> int:
        return self.root_node.get_number_of_nodes()

    @typechecked
    def update(self, observations: list[Observation]):
        
        # Associate observations with any current object nodes
        associated_pairs = self.hungarian_assignment(observations)

        # Merge associated pairs
        for i in range(len(observations)):
            for j, node in self.root_node:

                # See if this is a match
                for pair in associated_pairs:
                    if i == pair[0] and j == pair[1]:

                        # If so, merge it with the node
                        node.update_with_observation(observations[i])
        
        # Resolve any overlapping point clouds

    @typechecked
    def hungarian_assignment(self, obs: list[Observation]) -> list[tuple]:
        """
        Associates observations with nodes in this scene graph by solving as a linear sum assignment problem
        via the Hungarian method.
        """

        # Get number of observations and nodes
        num_obs = len(obs)
        num_nodes = self.len()

        # Setup score matrix
        scores = np.zeros((num_obs, num_nodes))

        # Iterate through each observation
        for i in range(num_obs):
            # Calculate a similarity score for every node in the graph
            for j, node in enumerate(self.root_node):
                # Calculate IOU and Semantic Similarity
                iou, _, _ = SceneGraph3D.convex_hull_geometric_overlap(node.get_convex_hull(), obs[i].get_convex_hull())
                sem_con = SceneGraph3D.semantic_consistency(node, obs[i])

                # See if they are within thresholds
                iou_fail: bool = node.iou < self.min_iou_for_association
                sem_fail: bool = node.sem_con < self.min_sem_con_for_association

                # Geometry similarity value
                if iou_fail or sem_fail:
                    score = 1e9
                else:
                    score = -iou - sem_con
                scores[i,j] = score

                # Print warning if necessary
                if not iou_fail and sem_fail:
                    print("WARNING: IOU overlap is high enough for assocication, but semantic is too far off. This shouldn't happen!")

        # augment cost to add option for no associations
        hungarian_cost = np.concatenate([ np.concatenate([scores, np.ones(scores.shape)], axis=1), np.ones((scores.shape[0], 2*scores.shape[1]))], axis=0)
        row_ind, col_ind = linear_sum_assignment(hungarian_cost)

        pairs = []
        for idx1, idx2 in zip(row_ind, col_ind):
            # state and measurement associated together
            if idx1 < num_obs and idx2 < num_nodes:
                pairs.append((idx1, idx2))
        return pairs
    
    @typechecked
    def convex_hull_geometric_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> tuple[float, float, float]:
        # Calculate the intersection trimesh
        intersection = a.intersection(b, engine='manifold')

        # Calculate the IOU value
        inter_vol = intersection.volume
        iou = inter_vol / (a.volume + b.volume - inter_vol)

        # Calculate the relative enclosure ratios
        enc_seg_ratio = inter_vol / a.volume
        enc_obs_ratio = inter_vol / b.volume

        return iou, enc_seg_ratio, enc_obs_ratio
    
    @typechecked
    def semantic_consistency(node: GraphNode, observation: Observation, rescaling: list = [0.7, 1.0]):
        # Normalize observation embeddings
        obs_emb = observation.clip_embedding / np.linalg.norm(observation.clip_embedding)

        # Calculate the cosine similarity (they are assumed to be normalized)
        cos_sim = np.dot(node.get_weighted_semantic_descriptor(), obs_emb)

        # Rescale the similarity so that a similiarity <=rescaling[0] is 0 and >=rescaling[1] is 1.
        min_val, max_val = rescaling
        rescaled = (cos_sim - min_val) / (max_val - min_val)
        rescaled_clamped = np.clip(rescaled, 0.0, 1.0)
        return rescaled_clamped
    
    @typechecked
    def resolve_overlapping_point_clouds(self):
        # Iterate through each node pair in the scene graph
        for i, node_i in enumerate(self.root_node):
            for j, node_j  in enumerate(self.root_node):
                if i < j:

                    # Calculate the geometric overlap between them
                    hull_i = node_i.get_convex_hull()
                    hull_j = node_j.get_convex_hull()
                    iou, _, _ = self.convex_hull_geometric_overlap(hull_i, hull_j)

                    # If this is above a threshold, then we've found an overlap
                    if iou > self.iou_threshold_overlapping_obj:
                        
                        # Get merged point clouds from both nodes
                        pc_i = node_i.get_point_cloud()
                        pc_j = node_j.get_point_cloud()
                        pc_merged = np.concatenate((pc_i, pc_j), dtype=np.float64)

                        # Find all points in this overlap region
                        contain_masks = self.find_point_overlap_with_hulls(pc_merged, [hull_i, hull_j])
                        num_mask_assignments = np.sum(contain_masks, axis=0)
                        overlaps = np.where(num_mask_assignments > 1)[0]
                        pc_overlap = pc_merged[overlaps,:]

                        # If there is at least one point in this region 
                        if len(pc_overlap) > 1:

                            # Then turn this overlap into a "new" observation and add to the graph.
                            raise NotImplementedError("FINISH ME")

                        else:
                            # Otherwise, this is a "fake" overlap due to our approximation via Convex Hulls
                            # Do nothing, and it shouldn't cause issues as no actual points overlap in another's hull
                            pass

    @typechecked
    def find_point_overlap_with_hulls(pc: np.ndarray, hulls: list[trimesh.Trimesh], fail_on_multi_assign: bool = False) -> np.ndarray:  
        # Find which points fall into which Convex hulls
        contain_masks = np.zeros((len(hulls), len(pc)), dtype=int)
        for i, hull in enumerate(hulls):
            contain_masks[i] = np.array(hull.contains(pc), dtype=int)

        # If fail_on_multi_assign, throw an error if some points fall into multiple hulls
        if fail_on_multi_assign:
            num_mask_assignments = np.sum(contain_masks, axis=0)
            if np.any(num_mask_assignments > 1):
                overlaps = np.where(num_mask_assignments > 1)[0]
                raise RuntimeError(f"Points in observation overlap wiht multiple child Convex Hulls: {overlaps.tolist()}")
            
        return contain_masks
        

