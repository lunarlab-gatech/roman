from __future__ import annotations

import copy
from enum import Enum
from .graph_node import GraphNode, ParentGraphNode, LeafGraphNode
from .id_manager import IDManager
from ..map.observation import Observation
import numpy as np
from ..object.segment import Segment
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import pdist
import trimesh
from typeguard import typechecked

# TODO: Make sure the root node is properly dealt with (never deleted, never associated, never merged with, etc)

class SceneGraph3D():
    root_node: ParentGraphNode
    id_manager: IDManager

    # The two requirements for an observation to be associated with a current graph node,
    # or for two nodes to be merged.
    min_iou_for_association = 0.8
    min_sem_con_for_association = 0.8 # Also used for "Nearby Object Semantic Merging" and "Parent-Child Semantic Merging"

    # Threshold to determine if two convex hulls are overlapping in resolve_overlapping_point_clouds()
    iou_threshold_overlapping_obj = 0.2

    # Ratio of distance to object volume threshold for "Nearby Object Semantic Merging"
    ratio_dist2length_threshold_nearby_obj_semantic_merge = 0.2

    # Ratio of distance to object volume threshold for "Higher Level Object Inference"
    ratio_dist2length_threshold_higher_level_object_inference = 1.0

    # Minimum semantic consistency for two objects to infer a shared parent
    min_sem_con_for_higher_level_object_inference = 0.65

    @typechecked
    def __init__(self, root_node: ParentGraphNode):
        self.root_node = root_node
        self.id_manager = IDManager()
        self.assert_root_node_has_no_parent()

    def assert_root_node_has_no_parent(self):
        if self.root_node.noParent() is False:
            raise RuntimeError("Top-level nodes list of SceneGraph3D should only have root nodes!")

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

                        # If so, update node with observation
                        node.merge_with_observation(observations[i].transformed_points, observations[i].clip_embedding)
        
        # Resolve any overlapping point clouds
        self.resolve_overlapping_point_clouds()

        # Add the remaining unassociated observations as nodes to the scene graph
        associated_obs_indices = [x[0] for x in associated_pairs]
        for i in range(len(observations)):
            if i not in associated_obs_indices:
                self.add_new_observation_to_graph(observations[i].transformed_points, observations[i].clip_embedding)

        # Merge nodes that can be merged, and infer higher level ones
        self.merging_and_generation()

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
                sem_con = SceneGraph3D.semantic_consistency(node.get_weighted_semantic_descriptor(), obs[i].clip_embedding)

                # Calculate simularity value
                if not self.pass_minimum_requirements_for_association(iou, sem_con):
                    # If fail, then give a very large number to practically disable association
                    score = 1e9
                else:
                    # Negative as hungarian algorithm tries to minimize cost
                    score = -iou - sem_con
                scores[i,j] = score

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
    @staticmethod
    def convex_hull_geometric_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> tuple[float, float, float]:
        """
        Returns:
            iou
            enc_a_ratio (float) -> The percentage of hull a that is enclosed by b.
            enc_b_ratio (float) -> The percentage of hull b that is enclosed by a.
        """
        # Calculate the intersection trimesh
        intersection = a.intersection(b, engine='manifold')

        # Calculate the IOU value
        inter_vol = intersection.volume
        iou = inter_vol / (a.volume + b.volume - inter_vol)

        # Calculate the relative enclosure ratios
        enc_a_ratio = inter_vol / a.volume
        enc_b_ratio = inter_vol / b.volume

        return iou, enc_a_ratio, enc_b_ratio
    
    @typechecked
    @staticmethod
    def semantic_consistency(a: np.ndarray, b: np.ndarray, rescaling: list = [0.7, 1.0]):
        # Normalize both embeddings (just in case they aren't already)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        # Calculate the cosine similarity
        cos_sim = np.dot(a, b)

        # Rescale the similarity so that a similiarity <=rescaling[0] is 0 and >=rescaling[1] is 1.
        min_val, max_val = rescaling
        rescaled = (cos_sim - min_val) / (max_val - min_val)
        rescaled_clamped = np.clip(rescaled, 0.0, 1.0)
        return rescaled_clamped
    
    @typechecked
    @staticmethod
    def shortest_dist_between_convex_hulls(a: trimesh.Trimesh, b: trimesh.Trimesh):
        """ Since we sample surfance points, this is an approximation. """

        # Sample surface points on each hull
        points_a = a.sample(1000)
        points_b = b.sample(1000)

        # Get minimum distance efficently with KDTree
        tree_a = KDTree(points_a)
        distances = tree_a.query(points_b, 1)
        return np.array(distances).min()
    
    @typechecked
    def longest_line_of_convex_hull(a: trimesh.Trimesh):
        return pdist(a.vertices).max()

    @typechecked
    def pass_minimum_requirements_for_association(self, iou: float, sem_con: float) ->  bool:
        # Make sure inputs are within required bounds
        SceneGraph3D.check_within_bounds(iou, (0, 1))
        SceneGraph3D.check_within_bounds(sem_con, (0, 1))

        # Check if within thresholds for association
        iou_fail: bool = iou < self.min_iou_for_association
        sem_fail: bool = sem_con < self.min_sem_con_for_association  

        # Print warning if IOU overlap is high and semantic is far off.
        # Could be indicative of underlying bug in algorithm.
        if not iou_fail and sem_fail:
            print("WARNING: IOU overlap is high enough for assocication, but semantic is too far off. This shouldn't happen!")      

        # Return true if both requirements are fulfilled
        return iou_fail and sem_fail
    
    @typechecked
    @staticmethod
    def get_convex_hull_from_point_cloud(point_cloud: np.ndarray) -> trimesh.Trimesh:
        hull = ConvexHull(point_cloud)
        mesh = trimesh.Trimesh(vertices=point_cloud, faces=hull.simplices, process=True)
        mesh.fix_normals()
        return mesh
    
    @typechecked
    def resolve_overlapping_point_clouds(self):
        # Iterate through entire graph until no overlaps with shared points are detected
        change_occured = True
        while change_occured:
            change_occured =  False

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

                                # Remove these points from their parents
                                node_i.remove_points(pc_overlap)
                                node_j.remove_points(pc_overlap)
                                
                                # Add these points as a new node to the graph
                                self.add_new_observation_to_graph(pc_overlap, None)

                                # TODO: Long-term want to keep track of when images/masks correspond to which points,
                                # so we can go back and get a clip embedding specific to this region. Will allow us
                                # to merge this in more robustly, instead of creating small "noisy" object regions like
                                # I believe this will do.

                                # Remember to reiterate
                                change_occured = True

                            else:
                                # Otherwise, this is a "fake" overlap due to our approximation via Convex Hulls
                                # Do nothing, and it shouldn't cause issues as no actual points overlap in another's hull
                                pass

    @typechecked
    @staticmethod
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
                raise RuntimeError(f"Points in observation overlap with multiple child Convex Hulls: {overlaps.tolist()}")
            
        return contain_masks
    
    @typechecked
    def add_new_observation_to_graph(self, new_pc: np.ndarray, new_descriptor: np.ndarray | None):
        """
        Args:
            new_pc (np.ndarray): Point Cloud in shape of (N, 3) in the global frame.
        """

        # Generate dummy nodes in the graph (to be potential locations for the new node)
        self.root_node.add_dummy_nodes()

        # Calculate convex hull for this new point cloud
        hull_new = SceneGraph3D.get_convex_hull_from_point_cloud(new_pc)

        # Choose the dummy node in the graph that is the best location to add our observation
        node_index_chosen = self.choose_dummy_with_highest_likelihood(hull_new, new_descriptor)
        
        # Find the chosen node again
        for i, node in enumerate(self.root_node):
            if i == node_index_chosen:
                
                # Double-check its a dummy
                if not node.is_dummy():
                    raise RuntimeError("We chose a non-dummy node as the location for a new node, this should never happen here!")
                
                # Update dummy node with observation, and make it a non-dummy node.
                node.merge_with_observation(new_pc, new_descriptor)
                node.set_is_dummy(False)
                break

        # Prune dummy nodes from the graph
        self.root_node.prune_dummy_nodes()

        # Resolve overlapping point clouds
        self.resolve_overlapping_point_clouds()

    @typechecked
    def add_new_node_to_graph(self, node: GraphNode):
        """
        This is really only called when nodes are merged, and since the merging process has already
        pulled the relevant nodes and their children out of the graph, the combined node should 
        logically be placed back on a leaf in the remaining graph. Thus, only generate dummy nodes
        on leafs.
        """

        # Generate dummy leaf nodes in the graph.
        self.root_node.add_dummy_nodes(only_leaf=True)

        # Choose the dummy node in the graph that is the best location to add our merged node
        node_index_chosen = self.choose_dummy_with_highest_likelihood(node.get_convex_hull(), node.get_weighted_semantic_descriptor())

        # Find the chosen node again
        for i, curr_node in enumerate(self.root_node):
            if i == node_index_chosen:
                
                # Double-check its a dummy and that its a Leaf Node
                if not curr_node.is_dummy():
                    raise RuntimeError("We chose a non-dummy node as the location for a new node, this should never happen here!")
                if not curr_node.is_LeafGraphNode():
                    raise RuntimeError("We chose a non-LeafGraphNode as location for new node, this should never happen in add_new_node_to_graph!")
                
                # Remove the dummy node from the graph
                parent_node = curr_node.get_parent()
                parent_node.remove_child(curr_node)
                curr_node.set_parent(None)

                # Place our already fully formed node in its spot
                parent_node.add_child(node)
                node.set_parent(parent_node)
                break

        # Prune dummy nodes from the graph
        self.root_node.prune_dummy_nodes()

        # Resolve overlapping point clouds
        self.resolve_overlapping_point_clouds()

    @typechecked
    def choose_dummy_with_highest_likelihood(self, hull_new: trimesh.Trimesh, new_descriptor: np.ndarray) -> int:
        # Generate likelihood scores for each dummy
        likelihood_scores = np.zeros((self.len()), dtype=np.float64)
        for i, node in enumerate(self.root_node):
            if node.is_dummy():
                
                # Get the convex hull comprising of all the children, and one for the parent
                # These are different because dummy node doesn't use point cloud of parent for its hull.
                hull_all_children = node.get_convex_hull()
                hull_parent = node.get_parent().get_convex_hull()
                
                # Add similarities for geometric overlaps with children
                children_iou, children_enclosure, _ = SceneGraph3D.convex_hull_geometric_overlap(hull_all_children, hull_new)

                # Add similarities for geometric overlaps with parents
                parent_iou, _, parent_encompassment = SceneGraph3D.convex_hull_geometric_overlap(hull_parent, hull_new)

                # Calculate semantic similarity
                if new_descriptor is not None:
                    children_sem_sim = SceneGraph3D.semantic_consistency(node.get_weighted_semantic_descriptor(), new_descriptor)
                    parent_sem_sim = SceneGraph3D.semantic_consistency(node.get_parent().get_weighted_semantic_descriptor(), new_descriptor)
                else:
                    # Give it a neutral similarity if there is no descriptor
                    children_sem_sim = 0.5
                    parent_sem_sim = 0.5

                # Calculate the final likelihood score
                likelihood_scores[i] = self.calculate_likelihood_score_for_dummy(children_iou, children_enclosure, children_sem_sim,
                                                                                 parent_iou, parent_encompassment, parent_sem_sim)
            
            # If it isn't a dummy, give it a likelihood score negative infinity
            else:
                likelihood_scores[i] = -np.inf

        # Pick the node with the highest likelihood
        node_index_chosen = np.argmax(likelihood_scores)

        # Make sure we didn't pick a node with likelihood of -np.inf (meaning its a normal node
        if likelihood_scores[node_index_chosen] == -np.inf:
            raise RuntimeError("We chose a non-dummy node as the location for a new node, this should never happen here!")
        return node_index_chosen

    @typechecked
    def calculate_likelihood_score_for_dummy(self, children_iou: float, children_enclosure: float, children_sem_sim: float,
                                                   parent_iou: float, parent_encompassment: float, parent_sem_sim: float) -> float:
        # Make sure each value is within expected thresholds
        SceneGraph3D.check_within_bounds(children_iou, (0, 1))
        SceneGraph3D.check_within_bounds(children_enclosure, (0, 1))
        SceneGraph3D.check_within_bounds(children_sem_sim, (0, 1))
        SceneGraph3D.check_within_bounds(parent_iou, (0, 1))
        SceneGraph3D.check_within_bounds(parent_encompassment, (0, 1))
        SceneGraph3D.check_within_bounds(parent_sem_sim, (0, 1))

        # Calculate the final score
        return children_iou + children_enclosure + children_sem_sim + parent_iou + parent_encompassment + parent_sem_sim
    
    @typechecked
    @staticmethod
    def check_within_bounds(a: float, bounds: tuple) -> None:
        """
        Helper method to check input values.

        Args:
            bounds (tuple): These bounds are inclusive, with index 0 as lower and 1 as upper.
        """

        if a < bounds[0] or a > bounds[1]:
            raise ValueError("Value {a} is outside the range of [0, 1] inclusive.")
    
    def merging_and_generation(self):
        # Iterate over graph repeatedly until no merges/generations occur
        merge_occured = True
        while merge_occured:
            merge_occured = False

            # Iterate through each pair of nodes that aren't an ascendent or descendent with each other
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i != j and not node_i.is_descendent_or_ascendent(node_j):

                        # ========== Minimum Requirements for Association Merge ==========

                        # Calculate IOU and Semantic Similarity
                        iou, _, _ = SceneGraph3D.convex_hull_geometric_overlap(node_i.get_convex_hull(), node_j.get_convex_hull())
                        sem_con = SceneGraph3D.semantic_consistency(node_i.get_weighted_semantic_descriptor(), node_j.get_weighted_semantic_descriptor())

                        # See if they pass minimum requirements for association
                        if self.pass_minimum_requirements_for_association(iou, sem_con):

                            # If so, merge nodes in the graph
                            merged_node = node_i.merge_with_node(node_j)
                            self.add_new_node_to_graph(merged_node)

                            # Remember to break out of double-nested loop to reset iterators
                            merge_occured = True
                            break
                        
                        # ========== Nearby Object Semantic Merge ==========

                        # Get shortest distance between the nodes
                        dist = SceneGraph3D.shortest_dist_between_convex_hulls(node_i.get_convex_hull(), node_j.get_convex_hull())

                        # Get longest line of either node
                        longest_line_node_i = SceneGraph3D.longest_line_of_convex_hull(node_i.get_convex_hull())
                        longest_line_node_j = SceneGraph3D.longest_line_of_convex_hull(node_j.get_convex_hull())
                        longest_line = max(longest_line_node_i, longest_line_node_j)

                        # Get ratio of shortest distance to longest line (in other words, dist to object length)
                        dist_to_object_length = dist / longest_line

                        # If ratio of shortest distance to object length is within threshold AND
                        # the semanatic embedding is close enough for association
                        if dist_to_object_length < self.ratio_dist2length_threshold_nearby_obj_semantic_merge and \
                            sem_con < self.min_sem_con_for_association:

                            # If so, merge these two nodes in the graph
                            merged_node = node_i.merge_with_node(node_j)
                            self.add_new_node_to_graph(merged_node)

                            # Break out of double-nested loop to reset iterators
                            merge_occured = True
                            break

                        # ========== Higher Level Object Inference ==========

                        # If ratio of shorest distance to object length is within larger threshold AND
                        # the semantic embedding is close enough to assume they could have a shared parent...
                        if dist_to_object_length < self.ratio_dist2length_threshold_higher_level_object_inference and \
                            sem_con < self.min_sem_con_for_higher_level_object_inference:
                            
                            # Disconnect both of these nodes from the graph
                            node_i.get_parent().remove_child(node_i)
                            node_j.get_parent().remove_child(node_j)
                            node_i.set_parent(None)
                            node_j.set_parent(None)

                            # Create a new parent node with these as children and put back in the graph
                            inferred_parent_node = ParentGraphNode(None, [], np.zeros((0, 3), dtype=np.float64), [node_i, node_j])
                            self.add_new_node_to_graph(inferred_parent_node)

                            # Break out of double-nested loop to reset iterators
                            merge_occured = True
                            break

                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break
            
            # Iterate through pairs of parent-child relationships (to check if they should merge)
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i != j and node_i.is_parent_or_child(node_j):
                        
                        # ========== Parent-Child Semantic Merging ==========

                        # Calculate Semantic Similarity
                        sem_con = SceneGraph3D.semantic_consistency(node_i.get_weighted_semantic_descriptor(), node_j.get_weighted_semantic_descriptor())

                        # If semantic embedding is enough for assocation, just merge the child into the parent
                        if sem_con < self.min_sem_con_for_association:

                            # Get which node is the parent and which is the child
                            if node_i.is_parent(node_j): 
                                parent_node = node_i
                                child_node = node_j
                            else: 
                                parent_node = node_j
                                child_node = node_i

                            # Merge the child into the parent
                            parent_node.merge_child_with_self(child_node)

                            # Make sure nothing wierd has happened to point clouds, though I don't believe this should do anything
                            self.resolve_overlapping_point_clouds()
                        
                            # Break out of double-nested loop to reset iterators
                            merge_occured = True
                            break
                
                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break
