from __future__ import annotations

import colorsys
from .graph_node import GraphNode, ParentGraphNode, RootGraphNode
from .hull_methods import find_point_overlap_with_hulls, get_convex_hull_from_point_cloud, \
convex_hull_geometric_overlap, shortest_dist_between_convex_hulls, longest_line_of_convex_hull
from matplotlib import rcParams
import matplotlib.pyplot as plt
from ..map.observation import Observation
import networkx as nx
import numpy as np
import open3d as o3d
import random
from scipy.optimize import linear_sum_assignment
import trimesh
from typeguard import typechecked
from typing import Union
from .visualize_graph import SceneGraphViewer

class SceneGraph3D():
    # Node that connects all highest-level objects together for implementation purposes
    root_node: ParentGraphNode

    # List of high-level nodes that have been inactivated
    inactive_nodes: list[GraphNode] = []

    # Keeps track of current time so we can keep track of when we are updated.
    times: list[float] = []

    # Keeps track of the current pose so we know where we are
    poses: list[np.ndarray] = []

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

    # If a NEW node isn't seen for this time, remove from graph
    max_t_no_sightings_to_prune_new = 0.4 # seconds

    # If a high-level node goes this long since it was first seen, inactivate
    max_t_active_for_node = 15 # seconds

    # If we travel a significant distance from the first camera pose where this object was seen, inactivate
    max_dist_active_for_node = 10 # meters

    @typechecked
    def __init__(self, _T_camera_flu: np.ndarray):
        self.root_node = RootGraphNode(None, [], np.zeros((0, 3), dtype=np.float64), [], 0, 0, 0, np.empty(0), np.empty(0))
        self.pose_FLU_wrt_Camera = _T_camera_flu
        self.assert_root_node_has_no_parent()
        self.viewer = SceneGraphViewer()

    def assert_root_node_has_no_parent(self):
        if self.root_node.no_parent() is False:
            raise RuntimeError("Top-level nodes list of SceneGraph3D should only have root nodes!")

    @typechecked
    def len(self) -> int:
        return self.root_node.get_number_of_nodes()

    @typechecked
    def update(self, time: float | np.longdouble, pose: np.ndarray, observations: list[Observation]):
        # Make sure that time ends up as a float
        time = float(time)

        # Update current time in all nodes and self
        self.times.append(time)
        self.root_node.update_curr_time(self.times[-1])

        # Set the current pose (in FLU frame) in all nodes and self
        self.poses.append(pose @ self.pose_FLU_wrt_Camera)
        self.root_node.update_curr_pose(self.poses[-1])

        # Throw away any observations that can't form a ConvexHull.
        valid_obs: list[Observation] = []
        for obs in observations:
            if obs.get_convex_hull() is not None:
                valid_obs.append(obs)
        
        # Associate observations with any current object nodes
        associated_pairs = self.hungarian_assignment(valid_obs)

        # Merge associated pairs
        for i in range(len(valid_obs)):
            for j, node in enumerate(self.root_node):

                # See if this is a match
                for pair in associated_pairs:
                    if i == pair[0] and j == pair[1]:

                        # If so, update node with observation
                        node.merge_with_observation(valid_obs[i].transformed_points, valid_obs[i].clip_embedding)
        
        # Resolve any overlapping point clouds
        self.resolve_overlapping_point_clouds()

        # Add the remaining unassociated valid_obs as nodes to the scene graph
        associated_obs_indices = [x[0] for x in associated_pairs]
        for i in range(len(valid_obs)):
            if i not in associated_obs_indices:
                self.add_new_observation_to_graph(valid_obs[i].transformed_points, valid_obs[i].clip_embedding)
                self.viewer.update(self)

        # Merge nodes that can be merged, and infer higher level ones
        self.merging_and_generation()

        # Run node retirement
        self.node_retirement()

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

                # If this is a root node, assign a score to practically disable association and skip
                if node.is_RootGraphNode():
                    scores[i,j] = 1e9
                    continue
                
                # Calculate IOU and Semantic Similarity
                iou, _, _ = convex_hull_geometric_overlap(node.get_convex_hull(), obs[i].get_convex_hull())
                sem_con = SceneGraph3D.semantic_consistency(node.get_weighted_semantic_descriptor(), obs[i].clip_embedding)

                # Check if it passes minimum requirements for association
                if self.pass_minimum_requirements_for_association(iou, sem_con):
                    # If fail (or root node), then give a very large number to practically disable association
                    scores[i,j] = 1e9
                else:
                    # Negative as hungarian algorithm tries to minimize cost
                    scores[i,j] = -iou - sem_con

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
    def pass_minimum_requirements_for_association(self, iou: float, sem_con: float) ->  bool:
        # Make sure inputs are within required bounds
        SceneGraph3D.check_within_bounds(iou, (0, 1))
        SceneGraph3D.check_within_bounds(sem_con, (0, 1))

        # Check if within thresholds for association
        iou_fail: bool = bool(iou < self.min_iou_for_association)
        sem_fail: bool = bool(sem_con < self.min_sem_con_for_association)

        # Print warning if IOU overlap is high and semantic is far off.
        # Could be indicative of underlying bug in algorithm.
        if not iou_fail and sem_fail:
            print("WARNING: IOU overlap is high enough for assocication, but semantic is too far off. This shouldn't happen!")      

        # Return true if both requirements are fulfilled
        return iou_fail and sem_fail
    
    @typechecked
    def resolve_overlapping_point_clouds(self):
        # Iterate through entire graph until no overlaps with shared points are detected
        change_occured = True
        while change_occured:
            change_occured =  False

            # Iterate through each pair of nodes that aren't an ascendent or descendent with each other
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i < j and not node_i.is_descendent_or_ascendent(node_j):

                        # Calculate the geometric overlap between them
                        hull_i = node_i.get_convex_hull()
                        hull_j = node_j.get_convex_hull()
                        iou, _, _ = convex_hull_geometric_overlap(hull_i, hull_j)

                        # If this is above a threshold, then we've found an overlap
                        if iou > self.iou_threshold_overlapping_obj:
                            
                            # Get merged point clouds from both nodes
                            pc_i = node_i.get_point_cloud()
                            pc_j = node_j.get_point_cloud()
                            pc_merged = np.concatenate((pc_i, pc_j), dtype=np.float64)

                            # Find all points in this overlap region
                            contain_masks = find_point_overlap_with_hulls(pc_merged, [hull_i, hull_j])
                            num_mask_assignments = np.sum(contain_masks, axis=0)
                            overlaps = np.where(num_mask_assignments > 1)[0]
                            pc_overlap = pc_merged[overlaps,:]

                            # If there is at least one point in this region 
                            if len(pc_overlap) > 1:

                                # Remove these points from their parents
                                node_i.remove_points(pc_overlap)
                                node_j.remove_points(pc_overlap)

                                # Try to add the new observation to the graph
                                self.add_new_observation_to_graph(pc_overlap, None)

                                # TODO: Long-term want to keep track of when images/masks correspond to which points,
                                # so we can go back and get a clip embedding specific to this region. Will allow us
                                # to merge this in more robustly, instead of creating small "noisy" object regions like
                                # I believe this will do.

                                # Remember to reiterate
                                change_occured = True
                                break

                            else:
                                # Otherwise, this is a "fake" overlap due to our approximation via Convex Hulls
                                # Do nothing, and it shouldn't cause issues as no actual points overlap in another's hull
                                pass
                
                if change_occured:
                    # Break out of both loops to reiterate from the beginning
                    break
    
    @typechecked
    def add_new_observation_to_graph(self, new_pc: np.ndarray, new_descriptor: np.ndarray | None):
        """
        Args:
            new_pc (np.ndarray): Point Cloud in shape of (N, 3) in the global frame.
        """

        # Calculate convex hull for this new point cloud
        hull_new = get_convex_hull_from_point_cloud(new_pc)
        if hull_new is None:
            # This observation doesn't have enough points to form a hull, so toss it.
            return

        # Generate dummy nodes in the graph (to be potential locations for the new node)
        self.root_node.add_dummy_nodes()
        self.visualize_2D()

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
                node.claim_parenthood_of_children()
                break

        # Prune dummy nodes from the graph
        self.visualize_2D()
        self.root_node.prune_dummy_nodes()
        self.visualize_2D()

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

        # Make sure our new node has a enough points to successfully form a hull
        hull_new = node.get_convex_hull()
        if hull_new is None: 
            raise ValueError("Merged node from two nodes can't form a Convex Hull, this should never happen!")

        # Generate dummy leaf nodes in the graph.
        self.root_node.add_dummy_nodes(only_leaf=True)

        # Choose the dummy node in the graph that is the best location to add our merged node
        node_index_chosen = self.choose_dummy_with_highest_likelihood(hull_new, node.get_weighted_semantic_descriptor())

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
    def choose_dummy_with_highest_likelihood(self, hull_new: trimesh.Trimesh, new_descriptor: np.ndarray | None) -> int:
        # Generate likelihood scores for each dummy
        likelihood_scores = np.zeros((self.len()), dtype=np.float64)
        for i, node in enumerate(self.root_node):
            if node.is_dummy():
                
                # Get the convex hull comprising of all the children, and one for the parent
                # These are different because dummy node doesn't use point cloud of parent for its hull.
                hull_all_children = node.get_convex_hull()
                hull_parent = node.get_parent().get_convex_hull()
                
                # Add similarities for geometric overlaps with children
                children_iou, children_enclosure, _ = convex_hull_geometric_overlap(hull_all_children, hull_new)

                # Add similarities for geometric overlaps with parents
                parent_iou, _, parent_encompassment = convex_hull_geometric_overlap(hull_parent, hull_new)

                # Calculate semantic similarity
                if new_descriptor is not None:
                    # Calculate similarity with children
                    if not node.is_LeafGraphNode():
                        children_sem_sim = SceneGraph3D.semantic_consistency(node.get_weighted_semantic_descriptor(), new_descriptor)
                    else:
                        # A dummy leaf graph node has no children, so assume neutral similarity
                        children_sem_sim = 0.5

                    # Calculate similarity with parent
                    if not node.get_parent().is_RootGraphNode():
                        parent_sem_sim = SceneGraph3D.semantic_consistency(node.get_parent().get_weighted_semantic_descriptor(), new_descriptor)
                    else:
                        # Neutral similarity with root node, as it represents everything
                        parent_sem_sim = 0.5
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
        return int(node_index_chosen)

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
                    if i < j and not node_i.is_descendent_or_ascendent(node_j):

                        # ========== Minimum Requirements for Association Merge ==========

                        # Calculate IOU and Semantic Similarity
                        iou, _, _ = convex_hull_geometric_overlap(node_i.get_convex_hull(), node_j.get_convex_hull())
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
                        dist = shortest_dist_between_convex_hulls(node_i.get_convex_hull(), node_j.get_convex_hull())

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
                            first_seen = min(node_i.get_time_first_seen(), node_j.get_time_first_seen())
                            inferred_parent_node = ParentGraphNode(None, [], np.zeros((0, 3), dtype=np.float64), [node_i, node_j], first_seen, self.times[-1])
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
                    if i < j and node_i.is_parent_or_child(node_j):
                        
                        # ========== Parent-Child Semantic Merging ==========

                        # If either is the root node, skip as this shouldn't really merge with any children
                        if node_i.is_RootGraphNode() or node_j.is_RootGraphNode():
                            continue

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

    def node_retirement(self):
        # Iterate only through the direct children of the root node
        for child in self.root_node.get_children()[:]:

            # If the time the child was first seen was too long ago OR
            # we've move substancially since then, Inactivate this node and descendents
            if self.times[-1] - child.get_time_first_seen() > self.max_t_active_for_node \
                or np.linalg.norm(self.poses[-1][:3,3] - child.get_first_pose()[:3,3]) > self.max_dist_active_for_node:

                # Pop this child off of the root node and put in our inactive nodes
                self.root_node.remove_child(child)
                self.inactive_nodes.append(child)

    def visualize_2D(self, filename: str = "scene_graph.png"):
        rcParams.update({'figure.autolayout': True})
        G = nx.DiGraph()

        node_colors = {}

        # Recursive traversal and node/edge addition
        def add_node_recursive(node, parent_id=None):
            node_id = id(node)
            label = getattr(node, "get_label", lambda: f"Node {node_id}")()
            G.add_node(node_id, label=label)

            # Color red if dummy, blue otherwise
            color = 'red' if getattr(node, "is_dummy", lambda: False)() else 'skyblue'
            node_colors[node_id] = color

            if parent_id is not None:
                G.add_edge(parent_id, node_id)

            for child in node.get_children():
                add_node_recursive(child, node_id)

        # Start from root
        add_node_recursive(self.root_node)

        # Layout and labels
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        labels = nx.get_node_attributes(G, 'label')

        # Draw with color distinction
        plt.figure(figsize=(14, 10))
        nx.draw(
            G, pos, with_labels=True, labels=labels,
            node_size=1500,
            node_color=[node_colors[n] for n in G.nodes()],
            font_size=8, font_color='black', arrows=True
        )

        plt.title("SceneGraph3D Structure")
        plt.savefig(filename, dpi=300)
        plt.close()
