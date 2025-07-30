from __future__ import annotations

from .graph_node import GraphNode, RootGraphNode
from graphviz import Digraph
from .hull_methods import find_point_overlap_with_hulls, get_convex_hull_from_point_cloud, \
convex_hull_geometric_overlap, shortest_dist_between_convex_hulls
from itertools import chain, combinations
from .logger import logger
from matplotlib import rcParams
import matplotlib.pyplot as plt
from ..map.observation import Observation
import networkx as nx
import numpy as np
import pickle
from .rerun_wrapper import RerunWrapper
from scipy.optimize import linear_sum_assignment
import trimesh
from typeguard import typechecked
from .visualize_graph import SceneGraphViewer

class SceneGraph3D():
    # Node that connects all highest-level objects together for implementation purposes
    root_node: RootGraphNode

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
        #self.viewer = SceneGraphViewer()

        # For visualize_2D()
        # self.G = None
        # self.pos = None
        # self.fig, self.ax = plt.subplots(figsize=(14, 10))
        # plt.ion()  # interactive mode on
        # self.fig.show()
        # self.node_colors = {}
        # self.last_num_nodes = 0

        self.rerun_viewer = RerunWrapper()

    def assert_root_node_has_no_parent(self):
        if self.root_node.no_parent() is False:
            raise RuntimeError("Top-level nodes list of SceneGraph3D should only have root nodes!")

    @typechecked
    def len(self) -> int:
        return self.root_node.get_number_of_nodes()

    @typechecked
    def update(self, time: float | np.longdouble, pose: np.ndarray, observations: list[Observation]):
        logger.info(f"Called with {len(observations)} observations")
        with open("observations.pkl", 'wb') as file:
            pickle.dump(observations, file)

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
        logger.info(f"Remaining valid observations: {len(valid_obs)}")
        
        # Associate observations with any current object nodes
        associated_pairs = self.hungarian_assignment(valid_obs)

        # Merge associated pairs
        logger.info(f"[dark_blue]Association Merges[/dark_blue] for {len(associated_pairs)} observations")
        for i in range(len(valid_obs)):
            for j, node in enumerate(self.root_node):

                # See if this is a match
                for pair in associated_pairs:
                    if i == pair[0] and j == pair[1]:

                        # If so, update node with observation
                        node.merge_with_observation(valid_obs[i].transformed_points, valid_obs[i].clip_embedding)
        
        # Add the remaining unassociated valid_obs as nodes to the scene graph
        associated_obs_indices = [x[0] for x in associated_pairs]
        for i in range(len(valid_obs)):
            if i not in associated_obs_indices:
                self.add_new_observation_to_graph(valid_obs[i].transformed_points, valid_obs[i].clip_embedding)

        # Merge nodes that can be merged, and infer higher level ones
        self.merging_and_generation()

        # Resolve any overlapping point clouds
        self.resolve_overlapping_point_clouds()

        # Run node retirement
        self.node_retirement()

        # Update the viewer
        self.rerun_viewer.update(self.root_node, self.times[-1])
        # self.viewer.update(self.root_node.get_children())
        # self.visualize_2D()

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
                    # Negative as hungarian algorithm tries to minimize cost
                    scores[i,j] = -iou - sem_con
                else:
                    # If fail (or root node), then give a very large number to practically disable association
                    scores[i,j] = 1e9
                    
        # Augment cost to add option for no associations
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
    def semantic_consistency(a: np.ndarray| None, b: np.ndarray | None, rescaling: list = [0.7, 1.0]) -> float:
        # If either is none, then just assume neutral consistency
        if a is None or b is None:
            return 0.5

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
        iou_pass: bool = bool(iou >= self.min_iou_for_association)
        sem_pass: bool = bool(sem_con >= self.min_sem_con_for_association)

        # Print warning if IOU overlap is high and semantic is far off.
        # Could be indicative of underlying bug in algorithm.
        if iou_pass and not sem_pass:
            raise RuntimeError(f"WARNING: IOU overlap {iou} is high enough for association, but semantic {sem_con} is too far off. This shouldn't happen!")      

        # Return true if both requirements are fulfilled
        return iou_pass and sem_pass
    
    @typechecked
    def resolve_overlapping_point_clouds(self):
        # Iterate through entire graph until no overlaps with shared points are detected
        change_occured = True
        while change_occured:
            self.rerun_viewer.update(self.root_node, self.times[-1])
            # self.viewer.update(self.root_node.get_children())
            # self.visualize_2D()
            change_occured =  False

            # Iterate through each pair of nodes that aren't an ascendent or descendent with each other
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i < j and not node_i.is_descendent_or_ascendent(node_j):

                        # If the nodes are obviously seperated, no point in checking.
                        if not self.check_if_nodes_are_somewhat_nearby(node_i, node_j):
                            continue

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

                            # If there is at least four points in this region 
                            if len(pc_overlap) >= 4:

                                # Double check this isn't the same node
                                assert node_i.get_id() != node_j.get_id(), f"Same node {node_i.get_id()} referred to as child of two parents!"

                                # Remove these points from their parents
                                logger.info(f"[bright_red]Overlap IOU {iou}[/bright_red] between Node {node_i.get_id()} and Node {node_j.get_id()}...")
                                node_i.remove_points_complete(pc_overlap)
                                node_j.remove_points_complete(pc_overlap)
                      
                                # Try to add the new observation to the graph
                                logger.info(f"Adding overlap region of {len(pc_overlap)} points as observation to graph...")
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

        # Create a new node for this observation
        new_node = GraphNode(self.root_node.request_new_ID(), None, [], new_pc, [], 
                             self.times[-1], self.times[-1], self.times[-1], self.poses[-1], self.poses[-1])
        if new_descriptor is not None:
            new_node.add_semantic_descriptors([(new_descriptor, new_node.get_volume())])
        
        # Add this node to the graph
        self.add_new_node_to_graph(new_node, only_leaf=False)

    @typechecked
    def add_new_node_to_graph(self, node: GraphNode, only_leaf=False):

        # Make sure our new node has a enough points to successfully form a hull
        hull_new = node.get_convex_hull()
        if hull_new is None: 
            logger.info(f"New Node {node.get_id()} discarded for inability to form a ConvexHull")
            return 

        # Keep track of the current likeliest node
        best_likelihood_score = -np.inf
        best_likelihood_position = None

        # Helper method to handle next loop in iteration below
        def next_loop(pos_parent_node: GraphNode, node_queue: list[GraphNode]):
            if len(node_queue) >= 1:
                pos_parent_node: GraphNode = node_queue[0]
                node_queue = node_queue[1:]
            else:
                pos_parent_node: None = None
            return pos_parent_node, node_queue

        # Iterate through each node, considering ourselves as their potential child
        pos_parent_node: GraphNode | None = self.root_node
        node_queue: list[GraphNode] = []
        while pos_parent_node is not None:

            # See if this node is anywhere near us. If not, we obviously can't be their child
            if not SceneGraph3D.check_if_nodes_are_somewhat_nearby(node, pos_parent_node):
                pos_parent_node, node_queue = next_loop(pos_parent_node, node_queue)
                continue

            # Since we're nearby this, add all of this nodes children to the queue as well
            node_queue += pos_parent_node.get_children()

            # Add similarities for geometric overlaps with parent
            parent_iou, _, parent_encompassment = convex_hull_geometric_overlap(pos_parent_node.get_convex_hull(), hull_new)

            # Add similarities for geometric overlaps with children
            children_iou, children_enclosure, children_volumes = [], [], []
            if not only_leaf:
                hull_all_children: list[trimesh.Trimesh | None] = [child.get_convex_hull() for child in pos_parent_node.get_children()]
                children_volumes = [child.get_volume() for child in pos_parent_node.get_children()]
                for hull in hull_all_children:
                    iou, child_enc, _ = convex_hull_geometric_overlap(hull, hull_new)
                    children_iou.append(iou)
                    children_enclosure.append(child_enc)

            # Calculate semantic similarity with children
            children_sem_sim = []
            node_sem_des = node.get_weighted_semantic_descriptor()
            if not only_leaf:
                for child in pos_parent_node.get_children():
                    children_sem_sim.append(SceneGraph3D.semantic_consistency(node_sem_des, child.get_weighted_semantic_descriptor()))

            # Calculate similarity with parent
            if not pos_parent_node.is_RootGraphNode():
                parent_sem_sim = SceneGraph3D.semantic_consistency(node_sem_des, pos_parent_node.get_weighted_semantic_descriptor())
            else:
                # Neutral similarity with root node, as it represents everything
                parent_sem_sim = 0.5

            # Calculate the final likelihood score
            score, subset = self.calculate_best_likelihood_score(children_iou, children_enclosure, children_sem_sim, children_volumes,
                                                                 parent_iou, parent_encompassment, parent_sem_sim, only_leaf=only_leaf)
            
            # If this is the best score so far, keep it
            logger.debug(f"Best Likelihood for Node {pos_parent_node.get_id()}: {score}, with children {subset}")
            if score >= best_likelihood_score:
                best_likelihood_score = score
                best_likelihood_position = (pos_parent_node, subset)

            # After checking this node, move to check the next one
            pos_parent_node, node_queue = next_loop(pos_parent_node, node_queue)
        
        # Extract new parent and children of this node
        new_parent: GraphNode = best_likelihood_position[0]
        new_children = [new_parent.get_children()[i] for i in best_likelihood_position[1]]

        # Place our already fully formed node into its spot
        if len(new_children) > 0:
            logger.info(f"Node {node.get_id()} added to graph as child of Node {new_parent.get_id()} and parent of Nodes {[c.get_id() for c in new_children]}")
        else:
            logger.info(f"Node {node.get_id()} added to graph as child of Node {new_parent.get_id()}")
        new_parent.add_child(node)
        node.set_parent(new_parent)

        new_parent.remove_children(new_children)
        for child in new_children:
            child.set_parent(None)

        node.add_children(new_children)
        for child in new_children:
            child.set_parent(node)

        # Resolve overlapping point clouds
        # self.resolve_overlapping_point_clouds()

    @typechecked
    def calculate_best_likelihood_score(self, children_iou: list[float], children_enclosure: list[float], children_sem_sim: list[float],
                                    children_volumes: list[float], parent_iou: float, parent_encompassment: float, parent_sem_sim: float,
                                    only_leaf: bool) -> tuple[float, tuple[int, ...]]:

        # Assert each array is of the same length
        expected_len = len(children_iou)
        assert expected_len == len(children_enclosure)
        assert expected_len == len(children_sem_sim)
        assert expected_len == len(children_volumes)

        # Make sure each value is within expected thresholds
        for i in range(len(children_iou)):
            SceneGraph3D.check_within_bounds(children_iou[i], (0, 1))
            SceneGraph3D.check_within_bounds(children_enclosure[i], (0, 1))
            SceneGraph3D.check_within_bounds(children_sem_sim[i], (0, 1))
            SceneGraph3D.check_within_bounds(children_volumes[i], (0, np.inf))
        SceneGraph3D.check_within_bounds(parent_iou, (0, 1))
        SceneGraph3D.check_within_bounds(parent_encompassment, (0, 1))
        SceneGraph3D.check_within_bounds(parent_sem_sim, (0, 1))

        # Keep track of the best score and subset
        best_score = -np.inf
        best_subset = None
        
        # Create powerset of all possible children combinations in the current node
        if not only_leaf:
            num_children = len(children_iou)
            powerset = chain.from_iterable(combinations(range(num_children), r) for r in range(num_children + 1))
        else:
            # Only consider the possiblity that we are a leaf node of this node
            powerset = [()]

        # Iterate through each subset of child combinations and create a dummy node with those children
        for subset in powerset:
            
            # Calculate children weighted IOU, enclosure, and semantic similarity
            if len(subset) == 0:
                # No children selected, put netural likelihoods
                temp_child_iou = 0.0
                temp_child_enc = 1.0
                temp_child_sem_sim = 0.5
            else:
                # Get the values for this specific combination of children
                rel_child_iou = [children_iou[i] for i in subset]
                rel_child_enc = [children_enclosure[i] for i in subset]
                rel_child_sem_sim = [children_sem_sim[i] for i in subset]
                rel_volumes = [children_volumes[i] for i in subset]

                # Get a weighted average of them based on each childs volume
                temp_child_iou = np.average(rel_child_iou, axis=0, weights=rel_volumes)
                temp_child_enc = np.average(rel_child_enc, axis=0, weights=rel_volumes)
                temp_child_sem_sim = np.average(rel_child_sem_sim, axis=0, weights=rel_volumes)

            # Calculate the final score
            score = temp_child_iou + temp_child_enc + temp_child_sem_sim + parent_iou + parent_encompassment + parent_sem_sim

            # If this is the best so far, keep track of it
            if score >= best_score:
                logger.debug(f"New best Subset ({subset}): {temp_child_iou} {temp_child_enc} {temp_child_sem_sim} {parent_iou}  {parent_encompassment}  {parent_sem_sim}")
                best_score = score
                best_subset = subset
        
        # Return the result of the best child combination score
        return best_score, best_subset
    
    @typechecked
    @staticmethod
    def check_within_bounds(a: float, bounds: tuple) -> None:
        """
        Helper method to check input values.

        Args:
            bounds (tuple): These bounds are inclusive, with index 0 as lower and 1 as upper.
        """

        if a < bounds[0] or a > bounds[1]:
            raise ValueError(f"Value {a} is outside the range of [0, 1] inclusive.")
    
    @staticmethod
    @typechecked
    def check_if_nodes_are_somewhat_nearby(a: GraphNode, b: GraphNode) -> bool:
        """ Returns False if node centroids are not within size_a + size_b of each other. """

        # If one is the RootGraphNode, then obviously they are nearby
        if a.is_RootGraphNode() or b.is_RootGraphNode():
            return True

        # See if they are close enough
        size_a = a.get_longest_line_size()
        size_b = b.get_longest_line_size()
        c_a = a.get_centroid()
        c_b = b.get_centroid()
        if np.linalg.norm(c_a - c_b) > size_a + size_b: return False
        else: return True

    def merging_and_generation(self):
        # Iterate over graph repeatedly until no merges/generations occur
        merge_occured = True
        while merge_occured:
            # Update the viewer
            self.rerun_viewer.update(self.root_node, self.times[-1])
            # self.viewer.update(self.root_node.get_children())
            # self.visualize_2D()

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
                            logger.info(f"[dark_blue]Association Merge[/dark_blue]: Merging Node {node_i.get_id()} into Node {node_j.get_id()} and popping off graph")
                            merged_node = node_i.merge_with_node(node_j)
                            self.add_new_node_to_graph(merged_node, only_leaf=True)

                            # Remember to break out of double-nested loop to reset iterators
                            merge_occured = True
                            break
                        
                        # ========== Nearby Object Semantic Merge ==========

                        # # Get shortest distance between the nodes
                        # dist = shortest_dist_between_convex_hulls(node_i.get_convex_hull(), node_j.get_convex_hull())

                        # # Get longest line of either node
                        # longest_line_node_i = node_i.get_longest_line_size()
                        # longest_line_node_j = node_j.get_longest_line_size()
                        # longest_line = max(longest_line_node_i, longest_line_node_j)

                        # # Get ratio of shortest distance to longest line (in other words, dist to object length)
                        # dist_to_object_length = dist / longest_line

                        # # If ratio of shortest distance to object length is within threshold AND
                        # # the semanatic embedding is close enough for association
                        # if dist_to_object_length < self.ratio_dist2length_threshold_nearby_obj_semantic_merge and \
                        #     sem_con < self.min_sem_con_for_association:

                        #     # If so, merge these two nodes in the graph
                        #     logger.info(f"[green1]Nearby Obj Sem Merge[/green1]: Merging Node {node_i.get_id()} into Node {node_j.get_id()} and popping off graph")
                        #     logger.info(f"SHORTEST DIST: {dist}")
                        #     logger.info(f"LONGEST LINE: {longest_line}")
                        #     print(f"DIST TO OBJEDT LENGTH: {dist_to_object_length} < {self.ratio_dist2length_threshold_nearby_obj_semantic_merge}")
                        #     merged_node = node_i.merge_with_node(node_j)
                        #     self.add_new_node_to_graph(merged_node, only_leaf=True)

                        #     # Break out of double-nested loop to reset iterators
                        #     merge_occured = True
                        #     break

                        # ========== Higher Level Object Inference ==========

                        # If they already have a shared parent that isn't the RootGraphNode, no point checking this...
                        # if node_i.get_parent() == node_j.get_parent() and not node_i.get_parent().is_RootGraphNode():
                        #     pass
                        
                        # # If ratio of shorest distance to object length is within larger threshold AND
                        # # the semantic embedding is close enough to assume they could have a shared parent...
                        # elif dist_to_object_length < self.ratio_dist2length_threshold_higher_level_object_inference and \
                        #     sem_con < self.min_sem_con_for_higher_level_object_inference:
                            
                        #     # Disconnect both of these nodes from the graph
                        #     node_i.remove_from_graph_complete()
                        #     node_j.remove_from_graph_complete()

                        #     # Create a new parent node with these as children
                        #     first_seen = min(node_i.get_time_first_seen(), node_j.get_time_first_seen())
                        #     if first_seen == node_i.get_time_first_seen(): first_pose = node_i.get_first_pose()
                        #     else: first_pose = node_j.get_first_pose()
                        #     inferred_parent_node = GraphNode(self.root_node.request_new_ID(), None, [], np.zeros((0, 3), dtype=np.float64), 
                        #                                      [node_i, node_j], first_seen, self.times[-1], self.times[-1], first_pose, self.poses[-1])
                            
                        #     # Tell the children who their new parent is
                        #     node_i.set_parent(inferred_parent_node)
                        #     node_j.set_parent(inferred_parent_node)

                        #     # Add this inferred node back to the graph
                        #     logger.info(f"[dark_orange3]Parent Inferred[/dark_orange3]: {inferred_parent_node.get_id()} from Node {node_i.get_id()} and Node {node_j.get_id()}")
                        #     self.add_new_node_to_graph(inferred_parent_node, only_leaf=True)
                        #     logger.info(f"Number of points in new Inferred Node {inferred_parent_node.get_id()}: {inferred_parent_node.get_point_cloud().shape[0]}")

                        #     # Break out of double-nested loop to reset iterators
                        #     merge_occured = True
                        #     break

                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break
            
            # Restart next iteration if merge already occured
            if merge_occured:
                continue

            # Iterate through pairs of parent-child relationships (to check if they should merge)
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i < j and node_i.is_parent_or_child(node_j):

                        pass
                        
                        # ========== Parent-Child Semantic Merging ==========

                        # # If either is the root node, skip as this shouldn't really merge with any children
                        # if node_i.is_RootGraphNode() or node_j.is_RootGraphNode():
                        #     continue

                        # # Calculate Semantic Similarity
                        # sem_con = SceneGraph3D.semantic_consistency(node_i.get_weighted_semantic_descriptor(), node_j.get_weighted_semantic_descriptor())

                        # # If semantic embedding is enough for assocation, just merge the child into the parent
                        # if sem_con < self.min_sem_con_for_association:

                        #     # Get which node is the parent and which is the child
                        #     if node_i.is_parent(node_j): 
                        #         parent_node = node_i
                        #         child_node = node_j
                        #     else: 
                        #         parent_node = node_j
                        #         child_node = node_i

                        #     # Merge the child into the parent
                        #     logger.info(f"[green1]Parent-Child Semantic Merge[/green1]: Merging Node {node_i.get_id()} into Node {node_j.get_id()} and popping off graph")
                        #     parent_node.merge_child_with_self(child_node)

                        #     # Make sure nothing wierd has happened to point clouds, though I don't believe this should do anything
                        #     #self.resolve_overlapping_point_clouds()
                        
                        #     # Break out of double-nested loop to reset iterators
                        #     merge_occured = True
                        #     break
                
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
                child.remove_from_graph_complete()
                self.inactive_nodes.append(child)

    def visualize_2D(self, filename: str = None):
        import networkx as nx

        G = nx.DiGraph()
        node_colors = {}

        def add_node_recursive(node, parent_id=None):
            node_id = node.get_id()
            label = getattr(node, "get_label", lambda: f"{node_id}")()
            G.add_node(node_id, label=label)
            node_colors[node_id] = 'skyblue'
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            for child in node.get_children():
                G.add_edge(child.get_id(), node_id)
            for child in node.get_children():
                add_node_recursive(child, node_id)

        add_node_recursive(self.root_node)

        if len(G.nodes) == 0:
            print("Graph is empty! Nothing to plot.")
            return

        current_nodes_set = set(G.nodes)
        if self.G is None or current_nodes_set != self.last_nodes_set:
            # Using 'dot' layout from graphviz is often better for hierarchy
            try:
                self.pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except:
                # fallback layout
                self.pos = nx.spring_layout(G, iterations=20)
            self.last_nodes_set = current_nodes_set

        self.G = G
        self.node_colors = node_colors

        labels = nx.get_node_attributes(G, 'label')

        self.ax.clear()

        nx.draw(
            G, self.pos, ax=self.ax, with_labels=True, labels=labels,
            node_size=1500,
            node_color=[node_colors[n] for n in G.nodes()],
            font_size=8, font_color='black', arrows=True
        )

        self.ax.set_title("SceneGraph3D Structure")
        self.ax.margins(0.1)  # Add some margin so nodes are not cut off
        self.ax.axis('off')   # Hide axes for cleaner look

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save without cutoff

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to update the figure
        
    def save_graph_to_file(self, file_path: str):
        with open(file_path, 'wb') as file:
            pickle.dump(self.root_node, file)