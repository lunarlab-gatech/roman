from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .graph_node import GraphNode, wordnetWrapper
from .hull_methods import find_point_overlap_with_hulls, convex_hull_geometric_overlap, shortest_dist_between_convex_hulls
from .logger import logger
from ..map.observation import Observation
import multiprocessing
import numpy as np
import os
from ..params.data_params import ImgDataParams
from .rerun_wrapper import RerunWrapper
from roman.map.map import ROMANMap
from roman.object.segment import Segment
from roman.params.fastsam_params import FastSAMParams
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked
from .word_net_wrapper import WordWrapper, WordListWrapper

multiprocessing.set_start_method("spawn", force=True)

class SceneGraph3D():

    # Requirement for an observation to be associated with a current graph node or for two nodes to be merged.
    min_iou_for_association = 0.25

    # Used for "Nearby Node Semantic Merging" and "Parent-Child Semantic Merging" if enabled
    min_sem_con_for_association = 0.8

    # Threshold to determine if two convex hulls are overlapping in resolve_overlapping_convex_hulls()
    iou_threshold_overlapping_obj = 0.2
    enc_threshold_overlapping_obj = 0.2

    # Ratio of distance to object volume thresholds
    ratio_dist2length_threshold_nearby_node_semantic_merge = 0.025
    ratio_dist2length_threshold_shared_holonym = 0.05
    ratio_dist2length_threshold_holonym_meronym = 0.05

    # Ratio of detected relationship weight vs. previous total weight
    ratio_relationship_weight_2_total_weight = 2

    # If a high-level node goes this long since it was first seen, inactivate
    max_t_active_for_node = 15 # seconds

    # If we travel a significant distance from the first camera pose where this object was seen, inactivate
    max_dist_active_for_node = 10 # meters

    @typechecked
    def __init__(self, _T_camera_flu: np.ndarray, fastsam_params: FastSAMParams, headless: bool = True):

        # Node that connects all highest-level objects together for implementation purposes
        self.root_node: GraphNode = GraphNode.create_node_if_possible(0, None, [], np.zeros((0, 3), dtype=np.float64), 
                                                                      [], 0, 0, 0, np.empty(0), np.empty(0), is_RootGraphNode=True)
        
        # List of high-level nodes that have been inactivated
        self.inactive_nodes: list[GraphNode] = []

        # Keeps track of current time so we can keep track of when we are updated.
        self.times: list[float] = []

        # Keeps track of the current pose so we know where we are
        self.poses: list[np.ndarray] = []

        # Track FLU pose wrt camera frame
        self.pose_FLU_wrt_Camera = _T_camera_flu

        # Create the visualization
        self.rerun_viewer = RerunWrapper(enable=not headless, fastsam_params=fastsam_params)

        # Dictionaries to cache results of calculations for speed
        self.overlap_dict: defaultdict = defaultdict(lambda: defaultdict(lambda: None))
        self.shortest_dist_dist: defaultdict = defaultdict(lambda: defaultdict(lambda: None))

        # TODO: Are there cases where we don't want to call remove from graph complete, as we want to retire the node instead?

    @typechecked
    def len(self) -> int:
        return self.root_node.get_number_of_nodes()

    @typechecked
    def update(self, time: float | np.longdouble, pose: np.ndarray, observations: list[Observation], img: np.ndarray, depth_img: np.ndarray, img_data_params: ImgDataParams, seg_img: np.ndarray):
        
        logger.debug(f"SceneGraph3D update called with {len(observations)} observations")

        # Make sure that time ends up as a float
        time = float(time)

        # Update current time in all nodes and self
        self.times.append(time)
        self.root_node.update_curr_time(self.times[-1])

        # Set the current pose (in FLU frame) in all nodes and self
        self.poses.append(pose @ self.pose_FLU_wrt_Camera)
        self.root_node.update_curr_pose(self.poses[-1])

        # Convert each observation into a node (if possible)
        nodes: list[GraphNode] = []
        node_to_obs_mapping = {}
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {}
            for i, obs in enumerate(observations):
                futures[executor.submit(SceneGraph3D.convert_observation_to_node, i, self.root_node.request_new_ID(), obs, self.times[-1], self.poses[-1])] = i

            for future in as_completed(futures):
                result = future.result()
                if result:
                    new_node, i = result
                    nodes.append(new_node)
                    node_to_obs_mapping[new_node.get_id()] = i

        logger.info(f"Called with {len(nodes)} valid observations")

        # Run operations that require at least one input node
        associated_pairs: list[tuple] = []
        if len(nodes) > 0:
            
            # Associate new nodes with previous nodes in the graph
            associated_pairs = self.hungarian_assignment(nodes)
            if len(associated_pairs) > 0:
                logger.info(f"[dark_blue]Association Merges[/dark_blue]: {len(associated_pairs)} new nodes successfully associated")

            # Parallelize the node association merging
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {}

                # Merge associated pairs
                for i, new_node in enumerate(nodes):
                    for j, node in enumerate(self.root_node):

                        # See if this is a match
                        for pair in associated_pairs:
                            if new_node.get_id() == pair[0] and node.get_id() == pair[1]:

                                # If so, update node with observation information
                                futures[executor.submit(node.merge_with_observation, new_node.get_point_cloud(), 
                                        new_node.get_semantic_descriptors())] = i
        
                # Wait for jobs to finish and update associated pairs
                for future in as_completed(futures):
                    result = future.result()

            self.rerun_viewer.update(self.root_node, self.times[-1],  img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

            # Add the remaining unassociated valid_obs as nodes to the scene graph
            associated_obs_indices = [x[0] for x in associated_pairs]
            for node in nodes:
                if node.get_id() not in associated_obs_indices:
                    new_id = self.add_new_node_to_graph(node)
                    associated_pairs.append((new_id, new_id))
            
            # Update the viewer
            self.rerun_viewer.update(self.root_node, self.times[-1],  img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        # Update the viewer
        # self.rerun_viewer.update(self.root_node, self.times[-1], img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        # Run merging operations
        self.association_merges()

        # Update the viewer
        # self.rerun_viewer.update(self.root_node, self.times[-1], img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        # self.parent_child_semantic_merges(use_wordnet=True)

        # Update the viewer
        # self.rerun_viewer.update(self.root_node, self.times[-1], img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        # self.nearby_node_semantic_merges(use_wordnet=True)

        # Update the viewer
        # self.rerun_viewer.update(self.root_node, self.times[-1], img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        # self.holonym_meronym_relationship_inference()
        
        # Update the viewer
        # self.rerun_viewer.update(self.root_node, self.times[-1], img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        # self.shared_holonym_relationship_inference()

        # TODO: After merging, update the associations and node_to_obs mapping so colors carry over

        # Resolve any overlapping point clouds (TODO: Do we even need this?)
        # TODO: This seems to be causing bugs currently
        # self.resolve_overlapping_convex_hulls()

        # Run node retirement
        self.node_retirement()

        # Update the viewer
        self.rerun_viewer.update(self.root_node, self.times[-1], img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

    @typechecked
    def hungarian_assignment(self, new_nodes: list[GraphNode]) -> list[tuple]:
        """
        Associates new nodes with nodes in this scene graph by solving as 
        a linear sum assignment problem via the Hungarian method.
        """

        # Get number of observations and nodes
        num_new = len(new_nodes)
        num_nodes = self.len()

        # Setup score matrix
        scores = np.zeros((num_new, num_nodes))

        # Iterate through each new node
        for i, new_node in enumerate(new_nodes):
            # Calculate a similarity score for every node in the graph
            for j, node in enumerate(self.root_node):

                # If this is a root node, assign a score to practically disable association and skip
                if node.is_RootGraphNode():
                    scores[i,j] = 1e9
                    continue
                
                # Calculate IOU
                iou, _, _ = self.convex_hull_overlap_cached(node, new_node)

                # Check if it passes minimum requirements for association
                logger.debug(f"Currently comparing Node {new_node.get_id()} to Node {node.get_id()} with IOU: {iou}")
                if self.pass_minimum_requirements_for_association(iou):
                    # Negative as hungarian algorithm tries to minimize cost
                    scores[i,j] = -iou
                else:
                    # If fail (or root node), then give a very large number to practically disable association
                    scores[i,j] = 1e9
                    
        # Augment cost to add option for no associations
        hungarian_cost = np.concatenate([ np.concatenate([scores, np.ones(scores.shape)], axis=1), np.ones((scores.shape[0], 2*scores.shape[1]))], axis=0)
        row_ind, col_ind = linear_sum_assignment(hungarian_cost)

        pairs = []
        for idx1, idx2 in zip(row_ind, col_ind):
            # state and measurement associated together
            if idx1 < num_new and idx2 < num_nodes:
                pairs.append((idx1, idx2))

        # Convert indices from index of iteration in node ids
        new_pairs = []
        for pair in pairs:
            for i, new_node in enumerate(new_nodes):
                for j, node in enumerate(self.root_node):
                    if i == pair[0] and j == pair[1]:
                        new_pairs.append((new_node.get_id(), node.get_id()))
                        break

        return new_pairs
    
    @typechecked
    def convex_hull_overlap_cached(self, a: GraphNode, b: GraphNode) -> tuple[float, float, float]:
        """ Wrapper around convex_hull_geometric_overlap that caches results for reuse. """

        # Rearrange so that a is the node with the smaller id
        if a.get_id() > b.get_id():
            temp = a
            a = b
            b = temp
        assert a.get_id() < b.get_id()

        # Pull value from dictionary 
        result: tuple[float, float, float] | None = self.overlap_dict[a.get_id()][b.get_id()]

        # Calculate (or recalculate) the overlap if necessary
        if result is None or a._redo_convex_hull_geometric_overlap or b._redo_convex_hull_geometric_overlap:
            result: tuple[float, float, float] = convex_hull_geometric_overlap(a.get_convex_hull(), b.get_convex_hull())
            self.overlap_dict[a.get_id()][b.get_id()] = result

            # Tell nodes that we've updated our cache with their latest info
            a._redo_convex_hull_geometric_overlap = False
            b._redo_convex_hull_geometric_overlap = False

        return result
    
    @typechecked
    def shortest_dist_between_hulls_cached(self, a: GraphNode, b: GraphNode) -> float:
        """ Wrapper around convex_hull_geometric_overlap that caches results for reuse. """

        # Rearrange so that a is the node with the smaller id
        if a.get_id() > b.get_id():
            temp = a
            a = b
            b = temp
        assert a.get_id() < b.get_id()

        # Pull value from dictionary 
        result: float | None = self.shortest_dist_dist[a.get_id()][b.get_id()]

        # Calculate (or recalculate) the overlap if necessary
        if result is None or a._redo_shortest_dist_between_convex_hulls or b._redo_shortest_dist_between_convex_hulls:
            result: float = shortest_dist_between_convex_hulls(a.get_convex_hull(), b.get_convex_hull())
            self.shortest_dist_dist[a.get_id()][b.get_id()] = result

            # Tell nodes that we've updated our cache with their latest info
            a._redo_shortest_dist_between_convex_hulls = False
            b._redo_shortest_dist_between_convex_hulls = False

        return result
        
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
    def pass_minimum_requirements_for_association(self, iou: float) ->  bool:
        """ Currently, only use geometric information for association as drift should
            have no negative impact in small trajectories. """

        # Make sure inputs are within required bounds
        SceneGraph3D.check_within_bounds(iou, (0, 1))

        # Check if within thresholds for association
        return bool(iou >= self.min_iou_for_association)

    @typechecked
    def resolve_overlapping_convex_hulls(self):
        # TODO: Should this be reworked? I've seen this triggered when two objects were the same
        # but didn't have enough semantic similarity for a semantic merge.

        # Iterate through entire graph until no overlaps with shared points are detected
        change_occured = True
        while change_occured:

            # Track if we need to loop again
            change_occured =  False

            # Iterate through each pair of nodes that aren't an ascendent or descendent with each other
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i < j and not node_i.is_descendent_or_ascendent(node_j):

                        # If the nodes are obviously seperated, no point in checking.
                        if not self.check_if_nodes_are_somewhat_nearby(node_i, node_j):
                            continue

                        # Calculate the geometric overlap between them
                        iou, enc_i, enc_j = self.convex_hull_overlap_cached(node_i, node_j)

                        # If this is above a threshold, then we've found an overlap
                        if iou > self.iou_threshold_overlapping_obj or enc_i > self.enc_threshold_overlapping_obj \
                            or enc_j > self.enc_threshold_overlapping_obj:
                            
                            # Get merged point clouds from both nodes
                            pc_i = node_i.get_point_cloud()
                            pc_j = node_j.get_point_cloud()
                            pc_merged = np.concatenate((pc_i, pc_j), dtype=np.float64)

                            # Find all points in this overlap region
                            contain_masks = find_point_overlap_with_hulls(pc_merged, 
                                                    [node_i.get_convex_hull(), node_j.get_convex_hull()])
                            num_mask_assignments = np.sum(contain_masks, axis=0)
                            overlaps = np.where(num_mask_assignments > 1)[0]
                            pc_overlap = pc_merged[overlaps,:]

                            # If there is at least four points in this region 
                            if len(pc_overlap) >= 4:

                                # Double check this isn't the same node
                                assert node_i.get_id() != node_j.get_id(), f"Same node {node_i.get_id()} referred to as child of two parents!"

                                # Remove these points from their parents
                                logger.info(f"Node_i: {node_i.get_id()}")
                                logger.info(f"Node_j: {node_j.get_id()}")
                                node_i.remove_points_complete(pc_overlap)
                                node_j.remove_points_complete(pc_overlap)

                                # Get the merged semantic descriptors and drop weight by factor of 10
                                # TODO: Test this section manually
                                descriptors = node_i.get_semantic_descriptors()
                                descriptors += node_j.get_semantic_descriptors()
                                for i in range(len(descriptors)):
                                    embedding, volume = descriptors[i]
                                    descriptors[i] = (embedding, volume)
                      
                                # Try to add the new observation to the graph
                                logger.info(f"Adding overlap region of {len(pc_overlap)} points as observation to graph...")
                                new_node: GraphNode | None  = self.convert_overlap_to_node(pc_overlap, descriptors)
                                if new_node is not None:
                                    logger.info(f"[bright_red]Overlap Detected:[/bright_red] Between Node {node_i.get_id()} and Node {node_j.get_id()}, adding as Node {new_node.get_id()} to graph.")
                                    new_id = self.add_new_node_to_graph(new_node, only_leaf=True)
                                else:
                                    logger.info(f"[bright_red]Overlap Detected:[/bright_red] Between Node {node_i.get_id()} and Node {node_j.get_id()}, discarding...")

                                # TODO: Lu recommended that we maybe just try throwing this away instead. Implement this!

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
    def convert_overlap_to_node(self, point_cloud: np.ndarray, 
                                semantic_descriptors: list[tuple[np.ndarray, float]]) -> GraphNode | None:

        # Create a new node for this observation
        new_node: GraphNode | None = GraphNode.create_node_if_possible(self.root_node.request_new_ID(), None, [], 
                                        point_cloud, [],  self.times[-1], 
                                        self.times[-1], self.times[-1], 
                                        self.poses[-1], self.poses[-1])
        if new_node is None: return None # Node creation failed
        
        # Add the descriptors to the node
        new_node.add_semantic_descriptors(semantic_descriptors)
        return new_node
    
    @typechecked
    def convert_observation_to_node(obs_idx: int, new_ID: int, obs: Observation, time: float, pose: np.ndarray) -> tuple[GraphNode, int] | None:
        """Attempts to create a node from an observation, returns None if it fails."""

        # Create a new node for this observation
        new_node: GraphNode | None = GraphNode.create_node_if_possible(new_ID, None, [], 
                        obs.transformed_points, [], time, time, time, pose, pose)
        if new_node is None: return None # Node creation failed

        # Add the descriptor to the node
        if obs.clip_embedding is not None:
            new_node.add_semantic_descriptors([(obs.clip_embedding, new_node.get_volume())])
        return new_node, obs_idx # Returns node and observation idx

    @typechecked
    def add_new_node_to_graph(self, node: GraphNode, only_leaf=False) -> int:

        # Make sure the node passed to the graph is actually valid
        if not node._class_method_creation_success:
            raise RuntimeError("Cannot add_new_node_to_graph when node is invalid!")

        # Keep track of the current likeliest node
        best_likelihood_score = -np.inf
        best_likelihood_parent = None

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
            if pos_parent_node.is_RootGraphNode():
                # Objects at least half encompassed by other node should be assigned there, no matter how small.
                parent_iou, parent_encompassment = 0.0, 0.5 
            else:
                parent_iou, _, parent_encompassment = self.convex_hull_overlap_cached(pos_parent_node, node)

            # Add similarities for geometric overlaps with children
            children_iou, children_enclosure = [], []
            if not only_leaf:
                for child in pos_parent_node.get_children():
                    iou, child_enc, _ = self.convex_hull_overlap_cached(child, node)
                    children_iou.append(iou)
                    children_enclosure.append(child_enc)

            # Calculate the final likelihood score
            score = self.calculate_best_likelihood_score(children_iou, children_enclosure, parent_iou, parent_encompassment, only_leaf=only_leaf)
            
            # If this is the best score so far, keep it
            logger.debug(f"Scores: {parent_iou} {parent_encompassment}")
            logger.debug(f"Best Likelihood for Node {pos_parent_node.get_id()}: {score}")
            if score >= best_likelihood_score:
                best_likelihood_score = score
                best_likelihood_parent = pos_parent_node

            # After checking this node, move to check the next one
            pos_parent_node, node_queue = next_loop(pos_parent_node, node_queue)

        # Extract new parent
        new_parent: GraphNode = best_likelihood_parent

        # Now, find the best subset of children, which comprises
        # of all children nodes with encompassment of 50% or more
        new_children: list[GraphNode] = []
        for child in new_parent.get_children():
            _, child_enc, _ = self.convex_hull_overlap_cached(child, node)
            if child_enc >= 0.5:
                new_children.append(child)

        # Place our already fully formed node into its spot
        self.place_node_in_graph(node, new_parent, new_children)

        # Return the id of the node just added
        return node.get_id()
    
    def place_node_in_graph(self, node: GraphNode, new_parent: GraphNode, new_children: list[GraphNode]):
        """ Helper method for placing node in graph once its relationships are already determined. """

        # Connect parent and node
        new_parent.add_child(node)
        node.set_parent(new_parent)
        
        # Connect children to node
        deleted_ids = []
        for child in new_children:
            deleted_ids = child.remove_from_graph_complete()
        for child in new_children:
            child.set_parent(None)
        node.add_children(new_children)
        for child in new_children:
            child.set_parent(node)

        # Print action
        if len(new_children) > 0:
            logger.info(f"[cyan]Added[/cyan]: Node {node.get_id()} added to graph as child of Node {new_parent.get_id()} and parent of Nodes {[c.get_id() for c in new_children]}")
        else:
            logger.info(f"[cyan]Added[/cyan]: Node {node.get_id()} added to graph as child of Node {new_parent.get_id()}")

        # Print any deleted nodes
        if len(deleted_ids) > 0:
            logger.info(f"[bright_red] Discard: [/bright_red] Node(s) {deleted_ids} removed as not enough remaining points with children removal.")

    @typechecked
    def calculate_best_likelihood_score(self, children_iou: list[float], children_enclosure: list[float], parent_iou: float, parent_encompassment: float, only_leaf: bool) -> float:

        # Make sure parent scores are within thresholds
        SceneGraph3D.check_within_bounds(parent_iou, (0, 1))
        SceneGraph3D.check_within_bounds(parent_encompassment, (0, 1))

        # If there is at least one child, calculate the best likelihood score using them
        expected_len = len(children_iou)
        best_child_score = 0.0
        if expected_len > 0 and not only_leaf: 
            # Assert each array is of the same length
            assert expected_len == len(children_enclosure)

            # Make sure each child value is within expected thresholds
            for i in range(len(children_iou)):
                SceneGraph3D.check_within_bounds(children_iou[i], (0, 1))
                SceneGraph3D.check_within_bounds(children_enclosure[i], (0, 1))

            # Convert arrays to numpy 
            children_iou: np.ndarray = np.array(children_iou, dtype=np.float32)
            children_enclosure: np.ndarray = np.array(children_enclosure, dtype=np.float32)

            # The best score will be the children tied for the maximum of their individual scores.
            # Thus, calculate individual scores for each child and choose the max.
            scores = children_iou + children_enclosure
            best_child_score = scores.max() + parent_iou + parent_encompassment

        # Calculate best score for being a child of root node with no children.
        # This assumes child IOU of 0, child encompassment of 1, and child sem sim of 0.5.
        best_alt_score = 1.0 + parent_iou + parent_encompassment
        
        # Return the best option
        if best_child_score > best_alt_score:
            logger.debug(f"Children Scores: {children_iou[scores.argmax()]} {children_enclosure[scores.argmax()]}")
            return best_child_score
        else:
            logger.debug(f"Children Scores: Default 1.0")
            return best_alt_score
    
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

    def association_merges(self):
        """ Checks for Association Merges using same minimum requirements used by Hungarian algorithm. """

        merge_occured = True
        while merge_occured:
            merge_occured = False

            # Iterate through each pair of nodes 
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i >= j: continue

                    # For those that aren't an ascendent or descendent with each other
                    if not node_i.is_descendent_or_ascendent(node_j):
                        
                        # Calculate IOU and Semantic Similarity
                        iou, _, _ = self.convex_hull_overlap_cached(node_i, node_j)
    
                        # See if they pass minimum requirements for association
                        if self.pass_minimum_requirements_for_association(iou):

                            # If so, merge nodes in the graph
                            logger.info(f"[dark_blue]Association Merge[/dark_blue]: Merging Node {node_i.get_id()} into Node {node_j.get_id()} and popping off graph")
                            merged_node = node_i.merge_with_node(node_j)
                            if merged_node is None:
                                logger.info(f"[bright_red]Merge Fail[/bright_red]: Resulting Node was invalid.")
                            else:
                                self.add_new_node_to_graph(merged_node, only_leaf=True)

                            # Remember to break out of double-nested loop to reset iterators
                            merge_occured = True
                            break
                                  
                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break

    def nearby_node_semantic_merges(self, use_wordnet=False):
        """ Checks for semantic merges between children of same parent. """

        # TODO: I wonder if using average distance would be better for 
        # being resistant to noise in our nearby children semantic merging. Maybe try this!

        merge_occured = True
        while merge_occured:
            merge_occured = False

            # Iterate through each pair of nodes 
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i >= j: continue
                        
                    # For those that are not ascendent or descendent with each other
                    if not node_i.is_descendent_or_ascendent(node_j):
                        # ========== Nearby Node Semantic Merge ==========

                        if not use_wordnet:

                            # Get shortest distance between the nodes
                            dist = self.shortest_dist_between_hulls_cached(node_i, node_j)

                            # Get longest line of either node
                            longest_line_node_i = node_i.get_longest_line_size()
                            longest_line_node_j = node_j.get_longest_line_size()
                            longest_line = np.mean([longest_line_node_i, longest_line_node_j])

                            # Get ratio of shortest distance to longest line (in other words, dist to object length)
                            dist_to_object_length = dist / longest_line

                            # Finally, calculate semantic consistency
                            sem_con = self.semantic_consistency(node_i.get_semantic_descriptor(),
                                                                node_j.get_semantic_descriptor())

                            # If ratio of shortest distance to object length is within threshold AND
                            # the semanatic embedding is close enough for association
                            if dist_to_object_length < self.ratio_dist2length_threshold_nearby_node_semantic_merge and \
                                sem_con > self.min_sem_con_for_association:

                                # If so, merge these two nodes in the graph
                                logger.info(f"[gold1]Nearby Node Sem Merge[/gold1]: Merging Node {node_j.get_id()} into Node {node_i.get_id()} and popping off graph")

                                merged_node = node_i.merge_with_node(node_j)
                                if merged_node is None: logger.info(f"[bright_red]Merge Fail[/bright_red]: Resulting Node was invalid.")
                                else: self.add_new_node_to_graph(merged_node, only_leaf=True)

                                # Break out of double-nested loop to reset iterators
                                merge_occured = True
                                break
                        
                        else:
                            
                            # Get words wrapped for each node
                            words_i = node_i.get_words()
                            words_j = node_j.get_words()

                            # Calculate Semantic Similarity
                            sem_con = SceneGraph3D.semantic_consistency(node_i.get_semantic_descriptor(), node_j.get_semantic_descriptor())

                            # If any of the words are the same word, check geometric info
                            if words_i == words_j or sem_con > self.min_sem_con_for_association:

                                # Get shortest distance between the nodes
                                dist = self.shortest_dist_between_hulls_cached(node_i, node_j)

                                # Get longest line of either node
                                longest_line_node_i = node_i.get_longest_line_size()
                                longest_line_node_j = node_j.get_longest_line_size()
                                longest_line = np.mean([longest_line_node_i, longest_line_node_j])

                                # Get ratio of shortest distance to longest line (in other words, dist to object length)
                                dist_to_object_length = dist / longest_line

                                if dist_to_object_length < self.ratio_dist2length_threshold_nearby_node_semantic_merge:
                                    # If so, merge these two nodes in the graph
                                    logger.info(f"[green3]Nearby Obj Sem Merge[/green3]: Merging Node {node_j.get_id()} into Node {node_i.get_id()} and popping off graph")

                                    merged_node = node_i.merge_with_node(node_j)
                                    if merged_node is None: logger.info(f"[bright_red]Merge Fail[/bright_red]: Resulting Node was invalid.")
                                    else: self.add_new_node_to_graph(merged_node, only_leaf=True)

                                    # Break out of double-nested loop to reset iterators
                                    merge_occured = True
                                    break
                    
                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break

    def parent_child_semantic_merges(self, use_wordnet: bool = False):
        """ Checks for semantic merges between parents and children. """

        merge_occured = True
        while merge_occured:
            merge_occured = False

            # Iterate through each pair of nodes 
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i >= j: continue
        
                    # For those in parent-child relationships
                    if node_i.is_parent_or_child(node_j):
              
                        # ========== Parent-Child Semantic Merging ==========
            
                        # If either is the root node, skip as this shouldn't really merge with any children
                        if node_i.is_RootGraphNode() or node_j.is_RootGraphNode():
                            continue
                            
                        if not use_wordnet:
                            # Calculate Semantic Similarity
                            sem_con = SceneGraph3D.semantic_consistency(node_i.get_semantic_descriptor(), node_j.get_semantic_descriptor())

                            # If semantic embedding is enough for assocation, just merge the child into the parent
                            if sem_con > self.min_sem_con_for_association:

                                # Get which node is the parent and which is the child
                                if node_i.is_parent(node_j): 
                                    parent_node = node_i
                                    child_node = node_j
                                else: 
                                    parent_node = node_j
                                    child_node = node_i

                                # Merge the child into the parent
                                logger.info(f"[green1]Parent-Child Semantic Merge[/green1]: Merging Node {child_node.get_id()} into Node {parent_node.get_id()}")
                                parent_node.merge_child_with_self(child_node)

                                # Break out of double-nested loop to reset iterators
                                merge_occured = True
                                break
                        else:
                            # Get words wrapped for each node
                            words_i = node_i.get_words()
                            words_j = node_j.get_words()

                            # Calculate Semantic Similarity
                            sem_con = SceneGraph3D.semantic_consistency(node_i.get_semantic_descriptor(), node_j.get_semantic_descriptor())

                            # If any of the words are the same word, merge them
                            if words_i == words_j or sem_con > self.min_sem_con_for_association:
                                
                                # Get which node is the parent and which is the child
                                if node_i.is_parent(node_j): 
                                    parent_node = node_i
                                    child_node = node_j
                                else: 
                                    parent_node = node_j
                                    child_node = node_i

                                # Merge the child into the parent
                                logger.info(f"[green1]Parent-Child Semantic Merge[/green1]: Merging Node {child_node.get_id()} into Node {parent_node.get_id()}")
                                parent_node.merge_child_with_self(child_node)

                                # Break out of double-nested loop to reset iterators
                                merge_occured = True
                                break
                        
                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break

    def holonym_meronym_relationship_inference(self):
        """ Detect parent-child relationships between non-ascendent/descendent nodes in the graph. """

        merge_occured = True
        while merge_occured:
            merge_occured = False

            # Iterate through each pair of nodes 
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i >= j: continue

                    # For those that are not ascendent or descendent with each other
                    if not node_i.is_descendent_or_ascendent(node_j):

                        # Get likeliest wrapped for each node
                        word_i: WordWrapper = node_i.get_words().words[0]
                        word_j: WordWrapper = node_j.get_words().words[0]

                        logger.debug(f"Words: {word_i.word} {word_j.word}")

                        # Get all holonyms/meronyms for words
                        word_i_meronyms: set[str] = node_i.get_all_meronyms(2)
                        word_j_meronyms: set[str] = node_j.get_all_meronyms(2)
                        word_i_holonyms: set[str] = node_i.get_all_holonyms(True, 2)
                        word_j_holonyms: set[str] = word_j.get_all_holonyms(True, 2)

                        logger.debug(f"All Holonyms: {word_i_holonyms} {word_j_holonyms}")

                        # Check if there is a Holonym-Meronym relationship
                        if word_i.word in word_j_meronyms or word_i.word in word_j_holonyms or \
                           word_j.word in word_i_meronyms or word_j.word in word_i_holonyms:
                            
                            logger.debug("Relationship found, close enough?")

                             # Get shortest distance between the nodes
                            dist = self.shortest_dist_between_hulls_cached(node_i, node_j)

                            # Get longest line of either node
                            longest_line_node_i = node_i.get_longest_line_size()
                            longest_line_node_j = node_j.get_longest_line_size()
                            longest_line = np.mean([longest_line_node_i, longest_line_node_j])

                            # Get ratio of shortest distance to longest line (in other words, dist to object length)
                            dist_to_object_length = dist / longest_line

                            # If they are close enough
                            if dist_to_object_length < self.ratio_dist2length_threshold_holonym_meronym:
                                
                                # Find the meronym
                                if word_i.word in word_j_meronyms or word_j.word in word_i_holonyms:
                                    node_meronym = node_i
                                    node_holonym = node_j
                                    word_meronym = word_i
                                    word_holonym = word_j
                                else:
                                    node_meronym = node_j
                                    node_holonym = node_i
                                    word_meronym = word_j
                                    word_holonym = word_i

                                # Move the meronym to be child of holonym
                                deleted_ids = node_meronym.remove_from_graph_complete()
                                node_meronym.set_parent(None)
                                node_holonym.add_child(node_meronym)
                                node_meronym.set_parent(node_holonym)

                                logger.info(f"[dark_goldenrod]Holonym-Meronym Relationship Detected[/dark_goldenrod]: Node {node_meronym.get_id()} as meronym of {node_holonym.get_id()}")

                                # Now that we've detected this relationship, we want to strengthen our
                                # believed embeddings towards these word for holonym (since adding meronym
                                # will skew holonym towards that word).
                                                                
                                holonym_emb: np.ndarray = wordnetWrapper.get_embedding_for_word(word_holonym.word)
                                total_weight = node_holonym.get_total_weight_of_semantic_descriptors()
                                node_holonym.add_semantic_descriptors([(holonym_emb, total_weight * self.ratio_relationship_weight_2_total_weight)])

                                # Print any deleted nodes
                                if len(deleted_ids) > 0:
                                    logger.info(f"[bright_red] Discard: [/bright_red] Node(s) {deleted_ids} removed as not enough remaining points with children removal.")

                                # Break out of the loop
                                merge_occured = True
                                break
                
                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break


    def shared_holonym_relationship_inference(self):
        """ Detect and add higher level objects to the graph. """

        merge_occured = True
        while merge_occured:
            merge_occured = False

            # Iterate through each pair of nodes that are children of the root node
            for i, node_i in enumerate(self.root_node.get_children()):
                for j, node_j in enumerate(self.root_node.get_children()):
                    if i >= j: continue

                    # Get likeliest wrapped for each node
                    word_i: WordWrapper = node_i.get_words().words[0]
                    word_j: WordWrapper = node_j.get_words().words[0]

                    # Get all holonyms for words
                    word_i_holonyms_pure: set[str] = node_i.get_all_holonyms(False)
                    word_j_holonyms_pure: set[str] = node_j.get_all_holonyms(False)

                    # TODO: Do an option where we can do upwards but excluding holonyms from a holonym/meronym relationship.

                    # Check if there is a shared holonym
                    shared_holonyms = sorted(set.intersection(word_i_holonyms_pure, word_j_holonyms_pure))
                    if len(shared_holonyms) > 0:

                        # Get shortest distance between the nodes
                        dist = self.shortest_dist_between_hulls_cached(node_i, node_j)

                        # Get longest line of either node
                        longest_line_node_i = node_i.get_longest_line_size()
                        longest_line_node_j = node_j.get_longest_line_size()
                        longest_line = np.mean([longest_line_node_i, longest_line_node_j])

                        # Get ratio of shortest distance to longest line (in other words, dist to object length)
                        dist_to_object_length = dist / longest_line

                        # If they are close enough
                        if dist_to_object_length < self.ratio_dist2length_threshold_shared_holonym:

                            # Calculate the first seen time as earliest from the two nodes
                            first_seen = min(node_i.get_time_first_seen(), node_j.get_time_first_seen())

                            # Also calculate the first pose
                            if first_seen == node_i.get_time_first_seen(): first_pose = node_i.get_first_pose()
                            else: first_pose = node_j.get_first_pose()

                            # Create the holonym node
                            holonym: GraphNode | None = GraphNode.create_node_if_possible(self.root_node.request_new_ID(), 
                                                            self.root_node, [], np.zeros((0, 3), dtype=np.float64), 
                                                            [node_i, node_j], first_seen, self.times[-1], self.times[-1], 
                                                            first_pose, self.poses[-1], is_RootGraphNode=False)

                            if holonym is not None:
                                # TODO: Maybe search to see if other nodes are in the overlap space between
                                # the two children which should also be part of the set?

                                self.place_node_in_graph(holonym, self.root_node, [node_i, node_j])

                                # Update embedding so that it matches the word most of all!
                                holonym_emb: np.ndarray = wordnetWrapper.get_embedding_for_word(shared_holonyms[0])
                                total_weight = holonym.get_total_weight_of_semantic_descriptors()
                                holonym.add_semantic_descriptors([(holonym_emb, total_weight * self.ratio_relationship_weight_2_total_weight)])

                                logger.info(f"[gold1]Shared Holonym Detected[/gold1]: Node {holonym.get_id()} with words {shared_holonyms} {holonym.get_words()} from children with words {word_i.word} and {word_j.word}")

                                merge_occured = True
                                break
                            else:
                                raise RuntimeError("Detected Holonym is not a valid GraphNode!")

                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break
    
    def node_retirement(self, retire_everything=False, delete_only_seen_once=True):
        # Iterate only through the direct children of the root node
        retired_ids = []
        for child in self.root_node.get_children()[:]: # Create shallow copy so removing doesn't break loop

            # If the time the child was first seen was too long ago OR we've move substancially since then 
            # OR we are retiring everything, Inactivate this node and descendents
            if self.times[-1] - child.get_time_first_seen_recursive() > self.max_t_active_for_node \
                or np.linalg.norm(self.poses[-1][:3,3] - child.get_first_pose_recursive()[:3,3]) > self.max_dist_active_for_node \
                or retire_everything:

                # Pop this child off of the root node and put in our inactive nodes
                retired_ids += [child.get_id()]
                retired_ids += child.remove_from_graph_complete()

                # Run DBSCan right before we finish for cleanup
                child.update_point_cloud(np.zeros((0, 3), dtype=np.float64), run_dbscan=True)
                self.inactive_nodes.append(child)

        # Delete nodes that were seen last frame but not this one
        deleted_ids = []
        if delete_only_seen_once:
            for node in self.root_node:
                if node.get_num_sightings() == 1:
                    if node.get_time_first_seen() != self.times[-1]:

                        # Pop just this node off the graph, reconnect children back to our parent
                        deleted_ids += [node.get_id()]
                        deleted_ids += node.remove_from_graph_complete(keep_children=False)

        if len(retired_ids) > 0:
            logger.info(f"[dark_magenta]Node Retirement[/dark_magenta]: {len(retired_ids)} nodes retired, including {retired_ids}.")
        if len(deleted_ids) > 0:
            logger.info(f"[magenta]Node Deletion[/magenta]: {len(deleted_ids)} nodes deleted after being seen only once, including {deleted_ids}.")

    def get_roman_map(self) -> ROMANMap:
        """
        Convert this SceneGraph3D into a ROMANMap of seperate object segments (no meronomy) to see how ROMAN's alignment
        algorithm works with our objects that have had extra merging operations performed between them.
        """

        # Convert all graph nodes to segments
        segment_map: list[Segment] = []
        for top_node in self.inactive_nodes:
            for node in top_node:
                segment_map.append(node.to_segment())

        # Return the ROMANMap
        return ROMANMap(
            segments=segment_map,
            trajectory=self.poses,
            times=self.times,
            poses_are_flu=True
        )