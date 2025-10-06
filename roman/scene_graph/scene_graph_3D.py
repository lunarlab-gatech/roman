from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from .graph_node import GraphNode, wordnetWrapper
from .hull_methods import find_point_overlap_with_hulls, convex_hull_geometric_overlap, shortest_dist_between_convex_hulls
from ..logger import logger
from ..map.observation import Observation
import multiprocessing
import numpy as np
import os
from ..params.data_params import ImgDataParams
from ..params.scene_graph_3D_params import SceneGraph3DParams, GraphNodeParams
import pickle
from .rerun_wrapper import RerunWrapper
from robotdatapy.data.img_data import CameraParams
from robotdatapy.transform import transform
from roman.map.map import ROMANMap
from roman.object.segment import Segment
from roman.params.fastsam_params import FastSAMParams
from roman.params.system_params import SystemParams
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked
from typing import Any
from .word_net_wrapper import WordWrapper, WordListWrapper

multiprocessing.set_start_method("spawn", force=True)

class SceneGraph3D():

    @typechecked
    def __init__(self, params: SystemParams, camera_params: CameraParams, _T_camera_flu: np.ndarray):

        # Save parameters 
        self.system_params: SystemParams = params
        self.params: SceneGraph3DParams = params.scene_graph_3D_params
        GraphNode.params = params.graph_node_params
        GraphNode.camera_params = camera_params

        # Node that connects all highest-level objects together for implementation purposes
        self.root_node: GraphNode = GraphNode.create_node_if_possible(-1, None, [], None, 0, np.zeros((0, 3), dtype=np.float64), 
                                                                      [], 0, 0, 0, np.empty(0), np.empty(0), np.empty(0), is_RootGraphNode=True)

        # Keeps track of current time so we can keep track of when we are updated.
        self.times: list[float] = []

        # Keeps track of the current pose so we know where we are
        self.poses: list[np.ndarray] = []

        # Track FLU pose wrt camera frame
        self.pose_FLU_wrt_Camera = _T_camera_flu

        # Create the visualization
        self.rerun_viewer = RerunWrapper(enable=self.params.enable_rerun_viz, fastsam_params=params.fastsam_params)

        # Dictionaries to cache results of calculations for speed
        self.overlap_dict: defaultdict = defaultdict(lambda: defaultdict(lambda: None))
        self.shortest_dist_dict: defaultdict = defaultdict(lambda: defaultdict(lambda: None))

        # TODO: Are there cases where we don't want to call remove from graph complete, as we want to retire the node instead?
        # TODO: I might need a mechanism to merge hypernyms/hyponyms in the graph.

        # TODO: Maybe use node average size instead of node longest size


    @typechecked
    def update(self, time: float | np.longdouble, pose: np.ndarray, observations: list[Observation], img: np.ndarray, depth_img: np.ndarray, img_data_params: ImgDataParams, seg_img: np.ndarray):
        
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

        # Sort nodes after parallel operations for determinism
        nodes = sorted(nodes, key=lambda node: node_to_obs_mapping[node.get_id()])

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
                                                        new_node.get_semantic_descriptor(), new_node.obs_descriptor)] = i
                                
        
                # Wait for jobs to finish and update associated pairs
                for future in as_completed(futures):
                    result = future.result()

                    # TODO: Update associated pairs?

            self.rerun_viewer.update(self.root_node, self.times[-1],  img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

            # Update the nodes segment statuses
            self.update_segment_statuses()

            # Forfeit all new node ids
            for node in nodes:
                self.root_node.forfeit_ID(node.get_id())

            # Add the remaining unassociated valid_obs as nodes to the scene graph
            associated_obs_indices = [x[0] for x in associated_pairs]
            for i, node in enumerate(nodes):
                if node.get_id() not in associated_obs_indices:

                    # Downsample/remove outliers if requested
                    if self.params.downsample_and_remove_outliers_after_hungarian_for_new_nodes:
                        to_delete = node.update_point_cloud(np.zeros((0, 3), dtype=np.float64), downsample=True, remove_outliers=True)
                        if len(to_delete) > 0:
                            raise RuntimeError("Node is no longer valid after downsampling and removing outliers")

                    # Discard if there are no points
                    if node.get_num_points() == 0: continue

                    # Re-assign ids to the non-associated nodes (so they line up with ROMAN ids)
                    node.set_id(self.root_node.request_new_ID())

                    # Add the node to the graph
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

        if self.params.enable_synonym_merges:
            self.synonym_relationship_merging()

        # Update the viewer
        # self.rerun_viewer.update(self.root_node, self.times[-1], img=img, depth_img=depth_img, camera_pose=pose, img_data_params=img_data_params, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        if self.params.enable_meronomy_relationship_inference:
            self.holonym_meronym_relationship_inference()
            self.shared_holonym_relationship_inference()

        # TODO: After merging, update the associations and node_to_obs mapping so colors carry over

        # Resolve any overlapping point clouds (TODO: Do we even need this?)
        # TODO: This seems to be causing bugs currently
        if self.params.enable_resolve_overlapping_nodes:
            self.resolve_overlapping_convex_hulls()

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
        nodes_in_graph: list[GraphNode] = self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT) + \
                                          self.get_nodes_with_status(GraphNode.SegmentStatus.NURSERY)
        num_nodes = len(nodes_in_graph)

        # Setup score matrix
        scores = np.zeros((num_nodes, num_new))

        # Iterate through each new node
        for i, node in enumerate(nodes_in_graph):
            # Calculate a similarity score with every node in the graph
            for j, new_node in enumerate(new_nodes):
                
                # Calculate IOU
                iou, _, _ = self.geometric_overlap(node, new_node)

                # Check if it passes minimum requirements for association
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
            if idx1 < num_nodes and idx2 < num_new:
                pairs.append((idx1, idx2))

        # Convert indices from index of iteration in node ids
        new_pairs = []
        for pair in pairs:
            for i, node in enumerate(nodes_in_graph):
                for j, new_node in enumerate(new_nodes):
                    if i == pair[0] and j == pair[1]:
                        new_pairs.append((new_node.get_id(), node.get_id()))
                        break

        return new_pairs
    
    def geometric_overlap(self, a: GraphNode, b: GraphNode) -> tuple[float, float | None, float | None]:

        # Calculate geometric overlap using hulls
        result = (None, None, None)
        if self.params.use_convex_hull_for_iou:
            result: tuple[float, float, float] = convex_hull_geometric_overlap(a.get_convex_hull(), b.get_convex_hull())

        # Use Voxel Grid for IOU if specified
        if not self.params.use_convex_hull_for_iou:
            voxel_size = self.params.voxel_size_for_voxel_grid_iou
            grid_a = a.get_voxel_grid(voxel_size)
            grid_b = b.get_voxel_grid(voxel_size)
            if grid_a is None or grid_b is None:
                raise RuntimeError("One or more Voxel Grids are None!")
                voxel_iou = 0.0
            else:
                voxel_iou = grid_a.iou(grid_b)
            result = (voxel_iou, 0.5, 0.5) # With voxel grid, we have no way to calculate enclosure, so just guess.

            # TODO: Just return None and have calling methods deal when enclosure is None!

        return result   

    @staticmethod
    def _purge_node_calculations_from_dict(dict: defaultdict, node: GraphNode) -> None:
        """Remove all cached overlaps that involve the given node."""
        node_id = node.get_id()

        # Remove when node_id is the first key
        dict.pop(node_id, None)

        # Remove when node_id is the second key
        for outer_id, inner_dict in list(dict.items()):
            if node_id in inner_dict:
                inner_dict.pop(node_id, None)

    @typechecked
    def geometric_overlap_cached(self, a: GraphNode, b: GraphNode) -> tuple[float, float | None, float | None]:
        """ Wrapper around convex_hull_geometric_overlap that caches results for reuse. """

        # Rearrange so that a is the node with the smaller id
        swap_enclosures = False
        if a.get_id() > b.get_id():
            temp = a
            a = b
            b = temp
            swap_enclosures = True
        assert a.get_id() < b.get_id()

        # For each node, if their calculations need to be redone then wipe the dictionary
        if a._redo_convex_hull_geometric_overlap:
            SceneGraph3D._purge_node_calculations_from_dict(self.overlap_dict, a)
            a._redo_convex_hull_geometric_overlap = False
        if b._redo_convex_hull_geometric_overlap:
            SceneGraph3D._purge_node_calculations_from_dict(self.overlap_dict, b)
            b._redo_convex_hull_geometric_overlap = False

        # Pull value from dictionary 
        result: tuple[float, float, float] | None = self.overlap_dict[a.get_id()][b.get_id()]

        # Calculate (or recalculate) the overlap if necessary
        if result is None:
            # Calculate geometric overlap and save
            result: tuple[float, float | None, float | None] = self.geometric_overlap(a, b)
            self.overlap_dict[a.get_id()][b.get_id()] = result

        # Swap enclosure values if necessary
        if swap_enclosures:
            result = (result[0], result[2], result[1])

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

        # For each node, if their calculations need to be redone then wipe the dictionary
        if a._redo_shortest_dist_between_convex_hulls:
            SceneGraph3D._purge_node_calculations_from_dict(self.shortest_dist_dict, a)
            a._redo_shortest_dist_between_convex_hulls = False
        if b._redo_shortest_dist_between_convex_hulls:
            SceneGraph3D._purge_node_calculations_from_dict(self.shortest_dist_dict, b)
            b._redo_shortest_dist_between_convex_hulls = False

        # Pull value from dictionary 
        result: float | None = self.shortest_dist_dict[a.get_id()][b.get_id()]

        # Calculate (or recalculate) the overlap if necessary
        if result is None:
            result: float = shortest_dist_between_convex_hulls(a.get_convex_hull(), b.get_convex_hull())
            self.shortest_dist_dict[a.get_id()][b.get_id()] = result

        return result
    
    def shortest_dist_to_node_size_ratio(self, a: GraphNode, b: GraphNode) -> float:
        # Get shortest distance between the nodes
        dist = self.shortest_dist_between_hulls_cached(a, b)

        # Get average of longest lines of both node
        longest_line_a = a.get_longest_line_size()
        longest_line_b = b.get_longest_line_size()
        longest_line = np.mean([longest_line_a, longest_line_b])

        # Get ratio of shortest distance to longest line (in other words, dist to object length)
        return dist / longest_line
        
    @typechecked
    @staticmethod
    def semantic_consistency(a: np.ndarray| None, b: np.ndarray | None) -> float:
        # If either is none, then just assume neutral consistency
        if a is None or b is None:
            return 0.5

        # Normalize both embeddings (just in case they aren't already)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        # Calculate the cosine similarity
        cos_sim = np.dot(a, b)
        return np.clip(cos_sim, 0.0, 1.0)
    
    @typechecked
    def pass_minimum_requirements_for_association(self, iou: float) ->  bool:
        """ Currently, only use geometric information for association as drift should
            have no negative impact in small trajectories. """

        # Make sure inputs are within required bounds
        SceneGraph3D.check_within_bounds(iou, (0, 1))

        # Check if within thresholds for association
        return bool(iou >= self.params.min_iou_for_association)

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
                        iou, enc_i, enc_j = self.geometric_overlap_cached(node_i, node_j)

                        # If this is above a threshold, then we've found an overlap
                        if iou > self.params.iou_threshold_overlapping_obj or enc_i > self.params.enc_threshold_overlapping_obj \
                            or enc_j > self.params.enc_threshold_overlapping_obj:
                            
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
                                new_node: GraphNode | None = self.convert_overlap_to_node(pc_overlap, descriptors)
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
                                        point_cloud, [], None, 0, self.times[-1], 
                                        self.times[-1], self.times[-1], self.poses[-1],
                                        self.poses[-1], self.poses[-1])
        if new_node is None: return None # Node creation failed
        
        # Add the descriptors to the node
        new_node.add_semantic_descriptors(semantic_descriptors)
        raise NotImplementedError("Need to figure out how to do the line below!")
        new_node.add_semantic_descriptors_incremental()
        return new_node
    
    @typechecked
    def convert_observation_to_node(obs_idx: int, new_ID: int, obs: Observation, time: float, pose: np.ndarray) -> tuple[GraphNode, int] | None:
        """Attempts to create a node from an observation, returns None if it fails."""

        # Create a new node for this observation
        new_node: GraphNode | None = GraphNode.create_node_if_possible(new_ID, None, [], None, 0,
                        obs.transformed_points, [], time, time, time, pose, pose, pose)
        if new_node is None: return None # Node creation failed

        # Add the descriptor to the node
        if obs.clip_embedding is not None:
            new_node.add_semantic_descriptors([(obs.clip_embedding, new_node.get_volume())])
            new_node.add_semantic_descriptors_incremental(obs.clip_embedding, 1)
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
            children = pos_parent_node.get_children()
            node_queue += children

            # Add similarities for geometric overlaps with parent
            if pos_parent_node.is_RootGraphNode():
                # Objects at least half encompassed by other node should be assigned there, no matter how small.
                parent_iou, parent_encompassment = 0.0, 0.5 
            else:
                parent_iou, _, parent_encompassment = self.geometric_overlap_cached(pos_parent_node, node)

            # Add similarities for geometric overlaps with children
            children_iou, children_enclosure = [], []
            for child in children:
                iou, child_enc, _ = self.geometric_overlap_cached(child, node)
                children_iou.append(iou)
                children_enclosure.append(child_enc)

            # Calculate the final likelihood score
            score = self.calculate_best_likelihood_score(children_iou, children_enclosure, parent_iou, parent_encompassment, only_leaf=only_leaf)
            
            # If this is the best score so far, keep it
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
        if not only_leaf:
            for child in new_parent.get_children():
                _, child_enc, _ = self.geometric_overlap_cached(child, node)
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
            return best_child_score
        else:
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

    def get_nodes_with_status(self, status: GraphNode.SegmentStatus) -> list[GraphNode]:
        node_list: list[GraphNode] = self.root_node.get_all_descendents()
        status_list: list[GraphNode] = []
        for node in node_list:
            if node.get_status() == status:
                status_list.append(node)
        return sorted(status_list, key=lambda n: n.get_id())

    def update_segment_statuses(self):
        # handle moving existing segments to inactive
        to_rm: list[GraphNode]= [node for node in self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT) \
                    if self.times[-1] - node.get_time_last_updated() > self.params.max_t_no_sightings \
                        or node.get_num_points() == 0]
        
        for node in to_rm:
            if node.get_num_points() == 0:
                logger.info(f"[bright_red] Deletion: [/bright_red] Node {node.get_id()} has zero points")
                node.remove_from_graph_complete(False)
                continue
            try:
                # Don't update voxel grid, as ROMAN keeps old one even though points are updated.
                node._dbscan_clustering(reset_voxel_grid=False) 

                node.set_status(GraphNode.SegmentStatus.INACTIVE)
            except Exception as e: # too few points to form clusters
                logger.info(f"[bright_red] Deletion: [/bright_red] Node {node.get_id()} has too few points to form clusters")
                node.remove_from_graph_complete(False)
            
        # handle moving inactive segments to graveyard
        to_rm : list[GraphNode]= [node for node in self.get_nodes_with_status(GraphNode.SegmentStatus.INACTIVE) \
                    if self.times[-1] - node.get_time_last_updated() > 15.0 \
                    or np.linalg.norm(node.get_last_pose()[:3,3] - self.poses[-1][:3,3]) > 10.0]
        for node in to_rm:
            node.set_status(GraphNode.SegmentStatus.GRAVEYARD)

        to_rm = [node for node in self.get_nodes_with_status(GraphNode.SegmentStatus.NURSERY) \
                    if self.times[-1] - node.get_time_last_updated() > self.params.max_t_no_sightings \
                        or node.get_num_points() == 0]
        for node in to_rm:
            logger.info(f"[bright_red] Deletion: [/bright_red] Node {node.get_id()} from nursery due to no sightings or zero points")
            node.remove_from_graph_complete(False)

        # handle moving segments from nursery to normal segments
        to_upgrade = [node for node in self.get_nodes_with_status(GraphNode.SegmentStatus.NURSERY) \
                        if node.get_num_sightings() >= 2]
        for node in to_upgrade:
            node.set_status(GraphNode.SegmentStatus.SEGMENT)

    def association_merges(self):
        """ Checks for Association Merges using same minimum requirements used by Hungarian algorithm. """

        max_iter = 100
        n = 0
        edited = True

        self.remove_bad_nodes(self.get_nodes_with_status(GraphNode.SegmentStatus.INACTIVE), 
            min_max_extent=0.25, plane_prune_params=[3.0, 3.0, 0.5])
        self.remove_bad_nodes(self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT))

        while n < max_iter and edited:
            edited = False
            n += 1

            for i, node1 in enumerate(self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT)):
                for j, node2 in enumerate(self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT) 
                                          + self.get_nodes_with_status(GraphNode.SegmentStatus.INACTIVE)):
                    if i >= j:
                        continue

                    # if segments are very far away, don't worry about doing extra checking
                    if np.mean(node1.get_point_cloud()) - np.mean(node2.get_point_cloud()) > \
                        .5 * (np.max(node1.get_extent()) + np.max(node2.get_extent())):
                        continue 
                    
                    H_camera_wrt_world = self.poses[-1] @ np.linalg.inv(self.pose_FLU_wrt_Camera)
                    mask1 = node1.reconstruct_mask(H_camera_wrt_world)
                    mask2 = node2.reconstruct_mask(H_camera_wrt_world)
                    intersection2d = np.logical_and(mask1, mask2).sum()
                    union2d = np.logical_or(mask1, mask2).sum()
                    iou2d = intersection2d / union2d

                    iou3d, _, _ = self.geometric_overlap_cached(node1, node2)

                    if iou3d > self.params.min_iou_for_association or iou2d > self.params.min_iou_2d_for_merging:
                        logger.info(f"Merging segments {node1.id} and {node2.id} with 3D IoU {iou3d:.2f} and 2D IoU {iou2d:.2f}")

                        new_node = node1.merge_with_node(node2, keep_children=GraphNode.params.parent_node_includes_child_node_for_data)
                        new_node.status = GraphNode.SegmentStatus.SEGMENT
                        if new_node is None or new_node.get_num_points() == 0:
                            logger.info(f"[bright_red]Merge Fail[/bright_red]: Resulting Node was invalid.")
                        else:
                            self.add_new_node_to_graph(new_node, only_leaf=False)

                        edited = True
                        break
                if edited:
                    break

    def remove_bad_nodes(self, nodes: list[GraphNode], min_volume: float=0.0, 
            min_max_extent: float=0.0, plane_prune_params: list[float]=[np.inf, np.inf, 0.0]) -> None:
        """ Remove nodes that have small volumes or have no points """
        to_delete: list[GraphNode] = []
        for node in nodes:
            try:
                extent = np.sort(node.get_extent())
                if node.get_num_points() == 0:
                    to_delete.append(node)
                elif node.get_volume() < min_volume:
                    to_delete.append(node)
                elif extent[-1] < min_max_extent:
                    to_delete.append(node)
                elif extent[2] > plane_prune_params[0] and extent[1] > plane_prune_params[1] and extent[0] < plane_prune_params[2]:
                    to_delete.append(node)
            except: 
                to_delete.append(node)
        for node in to_delete:
            # Print the id of the segment we're deleting
            logger.info(f"Deleting segment {node.id} as it is a bad segment")
            node.remove_from_graph_complete(False)

    class NodeRelationship(Enum):
        SHARED_HOLONYM = 0
        HOLONYM_MERONYM = 1
        SYNONYMY = 2

    @staticmethod 
    def find_putative_relationships(node_list_i: list[GraphNode], 
                                      node_list_j: list[GraphNode] | None = None,
                                      relationship_type: SceneGraph3D.NodeRelationship = NodeRelationship.SHARED_HOLONYM) \
                                      -> list[tuple[int, int, Any]]:
        """ 
        Finds putative relationships between two lists of nodes. 
        If second list is none, instead compute between nodes in the first list.
        """

        # Determine if comparing same list to self or two different lists
        comparing_to_self: bool = False
        if node_list_j is None:
            comparing_to_self = True
            node_list_j = node_list_i

        # Create structure to track all pairs of nodes with putative relationships
        putative_relationships: list[tuple[int, int, Any]] = []

        # Iterate through each pair of nodes
        for i, node_i in enumerate(node_list_i):
            for j, node_j in enumerate(node_list_j):

                # Skip if either node is the Root Graph Node
                if node_i.is_RootGraphNode() or node_j.is_RootGraphNode():
                    continue

                # If comparing to self, skip pairs already compared
                if comparing_to_self:
                    if i >= j: continue

                match relationship_type:
                    case SceneGraph3D.NodeRelationship.SHARED_HOLONYM:

                        # Get all holonyms for words
                        word_i_holonyms_pure: set[str] = node_i.get_all_holonyms(False)
                        word_j_holonyms_pure: set[str] = node_j.get_all_holonyms(False)

                        # TODO: Do an option where we can do upwards but excluding holonyms from a holonym/meronym relationship.

                        # Check if there is a shared holonym
                        shared_holonyms: list[str] = sorted(set.intersection(word_i_holonyms_pure, word_j_holonyms_pure))
                        if len(shared_holonyms) > 0:

                            # Append to list of potential nodes with shared holonyms
                            putative_relationships.append((i, j, shared_holonyms))
                            
                    case SceneGraph3D.NodeRelationship.HOLONYM_MERONYM:
                        
                        # Skip nodes that are ascendent or descendent with each other
                        if comparing_to_self and not node_i.is_descendent_or_ascendent(node_j): continue
                        
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
                            
                            node_list_i_has_meronym: bool = False
                            if word_i.word in word_j_meronyms or word_j.word in word_i_holonyms:
                                node_list_i_has_meronym = True

                            # Append to list of potential nodes with holonym-meronym relationships
                            # with meronym coming first
                            putative_relationships.append((i, j, node_list_i_has_meronym))
                    
                    case SceneGraph3D.NodeRelationship.SYNONYMY:

                        # Get words wrapped for each node
                        words_i = node_i.get_words()
                        words_j = node_j.get_words()

                        # Calculate Semantic Similarity
                        sem_con = SceneGraph3D.semantic_consistency(node_i.get_weighted_semantic_descriptor(), node_j.get_weighted_semantic_descriptor())

                        # Check for synonymy (if any of shared lemmas are same & cosine similarity is high)
                        if words_i == words_j or sem_con > SceneGraph3D.min_sem_con_for_association:
                            putative_relationships.append((i, j, None))

        # Return all found putative relationships
        return putative_relationships
    
    def synonym_relationship_merging(self):
        """ Detect parent-child and nearby node synonymy merges. """

        # Get all putative synonymys based on WordNet & cosine similarity
        putative_synonyms: list[tuple[int, int, None]] = SceneGraph3D.find_putative_relationships(
                    list(self.root_node), relationship_type=SceneGraph3D.NodeRelationship.SYNONYMY)
        
        # Get the list of nodes in the graph
        node_list_initial = list(self.root_node).copy()
        
        # For each putative synonymy, calculate detected ones based on graph structure & geometry
        detected_synonyms: list[set[int]] = []
        for putative_rel in putative_synonyms:
            
            # Extract the corresponding nodes
            node_i: GraphNode = node_list_initial[putative_rel[0]]
            node_j: GraphNode = node_list_initial[putative_rel[1]]

            # If they are parent-child OR geometrically close:
            if node_i.is_parent_or_child(node_j) or self.shortest_dist_to_node_size_ratio(node_i, node_j) < \
                                                    self.ratio_dist2length_threshold_nearby_node_semantic_merge:
                # Add to list of detected
                detected_synonyms.append({putative_rel[0], putative_rel[1]})

        # Merge all overlaps to get final sets of all synonymys
        i = 0
        while i < len(detected_synonyms):
            first = detected_synonyms[i]
            j = i + 1
            while j < len(detected_synonyms):

                # If overlap is detected
                if first & detected_synonyms[j]:  
                    # Merge the sets
                    first |= detected_synonyms.pop(j)
                else:
                    j += 1
            i += 1

        # For each detected synonymy (after merging), merge all nodes one-by-one
        for detected_syn in detected_synonyms:

            # Get all nodes that are synonyms
            synonyms: list[GraphNode] = [node_list_initial[idx] for idx in detected_syn]

            # Phase 1: Merge parent-child relationships while ignoring nearby-node ones
            i = 0
            while i < len(synonyms):
                node_i = synonyms[i]
                j = i + 1
                while j < len(synonyms):
                    node_j = synonyms[j]

                    # If they are parent & child
                    if node_i.is_parent_or_child(node_j):

                        # Merge child into parent & remove merged node from list
                        logger.info(f"[green1]Parent-Child Synonymy[/green1]: Merging Node {node_i.get_id()} and Node {node_j.get_id()}")
                        node_i.merge_parent_and_child(node_j)
                        synonyms.pop(j)  

                    else:
                        j += 1
                i += 1

            # Phase 2: Merge remaining nodes (non-parent/child)
            i = 0
            while i < len(synonyms):
                node_i = synonyms[i]
                j = i + 1
                while j < len(synonyms):
                    node_j = synonyms[j]

                    # Merge the two nodes
                    logger.info(f"[green3]Nearby Obj Sem Merge[/green3]: Merging Node {node_j.get_id()} and Node {node_i.get_id()} and popping off graph")
                    merged_node = node_i.merge_with_node(node_j)
                    if merged_node is None:
                        logger.info(f"[bright_red]Merge Fail[/bright_red]: Resulting Node was invalid.")
                        j += 1
                    else:
                        # Add merged node to graph
                        self.add_new_node_to_graph(merged_node, only_leaf=True)
                        synonyms.pop(j)
                        node_i = merged_node
                i += 1

    def holonym_meronym_relationship_inference(self):
        """ Detect parent-child relationships between non-ascendent/descendent nodes in the graph. """

        # Get all putative holonym-meronym relationships based on WordNet
        putative_holonym_meronyms: list[tuple[int, int, bool]] = SceneGraph3D.find_putative_relationships(
                    list(self.root_node), relationship_type=SceneGraph3D.NodeRelationship.HOLONYM_MERONYM)

        # Get the list of nodes in the graph
        node_list_initial = list(self.root_node).copy()

        # For each putative relationship, calculate detected ones based on geometry
        detected_holonym_meronyms: list[tuple[int, int]] = [] # Meronym will be first
        for putative_rel in putative_holonym_meronyms:

            # Extract the corresponding nodes
            node_i: GraphNode = node_list_initial[putative_rel[0]]
            node_j: GraphNode = node_list_initial[putative_rel[1]]

            # If they are close enough
            if self.shortest_dist_to_node_size_ratio(node_i, node_j) < self.ratio_dist2length_threshold_holonym_meronym:

                # Put into detected list with meronym first
                if putative_rel[2]:
                    detected_holonym_meronyms.append([(putative_rel[0], putative_rel[1])])
                else:
                    detected_holonym_meronyms.append([(putative_rel[1], putative_rel[0])])
       
        # Iterate over all detected holonym-meronym relationships to resolve overlaps and conflicts
        overlap = True
        while overlap:
            overlap = False

            # Map each child list index of the detected holonyms to the indices of tuples containing it
            meronym_list_index_to_holonym_list_index = defaultdict(list)
            for (a, b) in detected_holonym_meronyms:
                meronym_list_index_to_holonym_list_index[a].append(b)

            # Find out if any meronyms have two or more detected holonyms
            for i, detected_holonym_indices in meronym_list_index_to_holonym_list_index.items():
                if len(detected_holonym_indices) > 1:

                    # If so, pick whichever one is closest to it geometrically
                    putative_holonyms: list[GraphNode] = [node_list_initial[idx] for idx in detected_holonym_indices]
                    meronym: GraphNode = node_list_initial[i]

                    meronym_cen: np.ndarray = meronym.get_centroid()
                    centroid_distances = [np.linalg.norm(meronym_cen - holonym.get_centroid()) for holonym in putative_holonyms]
                    holonym_to_keep_idx = centroid_distances.index(min(centroid_distances))
                    
                    # Delete all other holonym_meronym relationships
                    for idx in sorted(detected_holonym_indices, reverse=True):
                        if idx != holonym_to_keep_idx:
                            detected_holonym_meronyms.pop(idx)

                    # Our mapping has changed, so reloop
                    overlap = True
                    break

        # For each detected holonym-meronym relationship (after conflict resolution), alter the graph
        for detected_rel in detected_holonym_meronyms:

            # Extract the corresponding nodes
            node_i: GraphNode = node_list_initial[detected_rel[0]]
            node_j: GraphNode = node_list_initial[detected_rel[1]]

            # Load words and some meronyms/holonyms
            word_i: WordWrapper = node_i.get_words().words[0]
            word_j: WordWrapper = node_j.get_words().words[0]
            word_j_meronyms: set[str] = node_j.get_all_meronyms(2)
            word_i_holonyms: set[str] = node_i.get_all_holonyms(True, 2)

            # Find the meronym
            if word_i.word in word_j_meronyms or word_j.word in word_i_holonyms:
                node_meronym = node_i
                node_holonym = node_j
                word_holonym = word_j
            else:
                node_meronym = node_j
                node_holonym = node_i
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
            node_holonym.add_semantic_descriptors([(holonym_emb, total_weight * self.params.ratio_relationship_weight_2_total_weight)])
            if GraphNode.params.calculate_descriptor_incrementally:
                raise NotImplementedError("Holonym_Meronym Inference not currently supported with incremental semantic descriptor!")

            # Print any deleted nodes
            if len(deleted_ids) > 0:
                logger.info(f"[bright_red] Discard: [/bright_red] Node(s) {deleted_ids} removed as not enough remaining points with children removal.")

    def shared_holonym_relationship_inference(self):
        """ Detect and add higher level objects to the graph. """

        # Get all putative holonyms based on WordNet
        putative_holonyms: list[tuple[int, int, list[str]]] = SceneGraph3D.find_putative_relationships(
                self.root_node.get_children(), relationship_type=SceneGraph3D.NodeRelationship.SHARED_HOLONYM)
        
        # Get the list of children nodes
        children_list_initial = self.root_node.get_children().copy()

        # For each putative holonym, calculate detected ones based on map geometric knowledge
        detected_holonyms: list[tuple[list[int], list[str]]] = []
        for putative_holonym in putative_holonyms:

            # Extract the corresponding nodes
            node_i: GraphNode = children_list_initial[putative_holonym[0]]
            node_j: GraphNode = children_list_initial[putative_holonym[1]]

            # If they are close enough, declare this putative holonym as detected
            if self.shortest_dist_to_node_size_ratio(node_i, node_j) < self.ratio_dist2length_threshold_shared_holonym:
                detected_holonyms.append(([putative_holonym[0], putative_holonym[1]], putative_holonym[2]))

        # Iterate over all detected shared holonyms to resolve overlaps and conflicts
        overlap = True
        while overlap:
            overlap = False

            # Map each child list index of the detected holonyms to the indices of tuples containing it
            child_list_index_to_detected_holonym_indices = defaultdict(list)
            for idx, (a, b, _) in enumerate(detected_holonyms):
                child_list_index_to_detected_holonym_indices[a].append(idx)
                child_list_index_to_detected_holonym_indices[b].append(idx)

            # Find out if any children of detected holonyms overlap, and if so, attempt to merge
            for detected_holonym_indices in child_list_index_to_detected_holonym_indices.values():
                if len(detected_holonym_indices) > 1:

                    # Get all sets of shared holonyms
                    holonym_strs: list[set[str]] = []
                    for detected_holonym_index in detected_holonym_indices:
                        holonym_strs.append(set(detected_holonyms[detected_holonym_index][1]))

                    # Calculate the intersection
                    intersection: set[str] = holonym_strs[0].intersection(*holonym_strs[1:])

                    # If there is a shared holonym to all of these 
                    if len(intersection) > 0:
                        # Add a merged detected holonym
                        all_child_list_indices: list[int] = []
                        for i in detected_holonym_indices:
                            all_child_list_indices += detected_holonyms[i][0]
                        detected_holonyms.append((all_child_list_indices, sorted(intersection)))

                        # Pop all previous detected holonyms that formed the merge one
                        for idx in sorted(detected_holonym_indices, reverse=True):
                            detected_holonyms.pop(idx)
                    else:
                        # There is no shared holonym, so these detected holonyms are NOT consistent.
                        # Just greedily pick the first one, which invalidates all others
                        for idx in sorted(detected_holonym_indices[1:], reverse=True):
                            detected_holonyms.pop(idx)

                    # Regardless, our mapping has changed, so reloop
                    overlap = True
                    break
                    
        # For each detected holonym (after conflict resolution), add to graph
        for detected_holonym in detected_holonyms:

            # Extract the corresponding nodes
            nodes: list[GraphNode] = children_list_initial[detected_holonym[0]]

            # Calculate the first seen time  and first pose as earliest from all nodes (and children)
            earliest_node = min(nodes, key=lambda n: n.get_time_first_seen())
            first_seen = earliest_node.get_time_first_seen()
            first_pose = earliest_node.get_first_pose()

            # Create the holonym node
            holonym: GraphNode | None = GraphNode.create_node_if_possible(self.root_node.request_new_ID(), 
                                            self.root_node, [], None, 0, np.zeros((0, 3), dtype=np.float64), 
                                            nodes, first_seen, self.times[-1], self.times[-1], 
                                            first_pose, self.poses[-1], self.poses[-1], is_RootGraphNode=False)

            # If the node was successfully created, add to the graph
            if holonym is not None:
                # TODO: Maybe search to see if other nodes are in the overlap space between
                # the two children which should also be part of the set?

                self.place_node_in_graph(holonym, self.root_node, nodes)

                # Update embedding so that it matches the word most of all!
                # TODO: If there are multiple shared holonyms, maybe update with combination of all?
                shared_holonyms: list[str] = putative_holonym[2]
                holonym_emb: np.ndarray = wordnetWrapper.get_embedding_for_word(shared_holonyms[0])
                total_weight = holonym.get_total_weight_of_semantic_descriptors()
                holonym.add_semantic_descriptors([(holonym_emb, total_weight * self.params.ratio_relationship_weight_2_total_weight)])
                if GraphNode.params.calculate_descriptor_incrementally:
                    raise NotImplementedError("Holonym Inference not currently supported with incremental semantic descriptor!")

                # TODO: Need to get words for these children
                logger.info(f"[gold1]Shared Holonym Detected[/gold1]: Node {holonym.get_id()} with words {shared_holonyms} {holonym.get_words()} from children with words {word_i.word} and {word_j.word}")

            else:
                raise RuntimeError("Detected Holonym is not a valid GraphNode!")

    def get_roman_map(self) -> ROMANMap:
        """
        Convert this SceneGraph3D into a ROMANMap of seperate object segments (no meronomy) to see how ROMAN's alignment
        algorithm works with our objects that have had extra merging operations performed between them.
        """

        # Remove bad nodes first
        self.remove_bad_nodes(self.get_nodes_with_status(GraphNode.SegmentStatus.GRAVEYARD) +
                              self.get_nodes_with_status(GraphNode.SegmentStatus.INACTIVE) +
                              self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT))

        # Convert all graph nodes to segments
        segment_map: list[Segment] = []
        relevant_nodes = self.get_nodes_with_status(GraphNode.SegmentStatus.GRAVEYARD) + \
                         self.get_nodes_with_status(GraphNode.SegmentStatus.INACTIVE) + \
                         self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT)
        for node in relevant_nodes:
            segment_map.append(node.to_segment())

        # Reset obb for each segment
        for seg in segment_map:
            seg.reset_obb()

        # Return the ROMANMap
        return ROMANMap(
            segments=segment_map,
            trajectory=self.poses,
            times=self.times,
            poses_are_flu=True
        )
    
    def load_map_from_pickle(file_path: str) -> SceneGraph3D:
        """ Load a SceneGraph3D from a pickled file. """

        with open(file_path, 'rb') as f:
            pickle_data = pickle.load(f)
            
            if type(pickle_data) == SceneGraph3D:
                return pickle_data
            else:
                raise ValueError("File path does not point to a pickled SceneGraph3D")
