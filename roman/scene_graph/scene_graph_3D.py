from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from .graph_node import GraphNode
from .scene_graph_utils import find_point_overlap_with_hulls, convex_hull_geometric_overlap, shortest_dist_between_convex_hulls, expand_hull_outward_by_fixed_offset, merge_overlapping_sets, merge_objs_via_function
from ..logger import logger
from ..map.observation import Observation
import multiprocessing
import numpy as np
import os
from ..params.data_params import ImgDataParams
from ..params.scene_graph_3D_params import SceneGraph3DParams, GraphNodeParams
import pickle
from ..rerun_wrapper.rerun_wrapper_window_map import RerunWrapperWindowMap
from robotdatapy.data.img_data import CameraParams
from robotdatapy.transform import transform
from roman.map.map import ROMANMap
from roman.object.segment import Segment
from .scene_graph_3D_base import SceneGraph3DBase
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked
from typing import Any

multiprocessing.set_start_method("spawn", force=True)

class SceneGraph3D(SceneGraph3DBase):

    @typechecked
    def __init__(self, camera_params: CameraParams, _T_camera_flu: np.ndarray, rerun_viewer: RerunWrapperWindowMap):
        super().__init__()

        # Save parameters 
        GraphNode.camera_params = camera_params

        # Keeps track of current time so we can keep track of when we are updated.
        self.times: list[float] = []

        # Keeps track of the current pose so we know where we are
        self.poses: list[np.ndarray] = []

        # Track FLU pose wrt camera frame
        self.pose_FLU_wrt_Camera = _T_camera_flu

        # Save the visualization
        self.rerun_viewer = rerun_viewer

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

        # Update the viewer
        self.rerun_viewer.update_curr_time(self.times[-1])
        self.rerun_viewer.update_graph(self.root_node)
        self.rerun_viewer.update_img(img)
        self.rerun_viewer.update_depth_img(depth_img)
        self.rerun_viewer.update_camera_pose(pose)
        self.rerun_viewer.update_camera_intrinsics(img_data_params)

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
                logger.info(f"[dark_blue]Associated Nodes[/dark_blue]: {len(associated_pairs)} new nodes successfully associated")

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
                                if GraphNode.params.ignore_descriptors_from_observation:
                                    obs_descriptor: np.ndarray = new_node.obs_descriptor
                                else:
                                    obs_descriptor: np.ndarray = new_node.semantic_descriptor_inc
                                futures[executor.submit(node.merge_with_observation, new_node.get_point_cloud(), 
                                                        new_node.semantic_descriptors, obs_descriptor)] = i
                                
        
                # Wait for jobs to finish and update associated pairs
                for future in as_completed(futures):
                    result = future.result()

                    # TODO: Update associated pairs?

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
                    if self.scene_graph_3D_params.downsample_and_remove_outliers_after_hungarian_for_new_nodes:
                        to_delete = node.update_point_cloud(np.zeros((0, 3), dtype=np.float64), downsample=True, remove_outliers=True)
                        if len(to_delete) > 0:
                            # Don't add this node since its no longer valid
                            logger.info(f"[bright_red]Deletion:[/bright_red] New node no longer valid after downsampling/removing outliers")
                            continue

                    # Discard if there are no points
                    if node.get_num_points() == 0: 
                        logger.info(f"[bright_red]Deletion:[/bright_red] New node has zero points")
                        continue

                    # Re-assign ids to the non-associated nodes (so they line up with ROMAN ids)
                    node.set_id(self.root_node.request_new_ID())

                    # Add the node to the graph
                    new_id = self.add_new_node_to_graph(node)
                    associated_pairs.append((new_id, new_id))
            
        # Update the viewer
        self.rerun_viewer.update_graph(self.root_node)
        self.rerun_viewer.update_seg_img(seg_img, img, associated_pairs, node_to_obs_mapping)

        # Run merging operations
        self.association_merges()

        # Update the viewer
        self.rerun_viewer.update_graph(self.root_node)

        # TODO: After merging, update the associations and node_to_obs mapping so colors carry over

        # Resolve any overlapping point clouds (TODO: Do we even need this?)
        # TODO: This seems to be causing bugs currently
        if self.scene_graph_3D_params.enable_resolve_overlapping_nodes:
            self.resolve_overlapping_nodes()

        # Update the viewer
        self.rerun_viewer.update_graph(self.root_node)
        self.rerun_viewer.update_seg_img(seg_img, img, associated_pairs, node_to_obs_mapping)

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
    
    @typechecked
    def pass_minimum_requirements_for_association(self, iou: float) ->  bool:
        """ Currently, only use geometric information for association as drift should
            have no negative impact in small trajectories. """

        # Make sure inputs are within required bounds
        SceneGraph3D.check_within_bounds(iou, (0, 1))

        # Check if within thresholds for association
        return bool(iou >= self.scene_graph_3D_params.min_iou_3d)

    @typechecked
    def resolve_overlapping_nodes(self):
        """ Resolve node overlaps to enforce impenetrability in our graph"""
               
        # Iterate through entire graph until no overlaps with shared points are detected
        change_occured = True
        while change_occured:

            # Track if we need to loop again
            change_occured = False

            seg_and_inactive_nodes = self.get_nodes_with_status(GraphNode.SegmentStatus.SEGMENT) + self.get_nodes_with_status(GraphNode.SegmentStatus.INACTIVE)

            # Iterate through each pair of nodes that aren't an ascendent or descendent with each other
            for i, node_i in enumerate(seg_and_inactive_nodes):
                for j, node_j in enumerate(seg_and_inactive_nodes):
                    if i < j and not node_i.is_descendent_or_ascendent(node_j):

                        # If the nodes are obviously seperated, no point in checking.
                        if not self.check_if_nodes_are_somewhat_nearby(node_i, node_j):
                            continue

                        # Calculate the geometric overlap between them
                        iou, enc_i, enc_j = self.geometric_overlap_cached(node_i, node_j)

                        # If this is above a threshold, then we've found an overlap
                        if iou > self.scene_graph_3D_params.iou_threshold_overlapping_obj or enc_i > self.scene_graph_3D_params.enc_threshold_overlapping_obj \
                            or enc_j > self.scene_graph_3D_params.enc_threshold_overlapping_obj:
                            
                            # Get merged point clouds from both nodes
                            pc_i = node_i.get_point_cloud()
                            pc_j = node_j.get_point_cloud()
                            pc_merged = np.concatenate((pc_i, pc_j), dtype=np.float64)

                            # Find all points in this overlap region
                            contain_masks = find_point_overlap_with_hulls(pc_merged, [node_i.get_convex_hull(), node_j.get_convex_hull()])
                            num_mask_assignments = np.sum(contain_masks, axis=0)
                            overlaps = np.where(num_mask_assignments > 1)[0]
                            pc_overlap = pc_merged[overlaps,:]

                            # If there is at least a single point in the region
                            if len(pc_overlap) > 0:

                                # Double check this isn't the same node
                                assert node_i.get_id() != node_j.get_id(), f"Same node {node_i.get_id()} referred to as child of two parents!"

                                # Remove these points from their parents
                                node_i.remove_points_complete(pc_overlap)
                                node_j.remove_points_complete(pc_overlap)

                                # If we don't just throw away the overlap
                                if not self.scene_graph_3D_params.overlapping_nodes_throw_away_overlap:
                                    raise NotImplementedError("Setting overlapping_nodes_throw_away_overlap to False isn't currently supported!")

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
                                        logger.info(f"[blue_violet]Overlap Detected:[/blue_violet] Between Node {node_i.get_id()} and Node {node_j.get_id()}, adding as Node {new_node.get_id()} to graph.")
                                        new_id = self.add_new_node_to_graph(new_node, only_leaf=True)
                                    else:
                                        logger.info(f"[blue_violet]Overlap Detected:[/blue_violet] Between Node {node_i.get_id()} and Node {node_j.get_id()}, discarding...")
                                else:
                                    logger.info(f"[blue_violet]Overlap Detected:[/blue_violet] Between Node {node_i.get_id()} and Node {node_j.get_id()}, discarding...")

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
    @staticmethod
    def check_within_bounds(a: float, bounds: tuple) -> None:
        """
        Helper method to check input values.

        Args:
            bounds (tuple): These bounds are inclusive, with index 0 as lower and 1 as upper.
        """

        if a < bounds[0] or a > bounds[1]:
            raise ValueError(f"Value {a} is outside the range of [0, 1] inclusive.")
    
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
                    if self.times[-1] - node.get_time_last_updated() > self.scene_graph_3D_params.max_t_no_sightings \
                        or node.get_num_points() == 0]
        
        for node in to_rm:
            if node.get_num_points() == 0:
                logger.info(f"[bright_red]Deletion:[/bright_red] Node {node.get_id()} has zero points")
                node.remove_from_graph_complete(False)
                continue
            try:
                # Don't update voxel grid, as ROMAN keeps old one even though points are updated.
                if self.scene_graph_3D_params.enable_dbscan_on_node_inactivation:
                    node._dbscan_clustering(reset_voxel_grid=self.scene_graph_3D_params.update_voxel_grid_on_inactivation_dbscan) 

                node.set_status(GraphNode.SegmentStatus.INACTIVE)
            except Exception as e: # too few points to form clusters
                logger.info(f"[bright_red]Deletion:[/bright_red] Node {node.get_id()} has too few points to form clusters")
                node.remove_from_graph_complete(False)
            
        # handle moving inactive segments to graveyard
        to_rm : list[GraphNode]= [node for node in self.get_nodes_with_status(GraphNode.SegmentStatus.INACTIVE) \
                    if self.times[-1] - node.get_time_last_updated() > 15.0 \
                    or np.linalg.norm(node.get_last_pose()[:3,3] - self.poses[-1][:3,3]) > 10.0]
        for node in to_rm:
            node.set_status(GraphNode.SegmentStatus.GRAVEYARD)

        to_rm = [node for node in self.get_nodes_with_status(GraphNode.SegmentStatus.NURSERY) \
                    if self.times[-1] - node.get_time_last_updated() > self.scene_graph_3D_params.max_t_no_sightings \
                        or node.get_num_points() == 0]
        for node in to_rm:
            logger.info(f"[bright_red]Deletion:[/bright_red] Node {node.get_id()} from nursery due to no sightings within timeframe or zero points")
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

                    if iou3d > self.scene_graph_3D_params.min_iou_3d or iou2d > self.scene_graph_3D_params.min_iou_2d_for_merging:
                        logger.info(f"[navy_blue]Association Merge[/navy_blue]: Merging segments {node1.id} and {node2.id} with 3D IoU {iou3d:.2f} and 2D IoU {iou2d:.2f}")

                        new_node = node1.merge_with_node_mapping(node2, keep_children=False)
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
