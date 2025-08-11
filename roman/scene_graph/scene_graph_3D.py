from __future__ import annotations

from .graph_node import GraphNode
from .hull_methods import find_point_overlap_with_hulls, convex_hull_geometric_overlap, shortest_dist_between_convex_hulls
from itertools import chain, combinations
from .logger import logger
from ..map.observation import Observation
import numpy as np
import open3d as o3d
from ..params.data_params import ImgDataParams
from .rerun_wrapper import RerunWrapper
from robotdatapy.transform import transform
from scipy.optimize import linear_sum_assignment
import trimesh
from typeguard import typechecked

class SceneGraph3D():
    # Node that connects all highest-level objects together for implementation purposes
    root_node: GraphNode

    # List of high-level nodes that have been inactivated
    inactive_nodes: list[GraphNode] = []

    # Keeps track of current time so we can keep track of when we are updated.
    times: list[float] = []

    # Keeps track of the current pose so we know where we are
    poses: list[np.ndarray] = []

    # Requirement for an observation to be associated with a current graph node or for two nodes to be merged.
    min_iou_for_association = 0.6

    # Used for "Nearby Children Semantic Merging" and "Parent-Child Semantic Merging"
    min_sem_con_for_association = 0.9 

    # Threshold to determine if two convex hulls are overlapping in resolve_overlapping_convex_hulls()
    iou_threshold_overlapping_obj = 0.2

    # Ratio of distance to object volume threshold for "Nearby Children Semantic Merging"
    ratio_dist2length_threshold_nearby_children_semantic_merge = 0.05

    # TODO: Should I keep or get rid of this?
    # If a NEW node isn't seen for this time, remove from graph
    # max_t_no_sightings_to_prune_new = 0.4 # seconds

    # If a high-level node goes this long since it was first seen, inactivate
    max_t_active_for_node = 15 # seconds

    # If we travel a significant distance from the first camera pose where this object was seen, inactivate
    max_dist_active_for_node = 10 # meters

    @typechecked
    def __init__(self, _T_camera_flu: np.ndarray, headless: bool = False):
        self.root_node = GraphNode.create_node_if_possible(0, None, [], np.zeros((0, 3), dtype=np.float64), [], 0, 0, 0, np.empty(0), np.empty(0), True)
        self.pose_FLU_wrt_Camera = _T_camera_flu
        self.rerun_viewer = RerunWrapper(enable=not headless)

    @typechecked
    def len(self) -> int:
        return self.root_node.get_number_of_nodes()

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
        self.rerun_viewer.update(self.root_node, self.times[-1], img, depth_img, pose, img_data_params, seg_img, [])

        # Convert each observation into a node (if possible)
        nodes = []
        node_to_obs_mapping = {}
        for i, obs in enumerate(observations):
            new_node: GraphNode | None = self.convert_observation_to_node(obs)
            if new_node is not None:
                nodes.append(new_node)
                node_to_obs_mapping[new_node.get_id()] = i
        logger.info(f"Called with {len(nodes)} valid observations")
        
        # Associate new nodes with previous nodes in the graph
        associated_pairs = self.hungarian_assignment(nodes)

        # Merge associated pairs
        if len(associated_pairs) > 0:
            logger.info(f"[dark_blue]Association Merges[/dark_blue]: {len(associated_pairs)} new nodes successfully associated")
        for i, new_node in enumerate(nodes):
            for j, node in enumerate(self.root_node):

                # See if this is a match
                for pair in associated_pairs:
                    if new_node.get_id() == pair[0] and node.get_id() == pair[1]:

                        # If so, update node with observation information
                        node.merge_with_observation(new_node.get_point_cloud(), new_node.get_semantic_descriptors())
        
        self.rerun_viewer.update(self.root_node, self.times[-1], img=img, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)
        
        # Add the remaining unassociated valid_obs as nodes to the scene graph
        associated_obs_indices = [x[0] for x in associated_pairs]
        for node in nodes:
            if node.get_id() not in associated_obs_indices:
                new_id = self.add_new_node_to_graph(node)
                associated_pairs.append((new_id, new_id))

        # Update the viewer
        self.rerun_viewer.update(self.root_node, self.times[-1], img=img, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

        # Merge nodes that can be merged, and infer higher level ones
        self.merging_and_generation()

        # Resolve any overlapping point clouds
        self.resolve_overlapping_convex_hulls()

        # Run node retirement
        self.node_retirement()

        # Update the viewer
        self.rerun_viewer.update(self.root_node, self.times[-1], img=img, seg_img=seg_img, associations=associated_pairs, node_to_obs_mapping=node_to_obs_mapping)

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
                iou, _, _ = convex_hull_geometric_overlap(node.get_convex_hull(), new_node.get_convex_hull())

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
        #logger.info(f"Cosine Similarity Before: {cos_sim}")
        min_val, max_val = rescaling
        rescaled = (cos_sim - min_val) / (max_val - min_val)
        rescaled_clamped = np.clip(rescaled, 0.0, 1.0)
        #logger.info(f"Cosine Similarity after Rescaling: {rescaled_clamped}")
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
        # Iterate through entire graph until no overlaps with shared points are detected
        change_occured = True
        while change_occured:
            self.rerun_viewer.update(self.root_node, self.times[-1])

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
                                new_node: GraphNode | None  = self.convert_point_cloud_to_node(pc_overlap)
                                if new_node is not None:
                                    new_id = self.add_new_node_to_graph(new_node, only_leaf=True)

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
    def convert_observation_to_node(self, obs: Observation) -> GraphNode | None:

        # Create a new node for this observation
        new_node: GraphNode | None = GraphNode.create_node_if_possible(self.root_node.request_new_ID(), None, [], 
                                        obs.transformed_points, [],  self.times[-1], 
                                        self.times[-1], self.times[-1], 
                                        self.poses[-1], self.poses[-1])
        if new_node is None: return None # Node creation failed

        # Add the descriptor to the node
        if obs.clip_embedding is not None:
            new_node.add_semantic_descriptors([(obs.clip_embedding, new_node.get_volume())])
        return new_node
    
    @typechecked
    def convert_point_cloud_to_node(self, point_cloud: np.ndarray) -> GraphNode | None:

        # Create a new node for this observation
        new_node: GraphNode | None = GraphNode.create_node_if_possible(self.root_node.request_new_ID(), None, [], 
                                        point_cloud, [],  self.times[-1], 
                                        self.times[-1], self.times[-1], 
                                        self.poses[-1], self.poses[-1])
        return new_node # If it failed, we return None to signify it

    @typechecked
    def add_new_node_to_graph(self, node: GraphNode, only_leaf=False) -> int:

        # Make sure the node passed to the graph is actually valid
        if not node._class_method_creation_success:
            raise RuntimeError("Cannot add_new_node_to_graph when node is invalid!")

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
        
        # Extract our convex hull
        hull_new = node.get_convex_hull()

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
            logger.info(f"[cyan]Added[/cyan]: Node {node.get_id()} added to graph as child of Node {new_parent.get_id()} and parent of Nodes {[c.get_id() for c in new_children]}")
        else:
            logger.info(f"[cyan]Added[/cyan]: Node {node.get_id()} added to graph as child of Node {new_parent.get_id()}")
        new_parent.add_child(node)
        node.set_parent(new_parent)
        
        for child in new_children:
            child.remove_from_graph_complete()
        for child in new_children:
            child.set_parent(None)

        node.add_children(new_children)
        for child in new_children:
            child.set_parent(node)

        # Return the id of the node just added
        return node.get_id()

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

            # TODO: Only consider children here that we are somewhat nearby (for speed)
            # Powerset is slowing us down significantly.

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

            # Track if we need to loop again
            merge_occured = False

            # Iterate through each pair of nodes that aren't an ascendent or descendent with each other
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i < j and not node_i.is_descendent_or_ascendent(node_j):
                        
                        # ========== Minimum Requirements for Association Merge ==========

                        # Calculate IOU and Semantic Similarity
                        iou, _, _ = convex_hull_geometric_overlap(node_i.get_convex_hull(), node_j.get_convex_hull())
    
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

            # Iterate through all pairs of children
            for i, node_i in enumerate(self.root_node):
                for j, node_j in enumerate(self.root_node):
                    if i < j and node_i.is_sibling(node_j):

                        # ========== Nearby Children Semantic Merge ==========

                        # Get shortest distance between the nodes
                        dist = shortest_dist_between_convex_hulls(node_i.get_convex_hull(), node_j.get_convex_hull())

                        # TODO: I wonder if using average distance would be better for 
                        # being resistant to noise in our nearby children semantic merging. Maybe try this!

                        # Get longest line of either node
                        longest_line_node_i = node_i.get_longest_line_size()
                        longest_line_node_j = node_j.get_longest_line_size()
                        longest_line = np.mean([longest_line_node_i, longest_line_node_j])

                        # Get ratio of shortest distance to longest line (in other words, dist to object length)
                        dist_to_object_length = dist / longest_line

                        # Finally, calculate semantic consistency
                        sem_con = self.semantic_consistency(node_i.get_weighted_semantic_descriptor(),
                                                            node_j.get_weighted_semantic_descriptor())

                        # If ratio of shortest distance to object length is within threshold AND
                        # the semanatic embedding is close enough for association
                        if dist_to_object_length < self.ratio_dist2length_threshold_nearby_children_semantic_merge and \
                            sem_con < self.min_sem_con_for_association:

                            # If so, merge these two nodes in the graph
                            logger.info(f"[gold1]Nearby Obj Sem Merge[/gold1]: Merging Node {node_j.get_id()} into Node {node_i.get_id()} and popping off graph")
                            logger.debug(f"SHORTEST DIST: {dist}")
                            logger.debug(f"LONGEST LINE: {longest_line}")
                            logger.debug(f"DIST TO OBJECT LENGTH: {dist_to_object_length} < {self.ratio_dist2length_threshold_nearby_children_semantic_merge}")
                            merged_node = node_i.merge_with_node(node_j)
                            if merged_node is None:
                                logger.info(f"[bright_red]Merge Fail[/bright_red]: Resulting Node was invalid.")
                            else:
                                self.add_new_node_to_graph(merged_node, only_leaf=True)

                            # Break out of double-nested loop to reset iterators
                            merge_occured = True
                            break
                
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
              
                        # ========== Parent-Child Semantic Merging ==========

                        # If either is the root node, skip as this shouldn't really merge with any children
                        if node_i.is_RootGraphNode() or node_j.is_RootGraphNode():
                            continue

                        # Calculate Semantic Similarity
                        sem_con = SceneGraph3D.semantic_consistency(node_i.get_weighted_semantic_descriptor(), node_j.get_weighted_semantic_descriptor())

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
                
                # If we break out of inner loop, leave outer loop too
                if merge_occured:
                    break
                


    def node_retirement(self):
        # Iterate only through the direct children of the root node
        retired_ids = []
        for child in self.root_node.get_children()[:]:

            # If the time the child was first seen was too long ago OR
            # we've move substancially since then, Inactivate this node and descendents
            if self.times[-1] - child.get_time_first_seen() > self.max_t_active_for_node \
                or np.linalg.norm(self.poses[-1][:3,3] - child.get_first_pose()[:3,3]) > self.max_dist_active_for_node:

                # Pop this child off of the root node and put in our inactive nodes
                retired_ids += [child.get_id()]
                retired_ids += child.remove_from_graph_complete()
                self.inactive_nodes.append(child)

        if len(retired_ids) > 0:
            logger.info(f"[dark_magenta]Node Retirement[/dark_magenta]: {len(retired_ids)} nodes retired, including {retired_ids}.")