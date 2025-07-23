from __future__ import annotations

from .graph_node import GraphNode, RootGraphNode
from .hull_methods import find_point_overlap_with_hulls, get_convex_hull_from_point_cloud, \
convex_hull_geometric_overlap, shortest_dist_between_convex_hulls
from itertools import chain, combinations
from matplotlib import rcParams
import matplotlib.pyplot as plt
from ..map.observation import Observation
import networkx as nx
import numpy as np
import pickle
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
        self.viewer = SceneGraphViewer()

    def assert_root_node_has_no_parent(self):
        if self.root_node.no_parent() is False:
            raise RuntimeError("Top-level nodes list of SceneGraph3D should only have root nodes!")

    @typechecked
    def len(self) -> int:
        return self.root_node.get_number_of_nodes()

    @typechecked
    def update(self, time: float | np.longdouble, pose: np.ndarray, observations: list[Observation]):
        print(f"Called with {len(observations)} observations")
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
        print(f"Remaining valid observations: {len(valid_obs)}")
        
        # Associate observations with any current object nodes
        associated_pairs = self.hungarian_assignment(valid_obs)

        # Merge associated pairs
        print(f"Number of assocations to merge: {len(associated_pairs)}")
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

        # Merge nodes that can be merged, and infer higher level ones
        self.merging_and_generation()

        # Run node retirement
        self.node_retirement()

        # Update the viewer
        self.viewer.update(self)

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
        iou_pass: bool = bool(iou >= self.min_iou_for_association)
        sem_pass: bool = bool(sem_con >= self.min_sem_con_for_association)

        # Print warning if IOU overlap is high and semantic is far off.
        # Could be indicative of underlying bug in algorithm.
        if iou_pass and not sem_pass:
            raise RuntimeError("WARNING: IOU overlap is high enough for association, but semantic is too far off. This shouldn't happen!")      

        # Return true if both requirements are fulfilled
        return iou_pass and sem_pass
    
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


        # Create a new node for this observation
        new_node = GraphNode(self.root_node.request_new_ID(), None, [], np.zeros((0, 3), dtype=np.float64), 
                             [node_i, node_j], first_seen, self.times[-1], self.times[-1], first_pose, self.poses[-1])
        
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
                print(f"Observation added to graph as Node {node.get_id()} and child of Node {node.get_parent().get_id()}")
                break


    @typechecked
    def add_new_node_to_graph(self, node: GraphNode, only_leaf=False):
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

        # Keep track of the current likeliest node
        best_likelihood_score = -np.inf
        best_likelihood_position = None

        # Iterate through each node, considering ourselves as their potential child
        pos_parent_node: RootGraphNode = self.root_node
        node_queue = []
        while pos_parent_node is not None:

            # See if this node is anywhere near us. If not, we obviously can't be their child
            if not self.check_if_nodes_are_somewhat_nearby(node, pos_parent_node):
                continue

            # Since we're nearby this, add all of this nodes children to the queue as well
            # so that we check if we should be their children as well
            node_queue += pos_parent_node.get_children()

            # If we only want to add ourselves as a LeafNode, make sure pos_parent_node has no children
            if only_leaf:
                if len(pos_parent_node.get_children()) > 0:
                    continue

            # Get the convex hull comprising of all the children, and one for the parent
            hull_all_children: list[trimesh.Trimesh] = [child.get_convex_hull() for child in node.get_children()]
            hull_parent: trimesh.Trimesh = node.get_parent().get_convex_hull()

            # Get the volumes of each of the children
            children_volumes = [child.get_volume() for child in node.get_children()]
            
            # Add similarities for geometric overlaps with children
            children_iou, children_enclosure = [], []
            for hull in hull_all_children:
                iou, child_enc, _ = convex_hull_geometric_overlap(hull, hull_new)
                children_iou.append(iou)
                children_enclosure.append(child_enc)

            # Add similarities for geometric overlaps with parent
            parent_iou, _, parent_encompassment = convex_hull_geometric_overlap(hull_parent, hull_new)

            # Calculate semantic similarity
            if node.get_weighted_semantic_descriptor() is not None:

                # Calculate similarity with children
                children_sem_sim = []
                for child in pos_parent_node.get_children():
                    children_sem_sim.append(SceneGraph3D.semantic_consistency(node.get_weighted_semantic_descriptor(), child.get_weighted_semantic_descriptor()))

                # Calculate similarity with parent
                if not node.get_parent().is_RootGraphNode():
                    parent_sem_sim = SceneGraph3D.semantic_consistency(node.get_weighted_semantic_descriptor(), pos_parent_node.get_weighted_semantic_descriptor())
                else:
                    # Neutral similarity with root node, as it represents everything
                    parent_sem_sim = 0.5
            else:
                # Give it a neutral similarity if there is no descriptor
                children_sem_sim = []
                parent_sem_sim = 0.5

            # Calculate the final likelihood score
            score, subset = self.calculate_best_likelihood_score(children_iou, children_enclosure, children_sem_sim, children_volumes,
                                                                        parent_iou, parent_encompassment, parent_sem_sim)
            
            # If this is the best score so far, keep it
            if score >= best_likelihood_score:
                best_likelihood_score = score
                best_likelihood_position = (pos_parent_node, subset)

            # After checking this node, move to check the next one
            if len(node_queue) >= 1:
                pos_parent_node: GraphNode = node_queue[0]
                node_queue = node_queue[1:]
            else:
                pos_parent_node = None
        
        # Place our already fully formed node into its spot
        new_parent = best_likelihood_position[0]
        new_children = new_parent.get_children()[best_likelihood_position[1]]

        print(f"Node {node.get_id()} added to graph as child of Node {new_parent.get_id()}")
        new_parent.add_child(node)
        node.set_parent(new_parent)
        node.add_children(new_children)

        # Resolve overlapping point clouds
        self.resolve_overlapping_point_clouds()

    @typechecked
    def calculate_best_likelihood_score(self, children_iou: list[float], children_enclosure: list[float], children_sem_sim: list[float],
        children_volumes: list[float], parent_iou: float, parent_encompassment: float, parent_sem_sim: float) -> tuple[float, tuple[int, ...]]:

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
        num_children = len(self.child_nodes)
        powerset = chain.from_iterable(combinations(range(num_children), r) for r in range(num_children + 1))

        # Iterate through each subset of child combinations and create a dummy node with those children
        for subset in powerset:
            # Get the values for this specific combination of children
            rel_child_iou = children_iou[subset]
            rel_child_enc = children_enclosure[subset]
            rel_child_sem_sim = children_sem_sim[subset]
            rel_volumes = children_volumes[subset]

            # Get a weighted average of them based on each childs volume
            temp_child_iou = np.average(rel_child_iou, axis=0, weights=rel_volumes)
            temp_child_enc = np.average(rel_child_enc, axis=0, weights=rel_volumes)
            temp_child_sem_sim = np.average(rel_child_sem_sim, axis=0, weights=rel_volumes)

            # Calculate the final score
            score = temp_child_iou + temp_child_enc + temp_child_sem_sim + parent_iou + parent_encompassment + parent_sem_sim

            # If this is the best so far, keep track of it
            if score >= best_score:
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
            raise ValueError("Value {a} is outside the range of [0, 1] inclusive.")
    
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
                        longest_line_node_i = node_i.get_longest_line_size()
                        longest_line_node_j = node_j.get_longest_line_size()
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

                        # If they already have a shared parent that isn't the RootGraphNode, no point checking this...
                        if node_i.get_parent() == node_j.get_parent() and not node_i.get_parent().is_RootGraphNode():
                            pass
                        
                        # If ratio of shorest distance to object length is within larger threshold AND
                        # the semantic embedding is close enough to assume they could have a shared parent...
                        elif dist_to_object_length < self.ratio_dist2length_threshold_higher_level_object_inference and \
                            sem_con < self.min_sem_con_for_higher_level_object_inference:
                            
                            # Disconnect both of these nodes from the graph
                            node_i.get_parent().remove_child(node_i)
                            node_j.get_parent().remove_child(node_j)
                            node_i.set_parent(None)
                            node_j.set_parent(None)

                            # Create a new parent node with these as children
                            first_seen = min(node_i.get_time_first_seen(), node_j.get_time_first_seen())
                            if first_seen == node_i.get_time_first_seen(): first_pose = node_i.get_first_pose()
                            else: first_pose = node_j.get_first_pose()
                            inferred_parent_node = GraphNode(self.root_node.request_new_ID(), None, [], np.zeros((0, 3), dtype=np.float64), 
                                                             [node_i, node_j], first_seen, self.times[-1], self.times[-1], first_pose, self.poses[-1])
                            
                            # Tell the children who their new parent is
                            node_i.set_parent(inferred_parent_node)
                            node_j.set_parent(inferred_parent_node)

                            # Add this inferred node back to the graph
                            print(f"Inferring new high-level Parent Node {inferred_parent_node.get_id()} from Node {node_i.get_id()} and Node {node_j.get_id()}")
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
        def add_node_recursive(node: GraphNode, parent_id=None):
            node_id = node.get_id()
            label = getattr(node, "get_label", lambda: f"{node_id}")()
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
    
    def save_graph_to_file(self, file_path: str):
        with open(file_path, 'wb') as file:
            pickle.dump(self.root_node, file)