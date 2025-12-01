from __future__ import annotations

from collections import defaultdict
from .graph_node import GraphNode
from ..logger import logger
import numpy as np
from roman.params.system_params import SystemParams, SceneGraph3DParams
from .scene_graph_utils import *
from typeguard import typechecked

class SceneGraph3DBase:
    
    def __init__(self, params: SystemParams):
        self.root_node: GraphNode = GraphNode.create_node_if_possible(-1, None, np.zeros(0), np.zeros((0, 3), dtype=np.float64), [], is_RootGraphNode=True)
        
        # Save parameters
        self.system_params: SystemParams = params
        self.scene_graph_3D_params: SceneGraph3DParams = params.scene_graph_3D_params
        GraphNode.params = params.graph_node_params

        # Dictionaries to cache results of calculations for speed
        self.overlap_dict: defaultdict = defaultdict(lambda: defaultdict(lambda: None))
        self.shortest_dist_dict: defaultdict = defaultdict(lambda: defaultdict(lambda: None))
     
    # ================== Node Addition Methods ==================
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
            if not SceneGraph3DBase.check_if_nodes_are_somewhat_nearby(node, pos_parent_node):
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
                # If SceneGraph3D is using this, uncomment "if child.is_segment_or_inactive():"

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
            logger.debug(f"[cyan]Added[/cyan]: Node {node.get_id()} added to graph as child of Node {new_parent.get_id()} and parent of Nodes {[c.get_id() for c in new_children]}")
        else:
            logger.debug(f"[cyan]Added[/cyan]: Node {node.get_id()} added to graph as child of Node {new_parent.get_id()}")

        # Print any deleted nodes
        if len(deleted_ids) > 0:
            logger.info(f"[bright_red] Discard: [/bright_red] Node(s) {deleted_ids} removed as not enough remaining points with children removal.")

    @typechecked
    def calculate_best_likelihood_score(self, children_iou: list[float], children_enclosure: list[float], parent_iou: float, parent_encompassment: float, only_leaf: bool) -> float:

        # Make sure parent scores are within thresholds
        SceneGraph3DBase.check_within_bounds(parent_iou, (0, 1))
        SceneGraph3DBase.check_within_bounds(parent_encompassment, (0, 1))

        # If there is at least one child, calculate the best likelihood score using them
        expected_len = len(children_iou)
        best_child_score = 0.0
        if expected_len > 0 and not only_leaf: 
            # Assert each array is of the same length
            assert expected_len == len(children_enclosure)

            # Make sure each child value is within expected thresholds
            for i in range(len(children_iou)):
                SceneGraph3DBase.check_within_bounds(children_iou[i], (0, 1))
                SceneGraph3DBase.check_within_bounds(children_enclosure[i], (0, 1))

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
    
    # ================== Geometric Overlap Methods ==================
    def geometric_overlap(self, a: GraphNode, b: GraphNode) -> tuple[float, float | None, float | None]:

        # Calculate geometric overlap using hulls
        result = (None, None, None)
        if GraphNode.params.require_valid_convex_hull:
            result: tuple[float, float, float] = convex_hull_geometric_overlap(a.get_convex_hull(), b.get_convex_hull())

        # Use Voxel Grid for IOU if specified
        if not self.scene_graph_3D_params.use_convex_hull_for_iou:
            voxel_size = self.scene_graph_3D_params.voxel_size_for_voxel_grid_iou
            grid_a = a.get_voxel_grid(voxel_size)
            grid_b = b.get_voxel_grid(voxel_size)
            if grid_a is None or grid_b is None:
                raise RuntimeError("One or more Voxel Grids are None!")
            else:
                voxel_iou = grid_a.iou(grid_b)
            result = (voxel_iou, result[1], result[2]) 

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
            SceneGraph3DBase._purge_node_calculations_from_dict(self.overlap_dict, a)
            a._redo_convex_hull_geometric_overlap = False
        if b._redo_convex_hull_geometric_overlap:
            SceneGraph3DBase._purge_node_calculations_from_dict(self.overlap_dict, b)
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
    
    @staticmethod
    @typechecked
    def check_if_nodes_are_somewhat_nearby(a: GraphNode, b: GraphNode) -> bool:
        """ Returns False if centroid distance between nodes is greater than sum
        of two distances from centroid to OBB corner for each node. """

        # If one is the RootGraphNode, then obviously they are nearby
        if a.is_RootGraphNode() or b.is_RootGraphNode():
            return True

        # See if they are close enough
        c_a = a.get_centroid()
        c_b = b.get_centroid()
        f_a = np.linalg.norm(a.get_extent() / 2)
        f_b = np.linalg.norm(b.get_extent() / 2)
        if np.linalg.norm(c_a - c_b) > f_a + f_b: return False
        else: return True

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
            SceneGraph3DBase._purge_node_calculations_from_dict(self.shortest_dist_dict, a)
            a._redo_shortest_dist_between_convex_hulls = False
        if b._redo_shortest_dist_between_convex_hulls:
            SceneGraph3DBase._purge_node_calculations_from_dict(self.shortest_dist_dict, b)
            b._redo_shortest_dist_between_convex_hulls = False

        # Pull value from dictionary 
        result: float | None = self.shortest_dist_dict[a.get_id()][b.get_id()]

        # Calculate (or recalculate) the overlap if necessary
        if result is None:
            result: float = shortest_dist_between_convex_hulls(a.get_convex_hull(), b.get_convex_hull())
            self.shortest_dist_dict[a.get_id()][b.get_id()] = result

        return result