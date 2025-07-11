from __future__ import annotations

from enum import Enum
from itertools import chain, combinations
from ..map.observation import Observation
import numpy as np
from numpy.typing import NDArray
from ..object.segment import Segment
import open3d as o3d
import trimesh
from .scene_graph_3D import SceneGraph3D
from scipy.spatial import ConvexHull
from typeguard import typechecked

class GraphNode():
    # ================== Attributes ==================
    # The parent node to this node.
    parent_node: ParentGraphNode | None

    # As Clip embeddings can't be split and passed to children,
    # track embeddings specific to each node. Will be combined
    # with child embeddings before returning the final embedding.
    semantic_descriptors: list[tuple[np.ndarray, float]] = []

    # Points that relate only to this object and not any children.
    # Ex. points on a wall, whose children are a painting and a window.
    point_cloud: np.ndarray = np.zeros((0, 3), dtype=np.float64)

    # Keeps track of time this node (not any children) was seen. Note 
    # that last_updated just keeps track of when point cloud or semantic 
    # descriptors was changed, nothing else.
    first_seen: float
    last_updated: float

    # Keeps track of current time & pose
    curr_time: float
    curr_pose: np.ndarray

    # Keeps track of the first pose associated with this node (not any descendents)
    first_pose: np.ndarray

    # If true, this is a placeholder node that stores values for likelihood
    # as the position for a new node added to the graph.
    _is_dummy: bool = False

    # ================ Initialization =================
    @typechecked
    def __init__(self, parent_node: ParentGraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]],
                       point_cloud: np.ndarray, first_seen: float, last_updated: float, curr_time: float, 
                       first_pose: np.ndarray, curr_pose: np.ndarray):
        self.parent_node = parent_node
        self.semantic_descriptors = semantic_descriptors
        self.point_cloud = point_cloud
        self.first_seen = first_seen
        self.last_updated = last_updated
        self.curr_time = curr_time
        self.first_pose = first_pose
        self.curr_pose = curr_pose

        # Always cleanup incoming point cloud
        self.cleanup_point_cloud()

    # ==================== Getters ====================
    @typechecked
    def no_parent(self) -> bool:
        if self.parent_node is None: return True
        else: return False

    @typechecked
    def get_parent(self) -> ParentGraphNode:
        return self.parent_node
    
    def is_parent_or_child(self, other: GraphNode) -> bool:
        """ Returns true if self is the parent or child of other. """
        return self.is_parent(other) or self.is_child(other)
    
    @typechecked
    def is_parent(self, other: GraphNode) -> bool:
        """ Returns True if self is the parent of other. """
        if other in self.get_children():
            return True
        return False
    
    @typechecked
    def is_child(self, other: GraphNode) -> bool:
        """ Returns True if self is the child of other. """
        return other.is_parent(self)
    
    @typechecked
    def is_dummy(self) -> bool:
        return self._is_dummy
    
    @typechecked
    def get_convex_hull(self)-> trimesh.Trimesh:
        return SceneGraph3D.get_convex_hull_from_point_cloud(self.get_point_cloud())
    
    @typechecked
    def get_weighted_semantic_descriptor(self) -> np.ndarray:
        return self.calculate_weighted_semantic_descriptor(self.get_semantic_descriptors())
    
    @typechecked
    def is_descendent_or_ascendent(self, other) -> bool:
        """ Returns True if this node is a descendent or ascendent of other."""
        return self.is_descendent(other) or self.is_ascendent(other)
    
    @typechecked
    def is_descendent(self, other: GraphNode) -> bool:
        """ Returns True if self is an descendent of other."""
        return other.is_ascendent(self)
    
    @typechecked
    def is_ParentGraphNode(self) -> bool:
        return isinstance(self, ParentGraphNode)
    
    @typechecked
    def is_RootGraphNode(self) -> bool:
        return isinstance(self, RootGraphNode)

    @typechecked
    def is_LeafGraphNode(self) -> bool:
        return isinstance(self, LeafGraphNode)
    
    @typechecked
    def get_time_first_seen(self) -> float:
        return self.first_seen
    
    @typechecked
    def get_time_last_updated(self) -> float:
        return self.last_updated
    
    @typechecked
    def get_first_pose(self) -> np.ndarray:
        return self.first_pose
    
    def get_time_last_updated_recursive(self) -> float:
        """ Gets the most recent last updated time for self and descendents. """
        raise NotImplementedError("Use a child class, not GraphNode itself!")

    def get_point_cloud(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def get_semantic_descriptors(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def get_number_of_nodes(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def is_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if self is an ascendent of other."""
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def get_children(self) -> list[GraphNode] | None:
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    # ==================== Calculations ====================
    def calculate_weighted_semantic_descriptor(descriptors: list[tuple[NDArray[np.float64], float]]):
        # If we have no descriptors, raise an error
        if len(descriptors) == 0:
            raise ValueError("Cannot calculate weighted semantic descriptor when descriptors is empty!")

        # Extract the embeddings and weights
        embeddings, weights = zip(*descriptors)
        embeddings = np.array(embeddings, dtype=np.float64) 
        weights = np.array(weights, dtype=np.float64)     

        # Make sure the input descriptors are normalized
        for i in range(embeddings.shape[0]):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

        # Calculate the new final semantic descriptor as weighted average
        semantic_descriptor = np.average(embeddings, axis=0, weights=weights)
        semantic_descriptor /= np.linalg.norm(semantic_descriptor)
        return semantic_descriptor

    # ==================== Setters ====================
    @typechecked
    def set_parent(self, node: ParentGraphNode) -> None:
        self.parent_node = node

    @typechecked
    def set_is_dummy(self, is_dummy: bool):
        self._is_dummy = is_dummy

    # ==================== Manipulators ====================
    @typechecked
    def add_semantic_descriptors(self, descriptors: list[tuple[NDArray[np.float64], float]]) -> None:
        self.semantic_descriptors += descriptors
        self.last_updated = self.curr_time

    @typechecked
    def update_point_cloud(self, new_points: np.ndarray) -> None:
        # Skip math if no new points are included
        if new_points.shape[0] == 0: return

        # Check the input arry is the shape we expect
        if new_points.shape[1] !=3: raise ValueError(f"Point array in a non-supported shape: {new_points.shape()}")

        # Append them to our point cloud
        self.point_cloud = np.concatenate((self.point_cloud, new_points), axis=0)
        self.last_updated = self.curr_time

        # Remove any outliers
        self.cleanup_point_cloud()

    def cleanup_point_cloud(self):
        # Only clean-up if there are points to clean-up
        if self.point_cloud.shape[0] is not None:

            # Convert into o3d PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)

            # Remove statistical outliers
            # TODO: See if removing this has any negative impact -> pcd_sampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            # TODO: Tune these parameters, see if it has any effect
            pcd_pruned, _ = pcd.remove_statistical_outlier(10, 1.0)

            # Save the new outlier free point cloud
            if pcd_pruned.is_empty():
                self.point_cloud = None
            else:
                self.point_cloud = np.asarray(pcd_pruned.points)
            self.last_updated = self.curr_time
    
    @typechecked
    def remove_points_from_self(self, pc_to_remove: np.ndarray):
        # Create views for efficiency
        dtype = [('x', float), ('y', float), ('z', float)]
        view_pc = self.point_cloud.view(dtype)
        view_remove = pc_to_remove.view(dtype)

        # Remove points that are in pc_to_remove
        mask = np.isin(view_pc, view_remove, invert=True)
        if np.sum(mask) != mask.shape[0]:
            self.point_cloud = self.point_cloud[mask]
            self.last_updated = self.curr_time

    @typechecked
    def merge_with_node(self, other: GraphNode) -> GraphNode:
        """
        As opposed to merge_with_observation (which can just be called), this method 
        will take out self and other from the graph and return a new node. This new
        node needs to be inserted back into the graph by the SceneGraph3D.

        NOTE: other cannot be a descendent or ascendent of self!
        """

        # Remove both nodes (and all descendants) from the graph
        self.get_parent().remove_child(self)
        other.get_parent().remove_child(other)
        self.set_parent(None)
        other.set_parent(None)

        # Combine only semantic descriptors in the two nodes directly, not from children
        combined_descriptors = self.semantic_descriptors + other.semantic_descriptors

        # Do the same with point clouds specific to these two nodes, not children
        combined_pc = np.concatenate((self.point_cloud, other.point_cloud), axis=0)

        # Make a list of children
        combined_children = (self.get_children() + other.get_children())

        # Calculate the first seen time as earliest from the two nodes
        first_seen = min(self.get_time_first_seen(), other.get_time_first_seen())

        # Also calculate the first pose
        if first_seen == self.get_time_first_seen(): first_pose = self.get_first_pose()
        else: first_pose = other.get_first_pose()

        # Create a new node representing the merge
        if self.is_LeafGraphNode() and other.is_LeafGraphNode():
            new_node = LeafGraphNode(None, combined_descriptors, combined_pc, first_seen, 
                                     self.curr_time, self.curr_time, first_pose, self.curr_pose)
        else:
            new_node = ParentGraphNode(None, combined_descriptors, combined_pc, combined_children, 
                                       first_seen, self.curr_time, self.curr_time, first_pose, 
                                       self.curr_pose)
        return new_node
                
    def merge_with_observation(self, new_pc: np.ndarray, new_descriptor: np.ndarray | None) -> None:
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def remove_points(self, pc_to_remove: np.ndarray):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def remove_all_children(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def update_curr_time(self, curr_time: float):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def update_curr_pose(self, curr_pose: np.ndarray):
        raise NotImplementedError("Use a child class, not GraphNode itself!")

    # ==================== Dummy Nodes ====================     
    def add_dummy_nodes(self, only_leaf: bool = False):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def prune_dummy_nodes(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")

    # ==================== Iterator ====================
    def __iter__(self):
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            if isinstance(node, ParentGraphNode):
                stack.extend(node.child_nodes)

class ParentGraphNode(GraphNode):
    # ================== Attributes ==================
    child_nodes: list[GraphNode]

    # ================ Initialization =================
    @typechecked
    def __init__(self, parent_node: ParentGraphNode | None, semantic_descriptors: list[tuple[np.ndarray]], 
                 point_cloud: np.ndarray, child_nodes: list[GraphNode], first_seen: float, last_updated: float, 
                 curr_time: float, first_pose: np.ndarray, curr_pose: np.ndarray):
        super().__init__(parent_node, semantic_descriptors, point_cloud, 
                         first_seen, last_updated, curr_time, first_pose, curr_pose)
        self.child_nodes = child_nodes

    @classmethod
    def from_leaf_node(cls, leaf_node: LeafGraphNode):
        return cls(leaf_node.parent_node, leaf_node.semantic_descriptors,
                   leaf_node.get_point_cloud(), [], leaf_node.first_seen, 
                   leaf_node.last_updated, leaf_node.curr_time, 
                   leaf_node.first_pose, leaf_node.curr_pose)
    
    # ==================== Getters ==================== 
    @typechecked
    def get_time_last_updated_recursive(self) -> float:
        """ Gets the most recent last updated time for self and descendents. """
        most_recent_update = self.last_updated
        for child in self.child_nodes:
            child_last_updated = child.get_time_last_updated_recursive()
            if most_recent_update < child_last_updated:
                most_recent_update = child_last_updated
        return most_recent_update

    @typechecked
    def get_point_cloud(self) -> np.ndarray:
        full_pc = np.zeros((0, 3), dtype=np.float64)
        for child in self.child_nodes:
            full_pc = np.concatenate((full_pc, child.get_point_cloud()), dtype=np.float64)
        full_pc = np.concatenate((full_pc, self.point_cloud), dtype=np.float64)
        return full_pc
    
    @typechecked
    def get_semantic_descriptors(self):
        descriptors = []
        for child in self.child_nodes:
            descriptors += child.get_semantic_descriptors()
        descriptors += self.semantic_descriptors
        return descriptors
    
    @typechecked
    def get_number_of_nodes(self) -> int:
        num = 1
        for child in self.child_nodes:
            num += child.get_number_of_nodes(child)
        return num
    
    def is_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if self is an ascendent of other."""

        # Check if we are directly their parent
        for child in self.child_nodes:
            if child is other:
                return True

        # If not, see if we are a grandparent or higher up
        for child in self.child_nodes:
            if child.is_ascendent(other):
                return True    

        # Otherwise, we aren't
        return False
    
    @typechecked
    def get_children(self) -> list[GraphNode] | None:
        return self.child_nodes
    
    # ==================== Manipulators ====================
    @typechecked
    def merge_with_observation(self, new_pc: np.ndarray, new_descriptor: np.ndarray | None):
        """
        Args:
            new_pc (np.ndarray): Point Cloud in shape of (N, 3) in the global frame.
        """

        # Save the semantic embedding into this parent graph node
        if new_descriptor is not None: 
            self.add_semantic_descriptors([(new_descriptor, ConvexHull(new_pc).volume)])

        # Get convex hulls of each child
        hulls: list[trimesh.Trimesh] = []
        for child in self.child_nodes:
            hulls.append(child.get_convex_hull())
        
        # Get masks of which points fall into which hulls
        contain_masks = SceneGraph3D.find_point_overlap_with_hulls(new_pc, hulls, fail_on_multi_assign=True)

        # Based on point assignments, update each child node
        for i, child in enumerate(self.child_nodes):
            child_pc = new_pc[contain_masks[i],:]
            child.merge_with_observation(child_pc, None)
            
        # Find points that had no assignment
        num_mask_assignments = np.sum(contain_masks, axis=0)
        orphan_mask = np.where(num_mask_assignments == 0)[0]
        orphan_pc = new_pc[orphan_mask,:]

        # If there are at least one point in this orphan point cloud, add them to this node's cloud
        if orphan_pc.shape[0] > 1:
            self.update_point_cloud(orphan_pc)

    @typechecked
    def merge_child_with_self(self, other: GraphNode) -> None:
        # Make sure other is a child of self
        if not self.is_parent(other):
            raise ValueError("Cannot merge_child_with_self; node {other} is not a child of self!")
        
        # Add semantic descriptors specific to child (not from grandchidren) to this node
        self.add_semantic_descriptors(other.semantic_descriptors)
        
        # Do the same with point cloud specific to the child (not grandchildren)
        self.update_point_cloud(other.point_cloud)

        # Add grandchildren as children and add self as grandchildrens' parent
        for grandchild in other.get_children():
            self.add_child(grandchild)
            grandchild.set_parent(self)

        # Remove child
        other.set_parent(None)
        self.remove_child(other)

        # Just to be extra careful, remove children from child
        other.remove_all_children()

    @typechecked
    def remove_points(self, pc_to_remove: np.ndarray):
        self.remove_points_from_self(pc_to_remove)
        for child in self.child_nodes:
            child.remove_points(pc_to_remove)

    @typechecked
    def add_child(self, new_child: GraphNode):
        self.child_nodes.append(new_child)

    @typechecked
    def remove_child(self, child: GraphNode):
        if child in self.child_nodes:
            self.child_nodes.remove(child)
        else:
            raise ValueError(f"Tried to remove {child} from {self}, but {child} not in self.child_nodes: {self.child_nodes}")
        
    @typechecked
    def remove_all_children(self):
        self.child_nodes = []

    @typechecked
    def update_curr_time(self, curr_time: float):
        self.curr_time = curr_time
        for child in self.child_nodes:
            child.update_curr_time(curr_time)

    @typechecked
    def update_curr_pose(self, curr_pose: np.ndarray):
        self.curr_pose = curr_pose
        for child in self.child_nodes:
            child.update_curr_pose(curr_pose)
    
    # ==================== Dummy Nodes ====================
    def add_dummy_nodes(self, only_leaf: bool = False):
        # First, iterate through each child and create their dummy nodes
        for child in self.child_nodes:
            child.add_dummy_nodes(only_leaf)

        # If we only want to add dummy nodes to leaf nodes and we have at least
        # one child, then we skip ourselves
        if only_leaf and len(self.child_nodes) > 0:
            return

        # Create powerset of all children in the current node
        num_children = len(self.child_nodes)
        powerset = chain.from_iterable(combinations(range(num_children), r) for r in range(num_children + 1))

        # Iterate through each subset of child combinations and create a dummy node with those children
        for subset in powerset:

            # Create the node
            if len(subset) > 0:
                dummy_child_nodes = [self.child_nodes[i] for i in list(subset)]
                dummy_node = ParentGraphNode(self, [], np.zeros((0, 3), dtype=np.float64), dummy_child_nodes,
                                             self.curr_time, self.curr_time, self.curr_time, 
                                             self.curr_pose, self.curr_pose)
            else: 
                # For empty set, we need to create Leaf instead of Parent
                dummy_node = LeafGraphNode(self, [], np.zeros((0, 3), dtype=np.float64),
                                           self.curr_time, self.curr_time, self.curr_time, 
                                           self.curr_pose, self.curr_pose)

            # Label it as a dummy
            dummy_node.set_is_dummy(True)

            # Add it as one of our children
            self.add_child(dummy_child_nodes)

    def prune_dummy_nodes(self):
        # First, iterate through children and prune their dummy nodes
        for child in self.child_nodes:
            child.prune_dummy_nodes()

        # If we are a dummy, remove ourselves
        if self.is_dummy():

            # Reconnect children nodes to our parent
            for child in self.child_nodes:
                child.set_parent(self.get_parent())
                self.get_parent().add_child(child)
            self.remove_all_children()

            # Disconnect ourselves
            self.get_parent().remove_child(self)
            self.parent_node = None

class RootGraphNode(ParentGraphNode):
    # ================ Initialization =================
    @typechecked
    def __init__(self, parent_node: ParentGraphNode | None, semantic_descriptors: list[tuple[np.ndarray]], 
                 point_cloud: np.ndarray, child_nodes: list[GraphNode], first_seen: float, last_updated: float, 
                 curr_time: float, first_pose: np.ndarray, curr_pose: np.ndarray):
        super().__init__(parent_node, semantic_descriptors, point_cloud, child_nodes, 
                         first_seen, last_updated, curr_time, first_pose, curr_pose)

    # ==================== Setters ====================
    def set_parent(self, node: ParentGraphNode) -> None:
        raise RuntimeError("Calling set_parent on RootGraphNode, which should never happen!")

    def set_is_dummy(self, is_dummy: bool):
        raise RuntimeError("Calling set_is_dummy on RootGraphNode, which should never happen!")
            
class LeafGraphNode(GraphNode):

    # ================ Initialization =================
    @typechecked
    def __init__(self, parent_node: ParentGraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]],
                       point_cloud: np.ndarray, first_seen: float, last_updated: float, curr_time: float, 
                       first_pose: np.ndarray, curr_pose: np.ndarray):
        super().__init__(parent_node, semantic_descriptors, point_cloud, 
                         first_seen, last_updated, curr_time, first_pose, curr_pose)

    # ==================== Getters ==================== 
    def get_time_last_updated_recursive(self) -> float:
        return self.last_updated
    
    def get_point_cloud(self) -> np.ndarray:
        return self.point_cloud
    
    def get_semantic_descriptors(self):
        return self.semantic_descriptors
    
    def get_number_of_nodes(self) -> int:
        return 1
    
    def is_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if self is an ascendent of other."""
        return False
    
    def get_children(self) -> list[GraphNode] | None:
        return None

    # ==================== Manipulators ====================
    @typechecked
    def merge_with_observation(self, new_pc: np.ndarray, new_descriptor: np.ndarray | None):

        # Update the point cloud
        self.update_point_cloud(new_pc)

        # Update the semantic embedding
        if new_descriptor is not None:  
            self.add_semantic_descriptors([(new_descriptor, ConvexHull(new_pc).volume)])

    @typechecked
    def remove_points(self, pc_to_remove: np.ndarray):
        self.remove_points_from_self(pc_to_remove)

    def remove_all_children(self):
        pass

    @typechecked
    def update_curr_time(self, curr_time: float):
        self.curr_time = curr_time

    @typechecked
    def update_curr_pose(self, curr_pose: np.ndarray):
        self.curr_pose = curr_pose

    # ==================== Dummy Nodes ====================
    def add_dummy_nodes(self, only_leaf: bool = False):
        """
        Since no children, add a single dummy node as a child. only_leaf has
        no effect here, as we are a leaf node and thus will always add a dummy
        node.
        
        NOTE: In terms of implementation, we actually create an 
        intermediate node between this node and the parent and pass
        all data from this leaf to the intermediate node.
        """

        # Get all three nodes
        parent_node = self.get_parent()
        intermediate_node = ParentGraphNode.from_leaf_node(self)
        leaf_node = self

        # Reconnect nodes to put intermediate in the middle
        parent_node.add_child(intermediate_node)
        parent_node.remove_child(leaf_node)
        intermediate_node.add_child(leaf_node)
        leaf_node.set_parent(intermediate_node)

        # Label ourselves as the dummy
        self.set_is_dummy(True)

        # Set the rest of our data as if we were just initialized.
        self.semantic_descriptors = []
        self.point_cloud = np.zeros((0, 3), dtype=np.float64)
        self.first_seen = self.curr_time
        self.last_updated = self.curr_time
        self.curr_time = self.curr_time
        self.first_pose = self.curr_pose
        self.curr_pose = self.curr_pose

    def prune_dummy_nodes(self):
        if self.is_dummy():
            self.get_parent().remove_child(self)
            self.parent_node = None