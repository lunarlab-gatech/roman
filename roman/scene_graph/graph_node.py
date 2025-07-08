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

    # If true, this is a placeholder node that stores values for likelihood
    # as the position for a new node added to the graph.
    is_dummy: bool = False

    # ================ Initialization =================
    @typechecked
    def __init__(self, parent_node: ParentGraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]],
                       point_cloud: np.ndarray):
        self.parent_node = parent_node
        self.semantic_descriptors = semantic_descriptors
        self.point_cloud = point_cloud

    # ==================== Getters ====================
    @typechecked
    def no_parent(self) -> bool:
        if self.parent_node is None: return True
        else: return False

    @typechecked
    def get_parent(self) -> ParentGraphNode:
        return self.parent_node
    
    @typechecked
    def get_is_dummy(self) -> bool:
        return self.is_dummy
    
    @typechecked
    def get_convex_hull(self)-> trimesh.Trimesh:
        return SceneGraph3D.get_convex_hull_from_point_cloud(self.get_point_cloud())
    
    @typechecked
    def get_weighted_semantic_descriptor(self) -> np.ndarray:
        return self.calculate_weighted_semantic_descriptor(self.get_semantic_descriptors())
    
    def get_point_cloud(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def get_semantic_descriptors(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def get_number_of_nodes(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    # ==================== Calculations ====================
    def calculate_weighted_semantic_descriptor(descriptors: list[tuple[NDArray[np.float64], float]]):
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
        self.is_dummy = is_dummy

    # ==================== Manipulators ====================
    @typechecked
    def add_semantic_descriptors(self, descriptors: list[tuple[NDArray[np.float64], float]]) -> None:
        self.semantic_descriptors += descriptors

    @typechecked
    def update_point_cloud(self, new_points: np.ndarray, pose: np.ndarray) -> None:
        # Skip math if no new points are included
        if new_points.shape[0] == 0: return

        # Check the input arry is the shape we expect
        if new_points.shape[1] !=3: raise ValueError(f"Point array in a non-supported shape: {new_points.shape()}")

        # Convert observed points to global frame
        R = pose[:3,:3]
        T = pose[:3,3].reshape((3,1))
        points_obs_body = new_points.T
        num_points_obs = points_obs_body.shape[1]
        points_obs_world = R @ points_obs_body + np.repeat(T, num_points_obs, axis=1)

        # Append them to our point cloud
        self.point_cloud = np.concatenate((self.points, points_obs_world.T), axis=0)

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
    
    @typechecked
    def remove_points_from_self(self, pc_to_remove: np.ndarray):
        # Create views for efficiency
        dtype = [('x', float), ('y', float), ('z', float)]
        view_pc = self.point_cloud.view(dtype)
        view_remove = pc_to_remove.view(dtype)

        # Remove points that are in pc_to_remove
        mask = np.isin(view_pc, view_remove, invert=True)
        self.point_cloud = self.point_cloud[mask]
                
    def update_node(self, new_pc: np.ndarray, new_pc_pose: np.ndarray, new_descriptor: np.ndarray | None):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def remove_points(self, pc_to_remove: np.ndarray):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    # ==================== Dummy Nodes ====================
    def add_dummy_nodes(self):
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
                 point_cloud: np.ndarray, child_nodes: list[GraphNode]):
        super().__init__(parent_node, semantic_descriptors, point_cloud)
        self.child_nodes = child_nodes

    @classmethod
    def from_leaf_node(cls, leaf_node: LeafGraphNode):
        return cls(leaf_node.parent_node, leaf_node.semantic_descriptors,
                   leaf_node.get_point_cloud(), [])
    
    # ==================== Getters ==================== 
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
    
    # ==================== Manipulators ====================
    @typechecked
    def update_node(self, new_pc: np.ndarray, new_pc_pose: np.ndarray, new_descriptor: np.ndarray | None):

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
            child.update_node(child_pc, new_pc_pose, None)
            
        # Find points that had no assignment
        num_mask_assignments = np.sum(contain_masks, axis=0)
        orphan_mask = np.where(num_mask_assignments == 0)[0]
        orphan_pc = new_pc[orphan_mask,:]

        # If there are at least one point in this orphan point cloud, add them to this node's cloud
        if orphan_pc.shape[0] > 1:
            self.update_point_cloud(orphan_pc, new_pc_pose)

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
    
    # ==================== Dummy Nodes ====================
    def add_dummy_nodes(self):
        # First, iterate through each child and create their dummy nodes
        for child in self.child_nodes:
            child.add_dummy_nodes()

        # Create powerset of all children in the current node
        num_children = len(self.child_nodes)
        powerset = chain.from_iterable(combinations(range(num_children), r) for r in range(num_children + 1))

        # Iterate through each subset of child combinations and create a dummy node with those children
        for subset in powerset:

            # Create the node
            if len(subset) > 0:
                dummy_child_nodes = [self.child_nodes[i] for i in list(subset)]
                dummy_node = ParentGraphNode(self, [], np.zeros((0, 3), dtype=np.float64), dummy_child_nodes)
            else: 
                # For empty set, we need to create Leaf instead of Parent
                dummy_node = LeafGraphNode(self, [], np.zeros((0, 3), dtype=np.float64))

            # Label it as a dummy
            dummy_node.set_is_dummy(True)

            # Add it as one of our children
            self.add_child(dummy_child_nodes)

class LeafGraphNode(GraphNode):

    # ================ Initialization =================
    @typechecked
    def __init__(self, parent_node: ParentGraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]],
                       point_cloud: np.ndarray):
        super().__init__(parent_node, semantic_descriptors, point_cloud)

    # ==================== Getters ==================== 
    def get_number_of_nodes(self) -> int:
        return 1
    
    def get_semantic_descriptors(self):
        return self.semantic_descriptors
    
    def get_point_cloud(self) -> np.ndarray:
        return self.point_cloud

    # ==================== Manipulators ====================
    @typechecked
    def update_node(self, new_pc: np.ndarray, new_pc_pose: np.ndarray, new_descriptor: np.ndarray | None):
        # Update the point cloud
        self.update_point_cloud(new_pc, new_pc_pose)

        # Update the semantic embedding
        if new_descriptor is not None:  
            self.add_semantic_descriptors([(new_descriptor, ConvexHull(new_pc).volume)])

    @typechecked
    def remove_points(self, pc_to_remove: np.ndarray):
        self.remove_points_from_self(pc_to_remove)

    # ==================== Dummy Nodes ====================
    def add_dummy_nodes(self):
        """
        Since no children, add a single dummy node as a child. 
        
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