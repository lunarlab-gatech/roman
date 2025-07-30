from __future__ import annotations

from .hull_methods import get_convex_hull_from_point_cloud, find_point_overlap_with_hulls, longest_line_of_point_cloud
from .id_manager import IDManager
from .logger import logger
import numpy as np
from numpy.typing import NDArray
from robotdatapy.camera import xyz_2_pixel
from robotdatapy.transform import transform
import trimesh
from typing import Iterator
from scipy.spatial import ConvexHull
from typeguard import typechecked

class GraphNode():
    # ================== Attributes ==================
    # Node ID
    id: int

    # The parent node to this node.
    parent_node: GraphNode | None

    # Any child nodes we might have
    child_nodes: list[GraphNode]

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

    # Holds values for reuse and to avoid recalculating them
    _convex_hull: trimesh.Trimesh = None
    _point_cloud: np.ndarray = None
    _longest_line_size: float = None
    _centroid: np.ndarray = None

    # ================ Initialization =================
    @typechecked
    def __init__(self, id: int, parent_node: GraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]], 
                 point_cloud: np.ndarray, child_nodes: list[GraphNode], first_seen: float, last_updated: float, 
                 curr_time: float, first_pose: np.ndarray, curr_pose: np.ndarray):
        
        self.id = id
        self.parent_node = parent_node
        self.semantic_descriptors = semantic_descriptors
        self.point_cloud = point_cloud
        self.child_nodes = child_nodes
        self.first_seen = first_seen
        self.last_updated = last_updated
        self.curr_time = curr_time
        self.first_pose = first_pose
        self.curr_pose = curr_pose

        # Always cleanup incoming point cloud
        self.cleanup_point_cloud()

    # ==================== Getters ====================
    def __str__(self):
        """ Print a human-readable explanation of the current node. """
        semantic_count = len(self.semantic_descriptors)
        semantic_weights = [round(w, 3) for _, w in self.semantic_descriptors]
        point_count = self.point_cloud.shape[0]

        return (
            f"GraphNode(\n"
            f"  Id: {id(self)}\n"
            f"  Parent: {'None' if self.parent_node is None else 'Present'}\n"
            f"  Dummy: {self._is_dummy}\n"
            f"  Semantic Descriptors: {semantic_count} descriptors\n"
            f"    Weights: {semantic_weights}\n"
            f"  Point Cloud: {point_count} points\n"
            f"  First Seen: {self.first_seen:.2f}, Last Updated: {self.last_updated:.2f}\n"
            f"  Current Time: {self.curr_time:.2f}\n"
            f"  First Pose: {np.array2string(self.first_pose, precision=2, separator=', ')}\n"
            f"  Current Pose: {np.array2string(self.curr_pose, precision=2, separator=', ')}\n"
            f")")

    @typechecked
    def get_id(self) -> int:
        return self.id

    @typechecked
    def no_parent(self) -> bool:
        if self.parent_node is None: return True
        else: return False

    @typechecked
    def get_parent(self) -> GraphNode:
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
    def get_convex_hull(self)-> trimesh.Trimesh | None:
        if self._convex_hull is None:
            self._convex_hull = get_convex_hull_from_point_cloud(self.get_point_cloud())
        return self._convex_hull
    
    @typechecked
    def get_weighted_semantic_descriptor(self) -> np.ndarray | None:
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
    def is_RootGraphNode(self) -> bool:
        return isinstance(self, RootGraphNode)

    @typechecked
    def get_time_first_seen(self) -> float:
        return self.first_seen
    
    @typechecked
    def get_time_last_updated(self) -> float:
        return self.last_updated
    
    @typechecked
    def get_first_pose(self) -> np.ndarray:
        return self.first_pose
    
    @typechecked
    def reprojected_bbox(self, pose: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
        """Calculates the bounding box for a graph node given camera extrinsics & intrinsics"""

        # Skip bbox calculation if we have no points
        points = self.get_point_cloud()
        if points.shape[0] == 0: return None

        # Calculate points in camera frame and prune those behind the camera
        points_c = transform(np.linalg.inv(pose), points, axis=0)
        points_c = points_c[points_c[:,2] >= 0]
        if points_c.shape[0] == 0: return None

        # Convert xyz to pixels and prune those outside of the image
        pixels = xyz_2_pixel(points_c, K)
        pixels = pixels[np.bitwise_and(pixels[:,0] >= 0, pixels[:,0] < width), :]
        pixels = pixels[np.bitwise_and(pixels[:,1] >= 0, pixels[:,1] < height), :]
        if pixels.shape[0] == 0: return None

        # Get upper left and lower right of bounding box
        upper_left = np.max([np.min(pixels, axis=0).astype(int), [0, 0]], axis=0)
        lower_right = np.min([np.max(pixels, axis=0).astype(int), [width, height]], axis=0)

        # Check for wierd bugs
        if lower_right[0] - upper_left[0] <= 0 or lower_right[1] - upper_left[1] <= 0:
            raise RuntimeError("lower_right[0] - upper_left[0] <= 0 or lower_right[1] - upper_left[1] <= 0 in reprojected_bbox!")
        
        # Return the bounding box corners
        return upper_left, lower_right
    
    @typechecked
    def request_new_ID(self) -> int:
        return self.parent_node.request_new_ID()
    
    @typechecked
    def get_longest_line_size(self) -> float:
        if self._longest_line_size is None:
            self._longest_line_size = longest_line_of_point_cloud(self.get_point_cloud())
        return self._longest_line_size
    
    @typechecked
    def get_centroid(self) -> np.ndarray[float]:
        if self._centroid is None:
            self._centroid = np.mean(self.get_point_cloud(), axis=0)
        return self._centroid
    
    @typechecked
    def get_volume(self) -> float:
        if self.get_convex_hull() is None:
            raise RuntimeError(f"Trying to get volume of ConvexHull for Node {self.get_id()}, but there isn't a valid ConvexHull!")
        if not self.get_convex_hull().is_watertight:
            raise RuntimeError(f"Trying to get volume of ConvexHull for Node {self.get_id()}, but its not watertight!")
        return self.get_convex_hull().volume
    
    @typechecked
    def get_time_last_updated_recursive(self) -> float:
        """ Gets the most recent last updated time for self and descendents. """
        most_recent_update = self.last_updated
        for child in self.get_children():
            child_last_updated = child.get_time_last_updated_recursive()
            if most_recent_update < child_last_updated:
                most_recent_update = child_last_updated
        return most_recent_update

    @typechecked
    def get_point_cloud(self) -> np.ndarray:
        if self._point_cloud is None:
            # Re-calculate the point cloud since its been changed
            full_pc = np.zeros((0, 3), dtype=np.float64)
            for child in self.get_children():
                full_pc = np.concatenate((full_pc, child.get_point_cloud()), dtype=np.float64)
            self._point_cloud = np.concatenate((full_pc, self.point_cloud), dtype=np.float64)
            num_points = self._point_cloud.shape[0]
            if num_points < 4:
                logger.debug(f"[bright_red]WARNING[/bright_red]: Current point cloud is {num_points} for Node {self.get_id()}")
            else:
                logger.debug(f"[bright_yellow]UPDATE[/bright_yellow]: Current point cloud is {num_points} for Node {self.get_id()}")
        #np.save(f"/home/dbutterfield3/roman/debug_pc/node_{self.get_id()}_pc.npy", self._point_cloud)
        return self._point_cloud
    
    @typechecked
    def get_semantic_descriptors(self):
        descriptors = []
        for child in self.get_children():
            descriptors += child.get_semantic_descriptors()
        descriptors += self.semantic_descriptors
        return descriptors
    
    @typechecked
    def get_number_of_nodes(self) -> int:
        num = 1
        for child in self.get_children():
            num += child.get_number_of_nodes()
        return num
    
    @typechecked
    def is_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if self is an ascendent of other."""

        # Check if we are directly their parent
        for child in self.get_children():
            if child is other:
                return True

        # If not, see if we are a grandparent or higher up
        for child in self.get_children():
            if child.is_ascendent(other):
                return True    

        # Otherwise, we aren't
        return False
    
    @typechecked
    def get_children(self) -> list[GraphNode]:
        return self.child_nodes
    
    # ==================== Calculations ====================
    def calculate_weighted_semantic_descriptor(self, descriptors: list[tuple[NDArray[np.float64], float]]) -> np.ndarray | None:
        # If we have no descriptors, just return None
        if len(descriptors) == 0:
            return None

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
    def set_parent(self, node: GraphNode | None) -> None:
        self.parent_node = node

    # ==================== Manipulators ====================
    @typechecked
    def remove_from_graph(self) -> set[GraphNode]:
        """ Does so by disconnecting self from parent both ways. Returns any remaining parents that need to be deleted now. """
        if self.is_RootGraphNode():
            raise RuntimeError("Can't call remove_from_graph() on RootGraphNode!")

        logger.info(f"remove_from_graph: Currently on Node {self.get_id()}")
        to_delete = self.get_parent().remove_child(self)
        self.set_parent(None)
        return to_delete
    
    @typechecked
    def remove_from_graph_complete(self) -> None:
        """ Does so by disconnecting self from parent both ways. Also immediately deletes any parent nodes that are now invalid. """

        to_delete = self.remove_from_graph()
        while to_delete:
            to_delete.update(to_delete.pop().remove_from_graph())

    @typechecked
    def add_semantic_descriptors(self, descriptors: list[tuple[np.ndarray, float]]) -> None:
        self.semantic_descriptors += descriptors
        self.last_updated = self.curr_time

    @typechecked
    def update_point_cloud(self, new_points: np.ndarray) -> None:
        # Skip math if no new points are included
        if new_points.shape[0] == 0: return

        # Check the input arry is the shape we expect
        if new_points.shape[1] != 3: raise ValueError(f"Point array in a non-supported shape: {new_points.shape()}")

        # Append them to our point cloud
        self.point_cloud = np.concatenate((self.point_cloud, new_points), axis=0)
        self.last_updated = self.curr_time

        # Remove any outliers
        self.cleanup_point_cloud()

        # Reset point cloud dependent saved variables
        logger.debug(f"update_point_cloud(): Resetting Point Cloud for Node {self.get_id()}")
        self.reset_saved_vars_safe()

    def cleanup_point_cloud(self):
        """ Remove duplicates from point-cloud. """

        # Only clean-up if there are points to clean-up
        if self.point_cloud.shape[0] > 0:

            # Turn off Outlier Removal, as we'd probably need to do it on the point cloud from node.get_point_cloud()
            # for it to actually work properly... #TODO: Implement this maybe...

            # Convert into o3d PointCloud
            #pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(self.point_cloud)

            # Remove statistical outliers
            # TODO: See if removing this has any negative impact -> pcd_sampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            # TODO: Tune these parameters, see if it has any effect
            #pcd_pruned, _ = pcd.remove_statistical_outlier(10, 1.0)
            
            # Save the new outlier free point cloud (while also removing duplicates)
            self.point_cloud = np.unique(self.point_cloud, axis=0)
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
        self.remove_from_graph_complete()
        other.remove_from_graph_complete()

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
        new_node = GraphNode(self.get_id(), None, combined_descriptors, combined_pc, combined_children, 
                             first_seen, self.curr_time, self.curr_time, first_pose, self.curr_pose)
            
        # Tell our children who their new parent is
        for child in combined_children:
            child.set_parent(new_node)
        return new_node

    def reset_saved_vars(self) -> set[GraphNode]:
        """ 
        Wipes saved variables since they need to be recalculated. 
        Returns list of nodes that are no longer valid and should be removed. 
        """
        
        # Wipe all variables
        self._convex_hull = None
        self._point_cloud = None
        self._longest_line_size = None
        self._centroid = None

        # Track nodes that might need to be deleted...
        to_delete = set()

        # Check if we can still make a ConvexHull...
        if self.get_convex_hull() is None:
            logger.debug(f"We CANNOT get Convex Hull for Node {self.get_id()}, so plan to delete")
            to_delete.add(self)
        else:
            logger.debug(f"We can still get Convex Hull for Node {self.get_id()}")

        # Reset variables in parents and get any of those that need to be deleted.
        if self.parent_node is not None:
            logger.debug(f"reset_saved_vars(): Resetting Point Cloud for Parent node {self.parent_node.get_id()}")
            to_delete.update(self.parent_node.reset_saved_vars())

        # Return nodes that need to be deleted
        return to_delete
    
    def reset_saved_vars_safe(self) -> set[GraphNode]:
        """ 
        Similar to reset_saved_vars(), but called if points were only
        possibly added to a node. Thus, no need to check node validity
        or return nodes that might need to be deleted.
        """
        
        # Wipe all variables
        self._convex_hull = None
        self._point_cloud = None
        self._longest_line_size = None
        self._centroid = None

        # Reset variables in parents 
        if self.parent_node is not None:
            self.parent_node.reset_saved_vars_safe()

    @typechecked
    def merge_with_observation(self, new_pc: np.ndarray, new_descriptor: np.ndarray | None) -> None:
        """
        Args:
            new_pc (np.ndarray): Point Cloud in shape of (N, 3) in the global frame.

        Returns:
            Set of Graph nodes that might need to be deleted.
        """

        # Save the semantic embedding into this parent graph node
        if new_descriptor is not None: 
            self.add_semantic_descriptors([(new_descriptor, ConvexHull(new_pc).volume)])

        # Get convex hulls of each child
        hulls: list[trimesh.Trimesh] = []
        for child in self.get_children():
            hulls.append(child.get_convex_hull())
        
        # Get masks of which points fall into which hulls
        contain_masks = find_point_overlap_with_hulls(new_pc, hulls, fail_on_multi_assign=True)

        # Based on point assignments, update each child node
        for i, child in enumerate(self.get_children()):
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
        if not self.is_parent(other) or not other in self.get_children():
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
        other.remove_from_graph_complete()
        logger.debug(f"[bright_blue]Parent-Child Merge[/bright_blue]: Merged Node {other.get_id()} into Parent Node {self.get_id()}")

    @typechecked
    def remove_points_from_self(self, pc_to_remove: np.ndarray) -> set[GraphNode]:
        """ Returns any nodes to delete if they are invalidated by this remove. """
        # Create views for efficiency
        dtype = [('x', float), ('y', float), ('z', float)]
        view_pc = self.point_cloud.view(dtype)
        view_remove = pc_to_remove.view(dtype)

        # Remove points that are in pc_to_remove
        mask = np.isin(view_pc, view_remove, invert=True).squeeze(axis=1)
        if np.sum(mask) != mask.shape[0]:
            num_points_before = self.point_cloud.shape[0]
            self.point_cloud = self.point_cloud[mask]
            num_points_after = self.point_cloud.shape[0]
            logger.info(f"NODE: {self.get_id()}, POINT CLOUD UPDATED from {num_points_before} to {num_points_after}")
            self.last_updated = self.curr_time
            logger.debug(f"remove_points_from_self(): Resetting Point Cloud for Node {self.get_id()}")
            return self.reset_saved_vars()

        # Otherwise, we don't want to be deleted
        return set()

    @typechecked
    def remove_points(self, pc_to_remove: np.ndarray) -> set[GraphNode]:
        """ Returns all nodes that are no longer valid after their points are removed. """
        nodes_to_delete = set()

        # Start with children, as parents validity depends on if children's clouds are altered
        for child in self.child_nodes:
            nodes_to_delete.update(child.remove_points(pc_to_remove))

        # Now, delete points from self
        nodes_to_delete.update(self.remove_points_from_self(pc_to_remove))
        return nodes_to_delete
    
    @typechecked
    def remove_points_complete(self, pc_to_remove: np.ndarray) -> None:
        """ Remove points and then delete any nodes that are no longer valid. """
        to_delete = self.remove_points(pc_to_remove)
        while to_delete:
            to_delete.update(to_delete.pop().remove_from_graph())

    @typechecked
    def add_child(self, new_child: GraphNode) -> None:
        if new_child in self.child_nodes:
            raise ValueError("Tried to add a child node that is already a child of this GraphNode!")
        self.child_nodes.append(new_child)
        logger.debug(f"add_child(): Resetting Point Cloud for Node {self.get_id()}")
        self.reset_saved_vars_safe()

    @typechecked
    def add_children(self, new_children: list[GraphNode]) -> None:
        for new_child in new_children:
            self.add_child(new_child)
    
    @typechecked
    def remove_child(self, child: GraphNode) -> set[GraphNode]:
        if child in self.child_nodes:
            self.child_nodes.remove(child)
            logger.debug(f"remove_child(): Resetting Point Cloud for Node {self.get_id()}")
            return self.reset_saved_vars()
        else:
            raise ValueError(f"Tried to remove {child} from {self}, but {child} not in self.child_nodes: {self.child_nodes}")
        
    @typechecked
    def remove_children(self, to_remove: list[GraphNode]) -> set[GraphNode]:
        to_delete = set()
        for child in to_remove:
            to_delete.update(self.remove_child(child))
        return to_delete
        
    @typechecked
    def update_curr_time(self, curr_time: float):
        self.curr_time = curr_time
        for child in self.get_children():
            child.update_curr_time(curr_time)

    @typechecked
    def update_curr_pose(self, curr_pose: np.ndarray):
        self.curr_pose = curr_pose
        for child in self.get_children():
            child.update_curr_pose(curr_pose)

    # ==================== Iterator ====================
    def __iter__(self) -> Iterator[GraphNode]:
        stack = [self]
        while stack:            
            node = stack.pop()
            yield node
            stack.extend(node.get_children())


class RootGraphNode(GraphNode):

    id_manager: IDManager

    # ================ Initialization =================
    @typechecked
    def __init__(self, parent_node: GraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]], 
                 point_cloud: np.ndarray, child_nodes: list[GraphNode], first_seen: float | None, 
                 last_updated: float | None, curr_time: float | None, first_pose: np.ndarray | None, 
                 curr_pose: np.ndarray | None):
        self.id_manager = IDManager()
        id = self.id_manager.assign_id()
        super().__init__(id, parent_node, semantic_descriptors, point_cloud, child_nodes, 
                         first_seen, last_updated, curr_time, first_pose, curr_pose)

    # ==================== Getters ==================== 
    def get_parent(self) -> None:
        raise RuntimeError("get_parent() should not be called on the RootGraphNode!")
    
    def get_convex_hull(self) -> None:
        return None
    
    def get_point_cloud(self) -> np.ndarray:
        raise RuntimeError("get_point_cloud() should not be called on RootGraphNode!")
    
    @typechecked
    def request_new_ID(self) -> int:
        return self.id_manager.assign_id()
    
    def get_longest_line_size(self) -> None:
        return None
    
    def get_centroid(self) -> None:
        return None
    
    def get_semantic_descriptors() -> None:
        raise RuntimeError("get_semantic_descriptors() should not be called on RootGraphNode!")

    # ==================== Setters ====================
    def set_parent(self, node: GraphNode) -> None:
        raise RuntimeError("Calling set_parent() on RootGraphNode, which should never happen!")
            
    # ==================== Manipulators ====================
    def reset_saved_vars(self) -> set[GraphNode]:
        return set()