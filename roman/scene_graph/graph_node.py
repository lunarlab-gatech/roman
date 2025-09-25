from __future__ import annotations

from collections import defaultdict
from .hull_methods import get_convex_hull_from_point_cloud, find_point_overlap_with_hulls, longest_line_of_point_cloud
from .logger import logger
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
from ..params.scene_graph_3D_params import GraphNodeParams
from robotdatapy.camera import xyz_2_pixel
from robotdatapy.transform import transform
from roman.map.observation import Observation
from roman.object.segment import Segment
import trimesh
from typing import Iterator
from scipy.spatial import ConvexHull
from typeguard import typechecked
from .word_net_wrapper import WordNetWrapper, WordListWrapper

# Initialize a WordNetWrapper for use by GraphNodes
wordnetWrapper = WordNetWrapper(["curb", "tree", "garbage can", "door", "window", "pole", "street lamp", "trunk", "wall", "sign", "crosswalk", "sidewalk", "mulch", "leaves", "grass", "retaining wall", "railing", "curbstone", "bush", "hedge" "floor marking", "stairs", "column", "car", "wheel", "bike", "street", "manhole", "parking meter", "tree pit", "fire hydrant", "road marking", "zebra strips"])

@typechecked
class GraphNode():

    params: GraphNodeParams

    # ================ Initialization =================
    def __init__(self, id: int, parent_node: GraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]], 
                 point_cloud: np.ndarray, child_nodes: list[GraphNode], first_seen: float, last_updated: float, 
                 curr_time: float, first_pose: np.ndarray, curr_pose: np.ndarray, run_dbscan: bool = False, is_RootGraphNode: bool = False):
        """
        Don't create a Graph Node directly! Instead do it with create_node_if_possible(), 
        as node can be invalid after creation.
        """
        
        # Node ID
        self.id: int = id            

        # RootGraphNode handles id assignments, but all nodes have this        
        self._next_id: int = id + 1

        # The parent node to this node.
        self.parent_node: GraphNode | None = parent_node

        # Any child nodes we might have. TODO: Make it a set, so duplicate children can't occur.
        self.child_nodes: list[GraphNode] = child_nodes
        
        # As Clip embeddings can't be split and passed to children,
        # track embeddings specific to each node. Will be combined
        # with child embeddings before returning the final embedding.
        self.semantic_descriptors: list[tuple[np.ndarray, float]]  = semantic_descriptors

        # Keeps track of time this node (not any children) was seen. Note 
        # that last_updated just keeps track of when point cloud or semantic 
        # descriptors was changed, nothing else.
        self.first_seen: float = first_seen
        self.last_updated: float = last_updated

        # Keeps track of current time & pose
        self.curr_time: float = curr_time
        self.curr_pose: np.ndarray = curr_pose

        # Keeps track of the first pose associated with this node (not any descendents)
        self.first_pose: np.ndarray = first_pose

        # Information tracking if we are the RootGraphNode
        self.is_root = is_RootGraphNode

        # Points that relate only to this object and not any children.
        self.point_cloud: np.ndarray = np.zeros((0, 3), dtype=np.float64)

        # Tracks number of times we've been sighted
        self.num_sightings = 1
        if self.is_root:
            self.num_sightings = 10000 # Root should never be removed for too little sightings

        # Holds values for reuse to avoid recalculating them
        self._convex_hull: trimesh.Trimesh = None
        self._point_cloud: np.ndarray = None
        self._longest_line_size: float = None
        self._centroid: np.ndarray = None
        self._semantic_descriptor: np.ndarray = None
        self._words: WordListWrapper = None
        self._meronyms: dict[int, set[str]] = defaultdict(lambda: None)
        self._holonyms: dict[int, set[str]] = defaultdict(lambda: None)
        self._holonyms_pure: dict[int, set[str]] = defaultdict(lambda: None)
        self._descendents: list[GraphNode] = None

        # Tracks if SceneGraph3D needs to redo a calculation or not
        self._redo_convex_hull_geometric_overlap: bool = True
        self._redo_shortest_dist_between_convex_hulls: bool = True
        self._redo_word_comparisons: bool = True

        # Track if creating the node succeeded with create_node_if_possible()
        self._class_method_creation_success: bool = True

        # Update point cloud and check that the cloud is good
        to_delete = self.update_point_cloud(point_cloud, run_dbscan=run_dbscan)
        if len(to_delete) > 0:
            self._class_method_creation_success = False
        else:
            # Try to create our ConvexHull and make sure that also succeeds
            hull = self.get_convex_hull()
            if hull is None:
                self._class_method_creation_success = False
            else:
                # Make sure its larger than a certain minimum size
                if self.get_longest_line_size() < 0.4:
                    self._class_method_creation_success = False

        # If we are RootGraphNode, creation is always successful as we don't 
        # ever use our ConvexHull or Point Cloud
        if self.is_RootGraphNode():
            self._class_method_creation_success = True

    @classmethod
    def create_node_if_possible(cls, id: int, parent_node: GraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]], point_cloud: np.ndarray, child_nodes: list[GraphNode], first_seen: float, last_updated: float, curr_time: float, first_pose: np.ndarray, curr_pose: np.ndarray, run_dbscan: bool = True, is_RootGraphNode: bool = False) -> GraphNode | None:
        """ 
        This method will create and return a GraphNode if that node is valid, 
        or return None if its point cloud or hull isn't reasonable after cleanup. 
        """

        # Create node and run dbscan to filter out extra objects included in one faulty segmentation
        potential_node = cls(id, parent_node, semantic_descriptors, point_cloud, 
                                   child_nodes, first_seen, last_updated, curr_time, 
                                   first_pose, curr_pose, run_dbscan=run_dbscan, is_RootGraphNode=is_RootGraphNode)
        
        # Return the node if node creation was successful
        if potential_node._class_method_creation_success:
            return potential_node
        else:
            return None

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
            f"  Semantic Descriptors: {semantic_count} descriptors\n"
            f"    Weights: {semantic_weights}\n"
            f"  Point Cloud: {point_count} points\n"
            f"  First Seen: {self.first_seen:.2f}, Last Updated: {self.last_updated:.2f}\n"
            f"  Current Time: {self.curr_time:.2f}\n"
            f"  First Pose: {np.array2string(self.first_pose, precision=2, separator=', ')}\n"
            f"  Current Pose: {np.array2string(self.curr_pose, precision=2, separator=', ')}\n"
            f")")

    def get_id(self) -> int:
        return self.id

    def get_parent(self) -> GraphNode:
        if self.is_RootGraphNode():
            raise RuntimeError("get_parent() should not be called on the RootGraphNode!")
        return self.parent_node
    
    def is_parent_or_child(self, other: GraphNode) -> bool:
        """ Returns true if self is the parent or child of other. """
        return self.is_parent(other) or self.is_child(other)
    
    def is_parent(self, other: GraphNode) -> bool:
        """ Returns True if self is the parent of other. """
        if other in self.get_children():
            return True
        return False
    
    def is_child(self, other: GraphNode) -> bool:
        """ Returns True if self is the child of other. """
        return other.is_parent(self)
    
    def get_convex_hull(self)-> trimesh.Trimesh | None:
        if self.is_RootGraphNode():
            return None
        if self._convex_hull is None:
            self._convex_hull = get_convex_hull_from_point_cloud(self.get_point_cloud())
        return self._convex_hull
    
    def get_semantic_descriptor(self) -> np.ndarray | None:
        if self._semantic_descriptor is None:
            self._semantic_descriptor = self.calculate_semantic_descriptor(self.get_semantic_descriptors())
        return self._semantic_descriptor
    
    def get_words(self) -> WordListWrapper:
        if self._words is None:
            descriptor = self.get_semantic_descriptor()
            self._words = WordListWrapper.from_words(wordnetWrapper.map_embedding_to_words(descriptor, self.params.num_words_to_consider_ourselves))
        return self._words
    
    def get_all_meronyms(self, meronym_level: int = 1) -> set[str]:
        if self._meronyms[meronym_level] is None:
            self._meronyms[meronym_level] = self.get_words().words[0].get_all_meronyms(True, meronym_level)
        return self._meronyms[meronym_level]
    
    def get_all_holonyms(self, include_hypernyms: bool, holonym_level: int = 1) -> set[str]:
        if include_hypernyms:
            if self._holonyms[holonym_level] is None:
                self._holonyms[holonym_level] = self.get_words().words[0].get_all_holonyms(True, holonym_level)
            return self._holonyms[holonym_level]
        else:
            if self._holonyms_pure[holonym_level] is None:
                self._holonyms_pure[holonym_level] = self.get_words().words[0].get_all_holonyms(False, holonym_level)
            return self._holonyms_pure[holonym_level]
    
    def is_descendent_or_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if this node is a descendent or ascendent of other."""
        return other.is_ascendent(self) or self.is_ascendent(other)
    
    def is_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if self is an ascendent of other."""

        if self is other: return False
        descendents: list[GraphNode] = self.get_descendents()
        if other in descendents: return True
        else: return False
    
    def get_descendents(self) -> list[GraphNode]:
        if self._descendents is None:
            self._descendents = []
            self._descendents += self.get_children()
            for child in self.get_children():
                self._descendents += child.get_descendents()
        return self._descendents
    
    def is_RootGraphNode(self) -> bool:
        return self.is_root

    def get_time_first_seen(self) -> float:
        return self.first_seen
    
    def get_time_first_seen_recursive(self) -> float:
        """ Gets the earliest first seen time for self and descendents. """
        most_early_seen = self.get_time_first_seen()
        for child in self.get_children():
            child_first_seen = child.get_time_first_seen_recursive()
            if child_first_seen < most_early_seen:
                most_early_seen = child_first_seen
        return most_early_seen
    
    def get_time_last_updated(self) -> float:
        return self.last_updated
    
    def get_time_last_updated_recursive(self) -> float:
        """ Gets the most recent last updated time for self and descendents. """
        most_recent_update = self.get_time_last_updated()
        for child in self.get_children():
            child_last_updated = child.get_time_last_updated_recursive()
            if most_recent_update < child_last_updated:
                most_recent_update = child_last_updated
        return most_recent_update

    def get_first_pose(self) -> np.ndarray:
        return self.first_pose
    
    def get_first_pose_recursive(self) -> np.ndarray:
        """ Gets the earliest first pose for self and descendents. """
        most_early_seen = self.get_time_first_seen()
        most_early_pose = self.get_first_pose()
        for child in self.get_children():
            child_first_seen = child.get_time_first_seen_recursive()
            child_first_pose = child.get_first_pose_recursive()
            if child_first_seen < most_early_seen:
                most_early_seen = child_first_seen
                most_early_pose = child_first_pose
        return most_early_pose
    
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
    
    def request_new_ID(self) -> int:
        if self.is_RootGraphNode():
            new_id = self._next_id
            self._next_id += 1
            return new_id
        else:
            return self.parent_node.request_new_ID()
    
    def get_longest_line_size(self) -> float | None:
        if self.is_RootGraphNode():
            return None
        if self._longest_line_size is None:
            self._longest_line_size = longest_line_of_point_cloud(self.get_point_cloud())
        return self._longest_line_size
    
    def get_centroid(self) -> np.ndarray[float] | None:
        if self.is_RootGraphNode():
            return None
        if self._centroid is None:
            self._centroid = np.mean(self.get_point_cloud(), axis=0)
        return self._centroid
    
    def get_volume(self) -> float:
        if self.get_convex_hull() is None:
            raise RuntimeError(f"Trying to get volume of ConvexHull for Node {self.get_id()}, but there isn't a valid ConvexHull!")
        if not self.get_convex_hull().is_watertight:
            raise RuntimeError(f"Trying to get volume of ConvexHull for Node {self.get_id()}, but its not watertight!")
        return self.get_convex_hull().volume
    
    def get_point_cloud(self, recursive: bool = False) -> np.ndarray:
        if self.is_RootGraphNode():
            raise RuntimeError("get_point_cloud() should not be called on RootGraphNode!")

        if self._point_cloud is None:
            if recursive:
                full_pc = np.zeros((0, 3), dtype=np.float64)
                for child in self.get_children():
                    full_pc = np.concatenate((full_pc, child.get_point_cloud()), dtype=np.float64)
                self._point_cloud = np.concatenate((full_pc, self.point_cloud), dtype=np.float64)
                num_points = self._point_cloud.shape[0]
                if num_points < 4:
                    logger.debug(f"[bright_red]WARNING[/bright_red]: Current point cloud is {num_points} for Node {self.get_id()}")
                else:
                    logger.debug(f"[bright_yellow]UPDATE[/bright_yellow]: Current point cloud is {num_points} for Node {self.get_id()}")
            else:
                self._point_cloud = self.point_cloud.copy()
        return self._point_cloud
    
    def get_semantic_descriptors(self, recursive: bool = False) -> list[tuple[np.ndarray, float]]:
        if self.is_RootGraphNode():
            raise RuntimeError("get_semantic_descriptors() should not be called on RootGraphNode!")

        descriptors = []
        if recursive:
            for child in self.get_children():
                descriptors += child.get_semantic_descriptors()
        descriptors += self.semantic_descriptors
        return descriptors
    
    def get_total_weight_of_semantic_descriptors(self) -> float:
        descriptors: list[tuple[np.ndarray, float]] = self.get_semantic_descriptors()
        weights = np.array([w for _, w in descriptors])
        return weights.sum()
    
    def get_number_of_nodes(self) -> int:
        num = 1
        for child in self.get_children():
            num += child.get_number_of_nodes()
        return num
        
    def get_children(self) -> list[GraphNode]:
        return self.child_nodes
    
    def get_num_sightings(self) -> int:
        return self.num_sightings
    
    def is_sibling(self, other: GraphNode) -> bool:
        if self.is_RootGraphNode():
            return False
        return self.get_parent() == other.get_parent()
    
    def to_segment(self) -> Segment:
        """Returns a segment representation of this graph node"""

        # Create a Segment
        obs = Observation(self.get_time_first_seen_recursive(), 
                          self.get_first_pose_recursive(), None, 
                          None, None, None, None, None, None)
        seg = Segment(obs, None, self.get_id(), None)

        # Update internal values of Segment so it matches Graph Node
        seg.last_seen = self.get_time_last_updated_recursive()
        seg.num_sightings = None
        seg.points = self.get_point_cloud()
        seg.semantic_descriptor = self.get_semantic_descriptor()
        seg.semantic_descriptor_cnt = None

        return seg

    # ==================== Calculations ====================
    def calculate_semantic_descriptor(self, descriptors: list[tuple[NDArray[np.float64], float]], method: str = "average") -> np.ndarray | None:
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

        # Calculate the new final semantic descriptor
        if method == "weighted average":
            semantic_descriptor = np.average(embeddings, axis=0, weights=weights)
        elif method == "average":
            semantic_descriptor = np.mean(embeddings, axis=0)
        semantic_descriptor /= np.linalg.norm(semantic_descriptor)
        return semantic_descriptor

    # ==================== Setters ====================
    def set_parent(self, node: GraphNode | None) -> None:
        if self.is_RootGraphNode():
            raise RuntimeError("Calling set_parent() on RootGraphNode, which should never happen!")
        self.parent_node = node

    def set_id(self, id: int) -> None:
        self.id = id

    # ==================== Removal ====================
    def remove_from_graph(self, keep_children: bool = True) -> set[GraphNode]:
        """ Does so by disconnecting self from parent both ways. Returns any remaining parents that need to be deleted now. """
        if self.is_RootGraphNode():
            raise RuntimeError("Can't call remove_from_graph() on RootGraphNode!")

        # Add children to our parent (removing from self) if requested
        to_delete = set()
        if not keep_children:
            self.get_parent().add_children(self.get_children())
            for child in self.get_children():
                child.set_parent(self.get_parent())
            to_delete.update(self.remove_children(self.get_children()))

        # Disconnect ourselves from our parent both ways
        to_delete.update(self.get_parent().remove_child(self))
        self.set_parent(None)
        
        return to_delete
    
    def remove_from_graph_complete(self, keep_children: bool = True) -> list[int]:
        """ Does so by disconnecting self from parent both ways. Also immediately deletes any parent nodes that are now invalid. 
            Returns ids of additional nodes that were also retired (not including self). """

        deleted_ids = []
        to_delete = self.remove_from_graph(keep_children)
        while to_delete:
            for node in to_delete:
                deleted_ids.append(node.get_id())
            to_delete.update(to_delete.pop().remove_from_graph(keep_children))
        return deleted_ids

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
            logger.debug(f"NODE: {self.get_id()}, POINT CLOUD UPDATED from {num_points_before} to {num_points_after}")
            self.last_updated = self.curr_time
            logger.debug(f"remove_points_from_self(): Resetting Point Cloud for Node {self.get_id()}")
            return self.reset_saved_point_vars()

        # Otherwise, we don't want to be deleted
        return set()

    def remove_points(self, pc_to_remove: np.ndarray) -> set[GraphNode]:
        """ Returns all nodes that are no longer valid after their points are removed. """
        nodes_to_delete = set()

        # Start with children, as parents validity depends on if children's clouds are altered
        for child in self.child_nodes:
            nodes_to_delete.update(child.remove_points(pc_to_remove))

        # Now, delete points from self
        nodes_to_delete.update(self.remove_points_from_self(pc_to_remove))
        return nodes_to_delete
    
    def remove_points_complete(self, pc_to_remove: np.ndarray) -> None:
        """ Remove points and then delete any nodes that are no longer valid. """
        to_delete = self.remove_points(pc_to_remove)
        while to_delete:
            to_delete.update(to_delete.pop().remove_from_graph())
    
    def remove_child(self, child: GraphNode) -> set[GraphNode]:
        if child in self.child_nodes:
            self.child_nodes.remove(child)
            logger.debug(f"remove_child(): Resetting Point Cloud for Node {self.get_id()}")
            self.reset_saved_descriptor_vars()
            self.reset_saved_inheritance_vars()
            return self.reset_saved_point_vars()
        else:
            raise ValueError(f"Tried to remove {child} from {self}, but {child} not in self.child_nodes: {self.child_nodes}")
        
    def remove_children(self, children: list[GraphNode]) -> set[GraphNode]:
        nodes_to_delete = set()
        for child in children:
            nodes_to_delete.update(self.remove_child(child))
        return nodes_to_delete

    # ==================== Updating / Adding ====================
    def update_curr_time(self, curr_time: float):
        self.curr_time = curr_time
        for child in self.get_children():
            child.update_curr_time(curr_time)

    def update_curr_pose(self, curr_pose: np.ndarray):
        self.curr_pose = curr_pose
        for child in self.get_children():
            child.update_curr_pose(curr_pose)

    def add_child(self, new_child: GraphNode) -> None:
        if new_child in self.child_nodes:
            return # Shouldn't add children more than once
        self.child_nodes.append(new_child)
        logger.debug(f"add_child(): Resetting Point Cloud for Node {self.get_id()}")
        self.reset_saved_descriptor_vars()
        self.reset_saved_inheritance_vars()
        self.reset_saved_point_vars_safe()

    def add_children(self, new_children: list[GraphNode]) -> None:
        for new_child in new_children:
            self.add_child(new_child)

    def add_semantic_descriptors(self, descriptors: list[tuple[np.ndarray, float]]) -> None:
        self.semantic_descriptors += descriptors
        self.last_updated = self.curr_time
        self.reset_saved_descriptor_vars()

    @staticmethod
    def _intersect_rows(A, B):
        """ Helper method to calculate points shared between two sets of points. """
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        A_view = A.view([('', A.dtype)] * A.shape[1])
        B_view = B.view([('', B.dtype)] * B.shape[1])
        return np.intersect1d(A_view, B_view).view(A.dtype).reshape(-1, A.shape[1])

    def update_point_cloud(self, new_points: np.ndarray, run_dbscan: bool = False) -> set[GraphNode]:
        """ Returns nodes that might need to be deleted due to cleanup removing points..."""

        # =========== Add to Point Cloud ============
        
        # Skip math if no new points are included
        if new_points.shape[0] == 0: return set()

        # Check the input array is the shape we expect
        if new_points.shape[1] != 3: raise ValueError(f"Point array in a non-supported shape: {new_points.shape()}")

        # Append them to our point cloud
        self.point_cloud = np.concatenate((self.point_cloud, new_points), axis=0)
        self.point_cloud = np.unique(self.point_cloud, axis=0) # Prune any duplicates
        logger.debug(f"Point Cloud size after appending: {self.point_cloud.shape[0]}")
        self.last_updated = self.curr_time
        self.reset_saved_point_vars_safe() # Wipe saved point cloud for next steps

        # =========== Clean-up Point Cloud  ============
        # Necessary to limit sizes of point clouds for computation purposes and for ensuring incoming point clouds only represent a single object.
        # Considers child cloud as part of self for determining how to downsample, remove outliers, and cluster. 

        # Perform DBSCAN clustering (if desired)
        if run_dbscan:
            # Now perform clustering
            self._dbscan_clustering()

        # Run a downsampling operation to keep the point clouds small enough for real-time
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.get_point_cloud())
        length = self.get_longest_line_size()
        if length > 0:
            voxel_size = self.params.voxel_size_not_variable
            if self.params.enable_variable_voxel_size:
                voxel_size = length * self.params.voxel_size_variable_ratio_to_length

            pcd_sampled = pcd.voxel_down_sample(voxel_size)
        else:
            return self.reset_saved_point_vars()
        
        # Update the point cloud
        self.point_cloud = GraphNode._intersect_rows(self.point_cloud, np.asarray(pcd_sampled.points))
        logger.debug(f"Point Cloud size after downsampling: {self.point_cloud.shape[0]}")
        self.last_updated = self.curr_time
        self.reset_saved_point_vars_safe()

        # Remove statistical outliers
        self._remove_statistical_outliers()

        # Reset point cloud dependent saved variables and return nodes to delete
        logger.debug(f"update_point_cloud(): Resetting Point Cloud for Node {self.get_id()}")
        return self.reset_saved_point_vars()

    def _dbscan_clustering(self) -> None:

        # Convert into o3d PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.get_point_cloud())

        # Perform clustering
        length = self.get_longest_line_size()

        epsilon = self.params.epsilon_not_variable
        if self.params.enable_variable_epsilon:
            epsilon = length * self.params.epsilon_variable_ratio_to_length

        labels = np.array(pcd.cluster_dbscan(eps=epsilon, min_points=self.params.min_points))
        max_cluster_index = labels.max()
        if max_cluster_index == -1:
            logger.info(f"[bright_red]WARNING[/bright_red]: All points in this node {self.get_id()} have been detected as noise!")

            # Right now, lets just assume that our epsilon is off and thus every point should be kept
            # Thus, let it continue the algorithm...

        # Check size of max cluster
        max_cluster_size = np.sum(labels == max_cluster_index)
        cluster_size_ratio = max_cluster_size / len(pcd.points)
        logger.debug(f"Cluster Size Ratio: {cluster_size_ratio}")
        if cluster_size_ratio < self.params.cluster_percentage_of_full:

            # Since this cluster is too small, the semantic embedding will not be 
            # representative. Thus, we must delete this node, so wipe our point cloud.
            self.point_cloud = np.zeros((0, 3), dtype=np.float64)
            logger.debug(f"Cluster size too small, rejecting...")
            self.reset_saved_point_vars_safe()
            return

        # Filter out any points not belonging to max cluster
        filtered_indices = np.asarray(labels == max_cluster_index).nonzero()
        clustered_points = np.asarray(pcd.points)[filtered_indices]

        # Save the new sampled point cloud 
        self.point_cloud = GraphNode._intersect_rows(self.point_cloud, clustered_points)
        logger.debug(f"Point Cloud size after DBScan: {self.point_cloud.shape[0]}")
        self.last_updated = self.curr_time
        self.reset_saved_point_vars_safe()

        # NOTE: This actually ISN'T a safe operation, so calling method MUST call reset_saved_point_vars().

    def _remove_statistical_outliers(self):

        # Convert into o3d PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.get_point_cloud())

        # Remove statistical outliers
        pcd_sampled, _ = pcd.remove_statistical_outlier(self.params.stat_out_num_neighbors, self.params.std_ratio)

        # Save the new sampled point cloud 
        self.point_cloud = GraphNode._intersect_rows(self.point_cloud, np.asarray(pcd_sampled.points))
        logger.debug(f"Point Cloud size after remove statistical outliers: {self.point_cloud.shape[0]}")
        self.last_updated = self.curr_time
        self.reset_saved_point_vars_safe()

        # NOTE: This actually ISN'T a safe operation, so calling method MUST call reset_saved_point_vars().

    # ==================== Merging ====================
    def merge_with_node(self, other: GraphNode) -> GraphNode | None:
        """
        As opposed to merge_with_observation (which can just be called), this method 
        will take out self and other from the graph and return a new node. This new
        node needs to be inserted back into the graph by the SceneGraph3D.

        NOTE: other cannot be a descendent or ascendent of self!
        NOTE: If the new node is invalid, then will just return None.
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
        smallest_id = self.get_id() if self.get_id() < other.get_id() else other.get_id()
        new_node = GraphNode.create_node_if_possible(smallest_id, None, combined_descriptors, 
                        combined_pc, combined_children, first_seen, self.curr_time, self.curr_time, 
                        first_pose, self.curr_pose, run_dbscan=False)
        if new_node is None:
            return None
        
        # Update the number of sightings
        total_sightings = self.get_num_sightings() + other.get_num_sightings()
        new_node.num_sightings = total_sightings
            
        # Tell our children who their new parent is
        for child in combined_children:
            child.set_parent(new_node)
        return new_node

    def merge_with_observation(self, new_pc: np.ndarray, new_descriptors: list[tuple[np.ndarray, float]] | None) -> None:
        """
        Args:
            new_pc (np.ndarray): Point Cloud in shape of (N, 3) in the global frame.

        Returns:
            Set of Graph nodes that might need to be deleted.
        """

        # Save the semantic embedding into this parent graph node
        if new_descriptors is not None: 
            self.add_semantic_descriptors(new_descriptors)

        # Get convex hulls of each child
        hulls: list[trimesh.Trimesh] = []
        for child in self.get_children():
            hulls.append(child.get_convex_hull())
        
        # Get masks of which points fall into which hulls
        logger.info(f"Current Node: {self.get_id()}")
        contain_masks = find_point_overlap_with_hulls(new_pc, hulls, fail_on_multi_assign=False)
        # TODO: Add statement to check if this multi-assignment happens too often!

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
            to_delete = self.update_point_cloud(orphan_pc)
            if len(to_delete) > 0:
                raise RuntimeError(f"Cannot merge_with_observation; Node {self.get_id()}'s point cloud invalid after adding additional points, which should never happen!")
            
        # Increase our sightings
        self.num_sightings += 1

    def merge_child_with_self(self, other: GraphNode) -> None:
        # Make sure other is a child of self
        if not self.is_parent(other) or not other in self.get_children():
            raise ValueError("Cannot merge_child_with_self; node {other} is not a child of self!")
        
        # Add semantic descriptors specific to child (not from grandchidren) to this node
        self.add_semantic_descriptors(other.semantic_descriptors)
        
        # Do the same with point cloud specific to the child (not grandchildren)
        to_delete = self.update_point_cloud(other.point_cloud)
        if len(to_delete) > 0:
            raise RuntimeError(f"Cannot merge_child_with_self; New point cloud is invalid, this should never happen")

        # Add grandchildren as children and add self as grandchildrens' parent
        for grandchild in other.get_children():
            self.add_child(grandchild)
            grandchild.set_parent(self)

        # Remove child
        other.remove_from_graph_complete()
        logger.debug(f"[bright_blue]Parent-Child Merge[/bright_blue]: Merged Node {other.get_id()} into Parent Node {self.get_id()}")

    # ==================== Resetting Vars ====================
    def reset_saved_point_vars(self) -> set[GraphNode]:
        """ 
        Wipes saved point variables since they need to be recalculated. 
        Returns list of nodes that are no longer valid and should be removed. 
        """

        # Do nothing if we are the RootGraphNode
        if self.is_RootGraphNode():
            return set()
        
        # Wipe all variables
        self._convex_hull = None
        self._point_cloud = None
        self._longest_line_size = None
        self._centroid = None

        # Make sure SceneGraph3D knows to redo some calculations
        self._redo_convex_hull_geometric_overlap = True
        self._redo_shortest_dist_between_convex_hulls = True

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
            logger.debug(f"reset_saved_point_vars(): Resetting Point Cloud for Parent node {self.parent_node.get_id()}")
            to_delete.update(self.parent_node.reset_saved_point_vars())

        # Return nodes that need to be deleted
        return to_delete
    
    def reset_saved_point_vars_safe(self) -> None:
        """ 
        Similar to reset_saved_point_vars(), but called if points were only
        possibly added to a node. Thus, no need to check node validity
        or return nodes that might need to be deleted.
        """
        
        # Wipe all variables
        self._convex_hull = None
        self._point_cloud = None
        self._longest_line_size = None
        self._centroid = None

        # Make sure SceneGraph3D knows to redo some calculations
        self._redo_convex_hull_geometric_overlap = True
        self._redo_shortest_dist_between_convex_hulls = True

        # Reset variables in parents 
        if self.parent_node is not None:
            self.parent_node.reset_saved_point_vars_safe()
    
    def reset_saved_descriptor_vars(self) -> None:
        """ Wipes saved descriptor variables as they need to be recalculated """

        # Track the previous word to see if it changed
        prev_word = None
        if self._words is not None:
            prev_word = self._words.words[0]

        # Wipe variables
        self._semantic_descriptor = None
        self._words = None

        # Get the new word and see if it changed
        if prev_word is None or not self.get_words().words[0] == prev_word:

            # Reset all our meronyms & holonyms since it changed            
            self._meronyms = defaultdict(lambda: None)
            self._holonyms = defaultdict(lambda: None)
            self._holonyms_pure = defaultdict(lambda: None)

            # Also let the scene graph know that some word comparisons will need rechecking
            self._redo_word_comparisons = True

        # Do the same in parents
        if self.parent_node is not None:
            self.parent_node.reset_saved_descriptor_vars()

    def reset_saved_inheritance_vars(self) -> None:

        # Wipe variables
        self._descendents = None

        # Also wipe this in parents
        if self.parent_node is not None:
            self.parent_node.reset_saved_inheritance_vars()

    # ==================== Iterator ====================
    def __iter__(self) -> Iterator[GraphNode]:
        stack = [self]
        while stack:            
            node = stack.pop()
            yield node
            stack.extend(node.get_children())
