from __future__ import annotations

from collections import defaultdict
import cv2 as cv
from enum import Enum
import heapq
from .scene_graph_utils import get_convex_hull_from_point_cloud, find_point_overlap_with_hulls, longest_line_of_point_cloud, expand_hull_outward_by_fixed_offset
from ..logger import logger
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
from open3d.geometry import OrientedBoundingBox
from ..params.scene_graph_3D_params import GraphNodeParams
from ..params.system_params import SystemParams
from robotdatapy.camera import xyz_2_pixel
from robotdatapy.data.img_data import CameraParams
from robotdatapy.transform import transform
from roman.map.observation import Observation
from roman.map.voxel_grid import VoxelGrid
from roman.object.segment import Segment
import trimesh
from typing import Iterator
from scipy.spatial import ConvexHull, cKDTree
from typeguard import typechecked
from .word_net_wrapper import WordNetWrapper, WordWrapper

# Initialize a WordNetWrapper for use by GraphNodes
@typechecked
class GraphNode():

    class SegmentStatus(Enum):
        NURSERY = 0
        SEGMENT = 1
        INACTIVE = 2
        GRAVEYARD = 3

    params: GraphNodeParams
    camera_params: CameraParams
    wordnetWrapper = WordNetWrapper()

    # ================ Initialization =================
    def __init__(self, id: int, parent_node: GraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]], 
                 semantic_descriptor_inc: np.ndarray | None, semantic_descriptor_inc_count: int, 
                 point_cloud: np.ndarray, child_nodes: list[GraphNode], first_seen: float, last_updated: float, 
                 curr_time: float, first_pose: np.ndarray, last_pose: np.ndarray, curr_pose: np.ndarray, run_dbscan: bool | None = None, is_RootGraphNode: bool = False):
        """
        Don't create a Graph Node directly! Instead do it with create_node_if_possible(), 
        as node can be invalid after creation.
        """
        
        # Node ID
        self.id: int = id           

        # Segment status for ROMAN operations
        self.status: GraphNode.SegmentStatus = GraphNode.SegmentStatus.NURSERY

        # RootGraphNode handles id assignments, but all nodes have this        
        self._next_id: int = id + 1
        self._forfeited_ids = []  # min-heap of released IDs

        # The parent node to this node.
        self.parent_node: GraphNode | None = parent_node

        # Any child nodes we might have. TODO: Make it a set, so duplicate children can't occur.
        self.child_nodes: list[GraphNode] = child_nodes
        
        # As Clip embeddings can't be split and passed to children,
        # track embeddings specific to each node. Will be combined
        # with child embeddings before returning the final embedding.
        self.semantic_descriptors: list[tuple[np.ndarray, float]]  = semantic_descriptors

        # Also calculate a running incremental semantic descriptor
        self.semantic_descriptor_inc: np.ndarray = semantic_descriptor_inc
        self.semantic_descriptor_inc_count: int = semantic_descriptor_inc_count
        self.obs_descriptor = None

        # Keeps track of first time (and pose) this node (not any children) was seen. Note 
        # that last_updated just keeps track of when point cloud or semantic 
        # descriptors was changed, nothing else.
        self.first_seen: float = first_seen
        self.first_pose: np.ndarray = first_pose

        # Keeps track of last time (and pose) this was updated with
        self.last_updated: float = last_updated
        self.last_pose: float = last_pose

        # Keeps track of current time & pose
        self.curr_time: float = curr_time
        self.curr_pose: np.ndarray = curr_pose

        # Information tracking if we are the RootGraphNode
        self.is_root = is_RootGraphNode

        # Points that relate only to this object and not any children (expressed in world frame).
        self.point_cloud: np.ndarray = np.zeros((0, 3), dtype=np.float64)

        # Points (could include children points) expressed in the robot frame aligned to gravity
        self.point_cloud_robot_aligned: np.ndarray | None = None

        # Tracks number of times we've been sighted
        self.num_sightings = 1
        if self.is_root:
            self.num_sightings = 10000 # Root should never be removed for too little sightings

        # Holds values for reuse to avoid recalculating them
        self._convex_hull: trimesh.Trimesh = None
        self._voxel_grid: VoxelGrid = None
        self._point_cloud: np.ndarray = None
        self._longest_line_size: float = None
        self._centroid: np.ndarray = None
        self._oriented_bbox: OrientedBoundingBox = None
        self._semantic_descriptor: np.ndarray = None
        self._word: WordWrapper = None
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
        if run_dbscan is None:
            run_dbscan = self.params.dbscan_and_remove_outliers_on_node_creation

        to_delete = self.update_point_cloud(point_cloud, run_dbscan=run_dbscan, 
                                            remove_outliers=run_dbscan, downsample=self.params.downsample_on_node_creation)
        if len(to_delete) > 0:
            self._class_method_creation_success = False

        # If it hasn't failed yet, check if our ConvexHull is valid (if requested)
        elif self.params.require_valid_convex_hull:
            hull = self.get_convex_hull()
            if hull is None:
                self._class_method_creation_success = False
        
        # Additionally, check if our size is reasonable
        if self.params.check_minimum_node_size_on_node_creation and not self.is_RootGraphNode() and self.get_longest_line_size() < 0.4:
            self._class_method_creation_success = False

        # If we are RootGraphNode, creation is always successful as we don't 
        # ever use our ConvexHull or Point Cloud
        if self.is_RootGraphNode():
            self._class_method_creation_success = True

    @classmethod
    def create_node_if_possible(cls, id: int, parent_node: GraphNode | None, semantic_descriptors: list[tuple[np.ndarray, float]], 
                                semantic_descriptor_inc: np.ndarray | None, semantic_descriptor_inc_count: int, point_cloud: np.ndarray, 
                                child_nodes: list[GraphNode], first_seen: float, last_updated: float, curr_time: float, 
                                first_pose: np.ndarray, last_pose: np.ndarray, curr_pose: np.ndarray, run_dbscan: bool | None = None, is_RootGraphNode: bool = False) -> GraphNode | None:
        """ 
        This method will create and return a GraphNode if that node is valid, 
        or return None if its point cloud or hull isn't reasonable after cleanup. 
        """

        # Create node and run dbscan to filter out extra objects included in one faulty segmentation
        potential_node = cls(id, parent_node, semantic_descriptors, semantic_descriptor_inc, semantic_descriptor_inc_count,
                             point_cloud, child_nodes, first_seen, last_updated, curr_time, first_pose, last_pose, curr_pose, 
                             run_dbscan=run_dbscan, is_RootGraphNode=is_RootGraphNode)
        
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
            f"  Id: {self.get_id()}\n"
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
    
    def get_status(self) -> GraphNode.SegmentStatus:
        return self.status

    def get_parent(self) -> GraphNode:
        if self.is_RootGraphNode():
            raise RuntimeError("get_parent() should not be called on the RootGraphNode!")
        return self.parent_node

    def get_all_descendents(self) -> list[GraphNode]:
        """ Returns a list of all descendents of this node. """
        descendents = []
        descendents += self.get_children()
        for child in self.get_children():
            descendents += child.get_all_descendents()
        return descendents
    
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
    
    def is_segment_or_inactive(self) -> bool:
        return self.get_status() == GraphNode.SegmentStatus.SEGMENT or \
               self.get_status() == GraphNode.SegmentStatus.INACTIVE
    
    def get_convex_hull(self) -> trimesh.Trimesh | None:
        if self.is_RootGraphNode():
            return None
        if self._convex_hull is None:
            self._convex_hull = get_convex_hull_from_point_cloud(self.get_point_cloud())
            if self._convex_hull is not None and self.params.convex_hull_outward_offset != 0:
                self._convex_hull = expand_hull_outward_by_fixed_offset(self._convex_hull, self.params.convex_hull_outward_offset)
        return self._convex_hull
    
    def get_voxel_grid(self, voxel_size: float) -> VoxelGrid | None:
        if self.is_RootGraphNode():
            return None
        if self._voxel_grid is None:
            self._voxel_grid = VoxelGrid.from_points(self.get_point_cloud(), voxel_size)
        return self._voxel_grid
    
    def get_num_points(self):
        if self.get_point_cloud() is None:
            return 0
        else:
            return self.get_point_cloud().shape[0]
    
    def get_semantic_descriptor(self) -> np.ndarray | None:
        if self._semantic_descriptor is None:
            if self.params.calculate_descriptor_incrementally:
                self._semantic_descriptor = self.semantic_descriptor_inc
            else:
                self._semantic_descriptor = self.calculate_semantic_descriptor(self.get_semantic_descriptors())
        return self._semantic_descriptor
    
    def get_word(self) -> WordWrapper | None:
        if self._word is None:
            descriptor = self.get_semantic_descriptor()
            if descriptor is None:
                self._word = None
            else:
                self._word = WordWrapper.from_word(self.wordnetWrapper.map_embedding_to_words(descriptor, 1)[0])
        return self._word
    
    def get_all_meronyms(self, meronym_level: int = 1) -> set[str]:
        if self._meronyms[meronym_level] is None:
            self._meronyms[meronym_level] = self.get_word().get_all_meronyms(True, meronym_level)
        return self._meronyms[meronym_level]
    
    def get_all_holonyms(self, include_hypernyms: bool, holonym_level: int = 1) -> set[str]:
        if self._holonyms[holonym_level] is None:
            self._holonyms[holonym_level] = self.get_word().get_all_holonyms(include_hypernyms, holonym_level)
        return self._holonyms[holonym_level]
    
    def check_if_meronoym_holonym_relationships(self, other: GraphNode) -> tuple[bool, bool]:
        """ Checks if self and other have meronym-holonym relationships. 
            Returns (relationship_exists, self_is_meronym) """
        
        # Get likeliest wrapped for each node
        word_s: WordWrapper = self.get_word()
        word_o: WordWrapper = other.get_word()

        logger.debug(f"Words: {word_s.word} {word_o.word}")

        # Get all holonyms/meronyms for each node
        word_s_meronyms: set[str] = self.get_all_meronyms(2)
        word_o_meronyms: set[str] = other.get_all_meronyms(2)
        word_s_holonyms: set[str] = self.get_all_holonyms(True, 2)
        word_o_holonyms: set[str] = other.get_all_holonyms(True, 2)

        logger.debug(f"All Holonyms: {word_s_holonyms} {word_o_holonyms}")

        # Check if there is a Holonym-Meronym relationship
        if word_s.word in word_o_meronyms or word_s.word in word_o_holonyms or \
            word_o.word in word_s_meronyms or word_o.word in word_s_holonyms:

            # Determine which is the meronym
            self_is_meronym: bool = False
            if word_s.word in word_o_meronyms or word_o.word in word_s_holonyms:
                self_is_meronym = True

            return (True, self_is_meronym)
    
        return (False, False)
    
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
        """ Gets the earliest first seen time """

        # If not including children, just return our own first seen
        if not self.params.parent_node_inherits_data_from_children:
            return self.first_seen

        # Otherwise, find first seen among ourselves and our children
        most_early_seen = self.first_seen
        for child in self.get_children():
            child_first_seen = child.get_time_first_seen()
            if child_first_seen < most_early_seen:
                most_early_seen = child_first_seen
        return most_early_seen
    
    def get_time_last_updated(self) -> float:
        """ Gets the most recent last updated time """

        # If not including children, just return our own last updated
        if not self.params.parent_node_inherits_data_from_children:
            return self.last_updated
        
        # Otherwise, find most recent last updated among ourselves and our children
        most_recent_update = self.last_updated
        for child in self.get_children():
            child_last_updated = child.get_time_last_updated()
            if most_recent_update < child_last_updated:
                most_recent_update = child_last_updated
        return most_recent_update

    def get_first_pose(self) -> np.ndarray:
        """ Gets the earliest first pose for self and descendents. """

        # If not including children, just return our own first pose
        if not self.params.parent_node_inherits_data_from_children:
            return self.first_pose
        
        # Otherwise, find first pose among ourselves and our children
        most_early_seen = self.get_time_first_seen()
        most_early_pose = self.first_pose
        for child in self.get_children():
            child_first_seen = child.get_time_first_seen()
            child_first_pose = child.get_first_pose()
            if child_first_seen < most_early_seen:
                most_early_seen = child_first_seen
                most_early_pose = child_first_pose
        return most_early_pose
    
    def get_last_pose(self) -> np.ndarray:
        # If not including children, just return our own last pose
        if not self.params.parent_node_inherits_data_from_children:
            return self.last_pose
    
        # Otherwise, find last pose among ourselves and our children
        most_recent_update = self.last_updated
        most_recent_pose = self.last_pose
        for child in self.get_children():
            child_last_updated = child.get_time_last_updated()
            child_last_pose = child.get_last_pose()
            if most_recent_update < child_last_updated:
                most_recent_update = child_last_updated
                most_recent_pose = child_last_pose
        return most_recent_pose
        
    def request_new_ID(self) -> int:
        if self.is_RootGraphNode():
            if self._forfeited_ids:  
                # Reuse the smallest forfeited ID
                return heapq.heappop(self._forfeited_ids)
            else:
                # Issue a fresh ID
                new_id = self._next_id
                self._next_id += 1
                return new_id
        else:
            return self.parent_node.request_new_ID()

    def forfeit_ID(self, id_to_forfeit: int):
        """Return an ID to the pool of available IDs."""
        if self.is_RootGraphNode():
            if id_to_forfeit < self._next_id:
                heapq.heappush(self._forfeited_ids, id_to_forfeit)
        else:
            self.parent_node.forfeit_ID(id_to_forfeit)
    
    def get_longest_line_size(self) -> float | None:
        if self.is_RootGraphNode():
            return None
        if self._longest_line_size is None:
            self._longest_line_size = longest_line_of_point_cloud(self.get_convex_hull().vertices)
        return self._longest_line_size
    
    def get_centroid(self) -> np.ndarray[float] | None:
        if self.is_RootGraphNode():
            return None
        if self._centroid is None:
            self._centroid = np.mean(self.get_point_cloud(), axis=0)
        return self._centroid
    
    def get_centroid_robot_aligned(self) -> np.ndarray | None:
        # Extract point cloud in robot frame aligned to gravity
        points: np.ndarray | None = self.get_point_cloud_robot_aligned()
        if points is None: return None
        
        # Compute the centroid as the mean
        return np.mean(points, axis=0)
    
    def get_oriented_bbox(self) -> OrientedBoundingBox | None:
        if self._oriented_bbox is None:
            if self.get_num_points() > 4:
                vector = o3d.utility.Vector3dVector(self.get_point_cloud())
                self._oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(vector)
        return self._oriented_bbox

    def get_volume(self) -> float:
        # Use an oriented bounding box (like ROMAN)
        if not self.params.use_convex_hull_for_volume:
            obb: OrientedBoundingBox | None = self.get_oriented_bbox()
            if obb is not None: return obb.volume()
            else: return 0.0
        
        # Use a Convex Hull instead
        else:
            if self.get_convex_hull() is None:
                raise RuntimeError(f"Trying to get volume of ConvexHull for Node {self.get_id()}, but there isn't a valid ConvexHull!")
            if not self.get_convex_hull().is_watertight:
                raise RuntimeError(f"Trying to get volume of ConvexHull for Node {self.get_id()}, but its not watertight!")
            return self.get_convex_hull().volume
    
    def get_extent(self) -> np.ndarray:
        obb: OrientedBoundingBox | None = self.get_oriented_bbox()
        if obb is not None: return obb.extent
        else: return np.zeros(3)
    
    def get_point_cloud(self) -> np.ndarray:
        if self.is_RootGraphNode():
            raise RuntimeError("get_point_cloud() should not be called on RootGraphNode!")
        
        if self._point_cloud is None:
            if self.params.parent_node_inherits_data_from_children:
                full_pc = np.zeros((0, 3), dtype=np.float64)
                for child in self.get_children():
                    full_pc = np.concatenate((full_pc, child.get_point_cloud()), dtype=np.float64)
                self._point_cloud = np.concatenate((full_pc, self.point_cloud), dtype=np.float64)
            else:
                self._point_cloud = self.point_cloud.copy()
        return self._point_cloud
    
    def get_point_cloud_robot_aligned(self) -> np.ndarray | None:
        return self.point_cloud_robot_aligned

    def get_semantic_descriptors(self) -> list[tuple[np.ndarray, float]]:
        if self.is_RootGraphNode():
            raise RuntimeError("get_semantic_descriptors() should not be called on RootGraphNode!")

        descriptors = []
        if self.params.parent_node_inherits_descriptors_from_children:
            for child in self.get_children():
                descriptors += child.get_semantic_descriptors()

        if self.params.ignore_descriptors_from_observation:
            descriptors += self.semantic_descriptors[1:] # Skip first descriptor, which is from observation
        else:
            descriptors += self.semantic_descriptors
        return descriptors
    
    def get_total_weight_of_semantic_descriptors(self) -> float:
        descriptors: list[tuple[np.ndarray, float]] = self.get_semantic_descriptors()
        weights = np.array([w for _, w in descriptors])
        return float(weights.sum())
    
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
        obs = Observation(self.get_time_first_seen(), 
                          self.get_first_pose(), None, 
                          None, None, None, None, None, None)
        seg = Segment(obs, None, self.get_id(), None)

        # Update internal values of Segment so it matches Graph Node
        seg.last_seen = self.get_time_last_updated()
        seg.num_sightings = self.get_num_sightings()
        seg.points = self.get_point_cloud()
        seg.semantic_descriptor = self.get_semantic_descriptor()
        seg.semantic_descriptor_cnt = None

        return seg
    
    @staticmethod
    def from_segment(seg: Segment) -> GraphNode | None:
        """ Creates a GraphNode from a Segment if possible, else returns None.
         
        NOTE: Doesn't support all functionality of GraphNode!
        """
        new_node: GraphNode | None = GraphNode.create_node_if_possible(seg.id, None, [], None, 0,
                        seg.points, [], 0.0, 0.0, 0.0, np.eye(4), np.eye(4), np.eye(4), run_dbscan=False)
        if new_node is None: return None # Node creation failed

        # Add the descriptor to the node
        if seg.semantic_descriptor is not None:
            new_node.add_semantic_descriptors([(seg.semantic_descriptor, new_node.get_volume())])
            new_node.add_semantic_descriptors_incremental(seg.semantic_descriptor, 1)
        return new_node

    # ==================== Calculations ====================
    def calculate_semantic_descriptor(self, descriptors: list[tuple[NDArray[np.float64], float]]) -> np.ndarray | None:
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
        if self.params.use_weighted_average_for_descriptor:
            semantic_descriptor = np.average(embeddings, axis=0, weights=weights)
        else:
            semantic_descriptor = np.mean(embeddings, axis=0)
        semantic_descriptor /= np.linalg.norm(semantic_descriptor)
        return semantic_descriptor

    def calculate_point_cloud_in_robot_frame_aligned(self, H_world_wrt_robot_aligned: np.ndarray):
        """ 
        Calculates this node point cloud in the robot frame aligned with gravity
        
        Args:
            H_world_wrt_robot_aligned: Pose of the world frame with respect to the robot frame aligned to gravity.
        """

        self.point_cloud_robot_aligned = transform(H_world_wrt_robot_aligned, self.get_point_cloud(), axis=0)


    # ==================== Methods for Mimicing ROMAN's functionality ====================
    def reconstruct_mask(self, pose: np.ndarray, downsample_factor=1) -> np.ndarray[np.uint8]:
        """ Create a mask representing this node in the camera frame """

        # Map reprojected_bbox onto the mask
        mask = np.zeros((self.camera_params.height, self.camera_params.width), dtype=np.uint8)
        bbox = self.reprojected_bbox(pose)
        if bbox is not None:
            upper_left, lower_right = bbox
            mask[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]] = 1

        # Run Downsampling
        if downsample_factor == 1:
            return mask.astype('uint8')
        return np.array(cv.resize(mask, (mask.shape[1]//downsample_factor, mask.shape[0]//downsample_factor), interpolation=cv.INTER_NEAREST)).astype('uint8')

    def reprojected_bbox(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        pixels = self._pixels_2d(pose)
        if pixels is None:
            return None
        upper_left = np.max([np.min(pixels, axis=0).astype(int), [0, 0]], axis=0)
        lower_right = np.min([np.max(pixels, axis=0).astype(int), 
                                [self.camera_params.width, self.camera_params.height]], axis=0)
        if lower_right[0] - upper_left[0] <= 0 or lower_right[1] - upper_left[1] <= 0:
            return None
        return upper_left, lower_right
    
    def _pixels_2d(self, pose: np.ndarray) -> np.ndarray | None:
        if self.get_point_cloud() is None:
            return None
        points_c = transform(np.linalg.inv(pose), self.get_point_cloud(), axis=0)
        points_c = points_c[points_c[:,2] >= 0]
        if len(points_c) == 0:
            return None
        pixels = xyz_2_pixel(points_c, self.camera_params.K)
        pixels = pixels[np.bitwise_and(pixels[:,0] >= 0, pixels[:,0] < self.camera_params.width), :]
        pixels = pixels[np.bitwise_and(pixels[:,1] >= 0, pixels[:,1] < self.camera_params.height), :]
        if len(pixels) == 0:
            return None
        return pixels

    # ==================== Setters ====================
    def set_parent(self, node: GraphNode | None) -> None:
        if self.is_RootGraphNode():
            raise RuntimeError("Calling set_parent() on RootGraphNode, which should never happen!")
        self.parent_node = node

    def set_id(self, id: int) -> None:
        """ With a new id, we need to tell the rest of our algorithms to recalculate everything for us. """

        self.reset_saved_point_vars_safe()
        self.reset_saved_descriptor_vars()
        self.reset_saved_inheritance_vars()
        self.id = id

    def set_status(self, status: GraphNode.SegmentStatus) -> None:
        self.status = status

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

        # Remove ourself from the to_delete set if we are in it
        to_delete.discard(self)
    
        return to_delete
    
    def remove_from_graph_complete(self, keep_children: bool = True) -> list[int]:
        """ Does so by disconnecting self from parent both ways. Also immediately deletes any parent nodes that are now invalid. 
            Returns ids of additional nodes that were also removed (not including self). """

        # TODO: If keep_children is false, then need to update saved variables

        deleted_ids = []
        to_delete = self.remove_from_graph(keep_children)
        while to_delete:
            node_to_delete = min(to_delete, key=lambda n: n.get_id())
            deleted_ids.append(node_to_delete.get_id())
            to_delete.remove(node_to_delete)
            to_delete.update(node_to_delete.remove_from_graph(keep_children))
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
            self.point_cloud = self.point_cloud[mask]
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
        to_delete: set[GraphNode] = self.remove_points(pc_to_remove)
        while to_delete:
            node_to_delete = min(to_delete, key=lambda n: n.get_id())
            to_delete.remove(node_to_delete)
            to_delete.update(node_to_delete.remove_from_graph())
    
    def remove_child(self, child: GraphNode) -> set[GraphNode]:
        if child in self.child_nodes:
            self.child_nodes.remove(child)

            if self.params.parent_node_inherits_descriptors_from_children:
                self.reset_saved_descriptor_vars()
            if self.params.parent_node_inherits_data_from_children:
                self.reset_saved_inheritance_vars()
                return self.reset_saved_point_vars()
            else:
                return set()
        else:
            raise ValueError(f"Tried to remove {child} from {self}, but {child} not in self.child_nodes: {self.child_nodes}")
        
    def remove_children(self, children: list[GraphNode]) -> set[GraphNode]:
        nodes_to_delete = set()
        for child in children[:]:
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
        if self.params.parent_node_inherits_descriptors_from_children:
            self.reset_saved_descriptor_vars()
        if self.params.parent_node_inherits_data_from_children:
            self.reset_saved_inheritance_vars()
            self.reset_saved_point_vars_safe()

    def add_children(self, new_children: list[GraphNode]) -> None:
        for new_child in new_children:
            self.add_child(new_child)

    def add_semantic_descriptors(self, descriptors: list[tuple[np.ndarray, float]]) -> None:
        """ Keep track of all descriptors for batched calculation """
        self.semantic_descriptors += descriptors
        self.reset_saved_descriptor_vars()

    def add_semantic_descriptors_incremental(self, descriptor: np.ndarray, count: int) -> None:
        """ For calculating the semantic descriptor in the same way as ROMAN """

        # Hold the observation descriptor in case this node is merged through hungarian assignment
        if self.params.ignore_descriptors_from_observation and self.obs_descriptor is None:
            assert count == 1
            self.obs_descriptor = descriptor
            return

        # Otherwise, add the descriptor to our current counter
        if self.semantic_descriptor_inc is None:
            assert count == 1, "Multiple Initialization of Semantic Descriptor Incremental"
            self.semantic_descriptor_inc = descriptor / np.linalg.norm(descriptor)
            self.semantic_descriptor_inc_count = count
        else:
            self.semantic_descriptor_inc = (
                self.semantic_descriptor_inc * self.semantic_descriptor_inc_count
                / (self.semantic_descriptor_inc_count + count) 
                + descriptor * count / np.linalg.norm(descriptor)
                / (self.semantic_descriptor_inc_count + count)
            )
            self.semantic_descriptor_inc_count += count
        self.semantic_descriptor_inc /= np.linalg.norm(self.semantic_descriptor_inc) # renormalize

        self.reset_saved_descriptor_vars()

    def update_point_cloud(self, new_points: np.ndarray, run_dbscan: bool = False, remove_outliers: bool | None = None,
                           downsample: bool = True) -> set[GraphNode]:
        """ Returns nodes that might need to be deleted due to cleanup removing points..."""
        
        # TODO: Should I consider children point clouds are considered when downsampling, removing outliers, etc?
        # If so, will need to either run downsampling and then pass points back to children (since points are changed),
        # or make my own custom method that does downsampling but keeping the original points.

        # =========== Add to Point Cloud ============
        
        # Skip math if no new points are included
        if new_points.shape[0] != 0: 

            # Check the input array is the shape we expect
            if new_points.shape[1] != 3: raise ValueError(f"Point array in a non-supported shape: {new_points.shape()}")

            # Append them to our point cloud
            self.point_cloud = np.concatenate((self.point_cloud, new_points), axis=0)
            # self.point_cloud = np.unique(self.point_cloud, axis=0) # Prune any duplicates
            self.reset_saved_point_vars_safe() # Wipe saved point cloud for next steps
        
        # =========== Clean-up Point Cloud  ============
        # Necessary to limit sizes of point clouds for computation purposes and for ensuring incoming point clouds only represent a single object.
        # Considers child cloud as part of self for determining how to downsample, remove outliers, and cluster. 

        # Perform DBSCAN clustering (if desired)
        if run_dbscan:
            self._dbscan_clustering()

        # Run a downsampling operation to keep the point clouds small enough for real-time
        if downsample:
            pcd = o3d.geometry.PointCloud()
            pcd.points.extend(self.point_cloud)
            voxel_size = self.params.voxel_size_not_variable
            if self.params.enable_variable_voxel_size:
                length = self.get_longest_line_size()
                if length is not None and length > 0:
                    voxel_size = length * self.params.voxel_size_variable_ratio_to_length
                else:
                    raise RuntimeError(f"Trying to use variable voxel size for Node {self.get_id()}, but length is invalid: {length}")
              
            pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)

            # Update the point cloud
            self.point_cloud = np.asarray(pcd_sampled.points)
            self.reset_saved_point_vars_safe()

        # Remove statistical outliers
        if remove_outliers is None:
            remove_outliers = self.params.enable_remove_statistical_outliers
        if remove_outliers:
            self._remove_statistical_outliers()

        # Reset point cloud dependent saved variables and return nodes to delete
        return self.reset_saved_point_vars()

    def _dbscan_clustering(self, reset_voxel_grid: bool = True) -> None:
        """ Run DBScan clustering to cleanup the point cloud"""
        # NOTE: This actually ISN'T a safe operation, so calling method MUST call reset_saved_point_vars().

        if self.point_cloud is not None and len(self.point_cloud) != 0:
            # Convert into o3d PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)

            # Determine the epsilon to use
            epsilon = self.params.epsilon_not_variable
            if self.params.enable_variable_epsilon:
                length = self.get_longest_line_size()
                epsilon = length * self.params.epsilon_variable_ratio_to_length

            # Perform clustering
            labels: np.ndarray = np.array(pcd.cluster_dbscan(eps=epsilon, min_points=self.params.dbscan_min_points))

            # Handle the clusters differently depending on if we are ROMAN or MERONOMY
            if self.params.enable_roman_dbscan:

                # Number of clusters, ignoring noise if present
                max_label = labels.max()
                if max_label == -1:
                    logger.info(f"[bright_red]WARNING[/bright_red]: Largest cluster for Node {self.get_id()} is noise! Aborting DBScan... ")
                    return
                
                # Get largest cluster
                cluster_sizes = np.zeros(max_label + 1)
                for i in range(max_label + 1):
                    cluster_sizes[i] = np.sum(labels == i)
                max_cluster: np.int64 = np.argmax(cluster_sizes)

                # Filter out any points not belonging to max cluster
                filtered_indices = np.where(labels == max_cluster)[0]
                self.point_cloud = self.point_cloud[filtered_indices]
                self.reset_saved_point_vars_safe(reset_voxel_grid)
            else:

                # Find cluster with largest size deterministically (break ties by smaller cluster ID)
                cluster_ids, counts = np.unique(labels, return_counts=True)
                cluster_id_to_count: dict = dict(zip(cluster_ids, counts))
                max_cluster_index = min([cid for cid, cnt in cluster_id_to_count.items() if cnt == max(cluster_id_to_count.values())])
                if max_cluster_index == -1:
                    logger.info(f"[bright_red]WARNING[/bright_red]: Largest cluster in this node {self.get_id()} have been detected as noise!")

                # Check size of max cluster
                max_cluster_size = np.sum(labels == max_cluster_index)
                cluster_size_ratio = max_cluster_size / len(pcd.points)
                if cluster_size_ratio < self.params.min_cluster_percentage:

                    # Since this cluster is too small, the semantic embedding will not be 
                    # representative. Thus, we must delete this node, so wipe our point cloud.
                    self.point_cloud = np.zeros((0, 3), dtype=np.float64)
                    self.reset_saved_point_vars_safe(reset_voxel_grid)
                    return

                # Filter out any points not belonging to max cluster
                filtered_indices = np.asarray(labels == max_cluster_index).nonzero()
                clustered_points = np.asarray(pcd.points)[filtered_indices]

                # Save the new sampled point cloud 
                self.point_cloud = clustered_points
                self.reset_saved_point_vars_safe(reset_voxel_grid)

    def _remove_statistical_outliers(self):

        # Convert into o3d PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)

        # Remove statistical outliers
        pcd_sampled, _ = pcd.remove_statistical_outlier(self.params.stat_out_num_neighbors, self.params.stat_out_std_ratio)

        # Save the new sampled point cloud 
        self.point_cloud = np.asarray(pcd_sampled.points)
        self.reset_saved_point_vars_safe()

        # NOTE: This actually ISN'T a safe operation, so calling method MUST call reset_saved_point_vars().

    # ==================== Merging ====================
    def merge_with_node_mapping(self, other: GraphNode, keep_children=True, new_id: int | None = None) -> GraphNode | None:
        """
        As opposed to merge_with_observation (which can just be called), this method 
        will take out self and other from the graph and return a new node. This new
        node needs to be inserted back into the graph by the SceneGraph3D.

        NOTE: other cannot be a descendent or ascendent of self!
        NOTE: If the new node is invalid, then will just return None.
        """

        # Make sure they are not related
        assert not (self.is_descendent_or_ascendent(other) and keep_children), "This method doesn't support merging ascendent/descendent nodes if keep_children is true!"

        # Remove both nodes (and all descendants) from the graph
        self.remove_from_graph_complete(keep_children)
        other.remove_from_graph_complete(keep_children)

        # Combine only semantic descriptors in the two nodes directly, not from children
        combined_descriptors = self.semantic_descriptors + other.semantic_descriptors

        # Do the same with point clouds specific to these two nodes, not children
        combined_pc = np.concatenate((self.point_cloud, other.point_cloud), axis=0)

        # Make a list of children
        combined_children = (self.get_children() + other.get_children())

        # Calculate the first seen time as earliest from the two nodes
        if self.params.merge_with_node_use_first_seen_time_from_self:
            first_seen = self.get_time_first_seen()
        else:
            first_seen = min(self.get_time_first_seen(), other.get_time_first_seen())

        # Also calculate the first pose
        if first_seen == self.get_time_first_seen(): first_pose = self.get_first_pose()
        else: first_pose = other.get_first_pose()

        # Caluclate the last updated time
        last_updated = max(self.get_time_last_updated(), other.get_time_last_updated())

        # Create a new node representing the merge
        if new_id is None:
            new_id = self.get_id() if self.get_id() < other.get_id() else other.get_id()
        new_node = GraphNode.create_node_if_possible(new_id, None, combined_descriptors, self.semantic_descriptor_inc,
                        self.semantic_descriptor_inc_count, combined_pc, combined_children, first_seen, 
                        last_updated, self.curr_time, first_pose, self.curr_pose, self.curr_pose, run_dbscan=False)
        if new_node is None:
            return None
        to_delete = new_node.update_point_cloud(np.zeros((0,3)), run_dbscan=False, remove_outliers=True, downsample=True)
        if len(to_delete) > 0:
            raise RuntimeError("Need to delete some nodes after downsampling point cloud!")
        if other.semantic_descriptor_inc is not None:
            new_node.obs_descriptor = np.zeros((0, 1))
            new_node.add_semantic_descriptors_incremental(other.semantic_descriptor_inc, other.semantic_descriptor_inc_count)

        # Update the number of sightings
        total_sightings = self.get_num_sightings() + other.get_num_sightings()
        new_node.num_sightings = total_sightings
            
        # Tell our children who their new parent is
        for child in combined_children:
            child.set_parent(new_node)
        return new_node
    
    def merge_with_node_meronomy(self, other: GraphNode, keep_children=True, new_id: int | None = None) -> GraphNode | None:
        """
        As opposed to merge_with_observation (which can just be called), this method 
        will take out self and other from the graph and return a new node. This new
        node needs to be inserted back into the graph by the SceneGraph3D.

        NOTE: other cannot be a descendent or ascendent of self!
        NOTE: If the new node is invalid, then will just return None.
        """

        # Make sure they are not related
        if  self.is_descendent_or_ascendent(other) and keep_children:
            # Keep children not possible in this situation, so turn it off
            keep_children = False

        # Remove both nodes (and all descendants) from the graph
        self.remove_from_graph_complete(keep_children)
        other.remove_from_graph_complete(keep_children)

        # Combine only semantic descriptors in the two nodes directly, not from children
        combined_descriptors = self.semantic_descriptors + other.semantic_descriptors

        # Do the same with point clouds specific to these two nodes, not children
        combined_pc = np.concatenate((self.point_cloud, other.point_cloud), axis=0)

        # Make a list of children
        combined_children = (self.get_children() + other.get_children())

        # Create a new node representing the merge
        if new_id is None:
            new_id = self.get_id() if self.get_id() < other.get_id() else other.get_id()
        new_node = GraphNode.create_node_if_possible(new_id, None, combined_descriptors, self.semantic_descriptor_inc,
                        self.semantic_descriptor_inc_count, combined_pc, combined_children, 0.0, 0.0, 0.0, np.eye(4),
                        np.eye(4), np.eye(4), run_dbscan=False)
        if new_node is None:
            return None
        if other.semantic_descriptor_inc is not None:
            new_node.obs_descriptor = np.zeros((0, 1))
            new_node.add_semantic_descriptors_incremental(other.semantic_descriptor_inc, other.semantic_descriptor_inc_count)
 
        # Tell our children who their new parent is
        for child in combined_children:
            child.set_parent(new_node)
        return new_node


    def merge_with_observation(self, new_pc: np.ndarray, new_descriptors: list[tuple[np.ndarray, float]] | None, 
                               descriptor_inc: np.ndarray | None) -> None:

        # Save the semantic embedding into this parent graph node
        if new_descriptors is not None: 
            self.add_semantic_descriptors(new_descriptors)
        if descriptor_inc is not None:
            self.add_semantic_descriptors_incremental(descriptor_inc, 1)

        # If desired, split the new point cloud into children and self
        if self.params.parent_node_inherits_data_from_children:
            # Get convex hulls of each child
            hulls: list[trimesh.Trimesh] = []
            for child in self.get_children():
                hulls.append(child.get_convex_hull())
            
            # Get masks of which points fall into which hulls
            contain_masks = find_point_overlap_with_hulls(new_pc, hulls, fail_on_multi_assign=False)
            # TODO: Add statement to check if this multi-assignment happens too often!

            # Based on point assignments, update each child node
            for i, child in enumerate(self.get_children()):
                child_pc = new_pc[contain_masks[i],:]
                child.merge_with_observation(child_pc, None, None)
                
            # Find points that had no assignment
            num_mask_assignments = np.sum(contain_masks, axis=0)
            orphan_mask = np.where(num_mask_assignments == 0)[0]
            orphan_pc = new_pc[orphan_mask,:]

            # If there are at least one point in this orphan point cloud, add them to this node's cloud
            if orphan_pc.shape[0] > 1:
                to_delete = self.update_point_cloud(orphan_pc, remove_outliers=False)
                if len(to_delete) > 0:
                    raise RuntimeError(f"Cannot merge_with_observation; Node {self.get_id()}'s point cloud invalid after adding additional points, which should never happen!")
        
        # Otherwise, just give the pointcloud to self
        else:
            to_delete = self.update_point_cloud(new_pc, remove_outliers=True, downsample=True)
            if len(to_delete) > 0:
                raise RuntimeError(f"Cannot merge_with_observation; Node {self.get_id()}'s point cloud invalid after adding additional points, which should never happen!")
            
        # Increase our sightings
        self.num_sightings += 1

        # Update our last seen time and pose
        self.last_updated = self.curr_time
        self.last_pose = self.curr_pose
            
    def merge_parent_and_child(self, other: GraphNode, new_id: int | None = None) -> GraphNode:
        """ Merge child into parent and keep parent, return parent node. """

        # Determine which node is the parent
        if self.is_parent(other): 
            parent_node = self
            child_node = other
            self_is_parent = True
        else: 
            parent_node = other
            child_node = self
            self_is_parent = False

        # Conduct the merge
        parent_node.merge_child_with_self(child_node, new_id=new_id)
        return parent_node

    def merge_child_with_self(self, other: GraphNode, new_id: int | None = None) -> None:
        
        # Make sure other is a child of self
        if not self.is_parent(other) or not other in self.get_children():
            raise ValueError("Cannot merge_child_with_self; node {other} is not a child of self!")
        
        # Add semantic descriptors specific to child (not from grandchidren) to this node
        self.add_semantic_descriptors(other.semantic_descriptors)
        self.add_semantic_descriptors_incremental(other.semantic_descriptor_inc, other.semantic_descriptor_inc_count)

        # Increase our sightings
        self.num_sightings += other.num_sightings
        
        # Do the same with point cloud specific to the child (not grandchildren)
        to_delete = self.update_point_cloud(other.point_cloud, remove_outliers=False)
        if len(to_delete) > 0:
            raise RuntimeError(f"Cannot merge_child_with_self; New point cloud is invalid, this should never happen")

        # Add grandchildren as children and add self as grandchildrens' parent
        for grandchild in other.get_children():
            self.add_child(grandchild)
            grandchild.set_parent(self)

        # Update the id if desired
        if new_id is not None:
            self.set_id(new_id)

        # Remove child
        other.remove_from_graph_complete()

    # ==================== Resetting Vars ====================
    def reset_saved_point_vars(self, reset_voxel_grid: bool = True) -> set[GraphNode]:
        """ 
        Wipes saved point variables since they need to be recalculated. 
        Returns list of nodes that are no longer valid and should be removed. 
        """

        # Do nothing if we are the RootGraphNode
        if self.is_RootGraphNode():
            return set()
        
        # Wipe all variables
        self._convex_hull = None
        if reset_voxel_grid:
            self._voxel_grid = None
            self._oriented_bbox = None
        self._point_cloud = None
        self._longest_line_size = None
        self._centroid = None

        # Make sure SceneGraph3D knows to redo some calculations
        self._redo_convex_hull_geometric_overlap = True
        self._redo_shortest_dist_between_convex_hulls = True

        # Track nodes that might need to be deleted...
        to_delete = set()

        # Check if we can still make a ConvexHull...
        if self.params.require_valid_convex_hull and self.get_convex_hull() is None:
            to_delete.add(self)

        # Reset variables in parents and get any of those that need to be deleted.
        if self.parent_node is not None and self.params.parent_node_inherits_data_from_children:
            to_delete.update(self.parent_node.reset_saved_point_vars(reset_voxel_grid))

        # Return nodes that need to be deleted
        return to_delete
    
    def reset_saved_point_vars_safe(self, reset_voxel_grid: bool = True) -> None:
        """ 
        Similar to reset_saved_point_vars(), but called if points were only
        possibly added to a node. Thus, no need to check node validity
        or return nodes that might need to be deleted.
        """
        
        # Wipe all variables
        self._convex_hull = None
        if reset_voxel_grid:
            self._voxel_grid = None
            self._oriented_bbox = None
        self._point_cloud = None
        self._longest_line_size = None
        self._centroid = None

        # Make sure SceneGraph3D knows to redo some calculations
        self._redo_convex_hull_geometric_overlap = True
        self._redo_shortest_dist_between_convex_hulls = True

        # Reset variables in parents 
        if self.parent_node is not None and self.params.parent_node_inherits_data_from_children:
            self.parent_node.reset_saved_point_vars_safe(reset_voxel_grid)
    
    def reset_saved_descriptor_vars(self) -> None:
        """ Wipes saved descriptor variables as they need to be recalculated """

        # Track the previous word to see if it changed
        prev_word = None
        if self._word is not None:
            prev_word = self._word

        # Wipe variables
        self._semantic_descriptor = None
        self._word = None

        # Get the new word and see if it changed
        if prev_word is None or not self.get_word() == prev_word:

            # Reset all our meronyms & holonyms since it changed            
            self._meronyms = defaultdict(lambda: None)
            self._holonyms = defaultdict(lambda: None)
            self._holonyms_pure = defaultdict(lambda: None)

            # Also let the scene graph know that some word comparisons will need rechecking
            self._redo_word_comparisons = True

        # Do the same in parents
        if self.parent_node is not None and self.params.parent_node_inherits_descriptors_from_children:
            self.parent_node.reset_saved_descriptor_vars()

    def reset_saved_inheritance_vars(self) -> None:

        # Wipe variables
        self._descendents = None

        # Also wipe this in parents
        print("")
        if self.parent_node is not None:
            self.parent_node.reset_saved_inheritance_vars()

    # ==================== Iterator ====================
    def __iter__(self) -> Iterator[GraphNode]:
        stack: list[GraphNode] = [self]
        while stack:            
            node = stack.pop()
            yield node
            stack.extend(node.get_children())

