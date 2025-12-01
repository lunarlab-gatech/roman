from __future__ import annotations

from collections import defaultdict
import cv2 as cv
from enum import Enum
import heapq
from .scene_graph_utils import get_convex_hull_from_point_cloud, longest_line_of_point_cloud
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

    params: GraphNodeParams
    wordnet_wrapper: WordNetWrapper | None = None # Set externally before use

    # ================ Initialization =================
    def __init__(self, id: int, parent_node: GraphNode | None, semantic_descriptor: np.ndarray, 
                 point_cloud: np.ndarray, child_nodes: list[GraphNode], is_RootGraphNode: bool = False):
        """
        Don't create a Graph Node directly! Instead do it with create_node_if_possible(), 
        as node can be invalid after creation.
        """
        
        # Node ID
        self.id: int = id           

        # RootGraphNode handles id assignments, but all nodes have this        
        self._next_id: int = id + 1
        self._forfeited_ids = []  # min-heap of released IDs

        # The parent node to this node.
        self.parent_node: GraphNode | None = parent_node

        # Any child nodes we might have. TODO: Make it a set, so duplicate children can't occur.
        self.child_nodes: list[GraphNode] = child_nodes
        
        # Semantic descriptor for this node
        self.semantic_descriptor: np.ndarray  = semantic_descriptor

       # Information tracking if we are the RootGraphNode
        self.is_root = is_RootGraphNode

        # Points that relate only to this object and not any children (expressed in world frame).
        self.point_cloud: np.ndarray = np.zeros((0, 3), dtype=np.float64)

        # Track if this is an original segment or a newly created node
        self.is_meronomy_created_or_altered: bool = False

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
        to_delete = self.update_point_cloud(point_cloud)
        if len(to_delete) > 0:
            self._class_method_creation_success = False

        # If it hasn't failed yet, check if our ConvexHull is valid (if requested)
        elif self.params.require_valid_convex_hull:
            hull = self.get_convex_hull()
            if hull is None:
                self._class_method_creation_success = False
        
        # If we are RootGraphNode, creation is always successful as we don't 
        # ever use our ConvexHull or Point Cloud
        if self.is_RootGraphNode():
            self._class_method_creation_success = True

    @classmethod
    def create_node_if_possible(cls, id: int, parent_node: GraphNode | None, semantic_descriptor: np.ndarray, 
                                point_cloud: np.ndarray, child_nodes: list[GraphNode], is_RootGraphNode: bool = False) -> GraphNode | None:
        """ This method will create and return a GraphNode if that node is valid. """

        # Create node and run dbscan to filter out extra objects included in one faulty segmentation
        potential_node = cls(id, parent_node, semantic_descriptor, point_cloud, child_nodes, is_RootGraphNode=is_RootGraphNode)
        
        # Return the node if node creation was successful
        if potential_node._class_method_creation_success: return potential_node
        else: return None

    # ==================== Predicates ====================
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
    
    def is_sibling(self, other: GraphNode) -> bool:
        if self.is_RootGraphNode():
            return False
        return self.get_parent() == other.get_parent()
    
    def is_descendent_or_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if this node is a descendent or ascendent of other."""
        return other.is_ascendent(self) or self.is_ascendent(other)
    
    def is_ascendent(self, other: GraphNode) -> bool:
        """ Returns True if self is an ascendent of other."""

        if self is other: return False
        descendents: list[GraphNode] = self.get_descendents()
        if other in descendents: return True
        else: return False

    def is_RootGraphNode(self) -> bool:
        return self.is_root

    def is_descendent_of_meronomy_created_or_altered_node(self) -> bool:
        """ Returns True if this node is a descendent of a meronomy created/altered node. """

        if self.is_RootGraphNode():
            return False

        # Check if any of our ancestors are meronomy created/altered nodes
        current_node: GraphNode = self.get_parent()
        while True:
            if current_node.is_meronomy_created_or_altered:
                return True
            if current_node.is_RootGraphNode():
                return False
            current_node = current_node.get_parent()
        
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

    # ==================== Getters ====================
    def get_id(self) -> int:
        return self.id
    
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
           
    def get_convex_hull(self) -> trimesh.Trimesh | None:
        if self.is_RootGraphNode():
            return None
        if self._convex_hull is None:
            self._convex_hull = get_convex_hull_from_point_cloud(self.get_point_cloud())

        return self._convex_hull
    
    def get_voxel_grid(self, voxel_size: float) -> VoxelGrid | None:
        if self.is_RootGraphNode():
            return None
        if self._voxel_grid is None:
            self._voxel_grid = VoxelGrid.from_points(self.get_point_cloud(), voxel_size)
        return self._voxel_grid
    
    def get_num_points(self):
        if self.is_RootGraphNode() or self.get_point_cloud() is None:
            return 0
        else:
            return self.get_point_cloud().shape[0]
    
    def get_semantic_descriptor(self) -> np.ndarray:
        return self.semantic_descriptor
    
    def get_word(self) -> WordWrapper | None:
        if self._word is None:
            descriptor = self.get_semantic_descriptor()
            if descriptor is None:
                self._word = None
            else:
                self._word = WordWrapper.from_word(self.wordnet_wrapper.map_embedding_to_words(descriptor, 1)[0])
        return self._word
    
    def get_all_meronyms(self, meronym_level: int = 1) -> set[str]:
        if self._meronyms[meronym_level] is None:
            self._meronyms[meronym_level] = self.get_word().get_all_meronyms(True, meronym_level)
        return self._meronyms[meronym_level]
    
    def get_all_holonyms(self, include_hypernyms: bool, holonym_level: int = 1) -> set[str]:
        if self._holonyms[holonym_level] is None:
            self._holonyms[holonym_level] = self.get_word().get_all_holonyms(include_hypernyms, holonym_level)
        return self._holonyms[holonym_level]
    
    def get_descendents(self) -> list[GraphNode]:
        if self._descendents is None:
            self._descendents = []
            self._descendents += self.get_children()
            for child in self.get_children():
                self._descendents += child.get_descendents()
        return self._descendents
    
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
            full_pc = np.zeros((0, 3), dtype=np.float64)
            for child in self.get_children():
                full_pc = np.concatenate((full_pc, child.get_point_cloud()), dtype=np.float64)
            self._point_cloud = np.concatenate((full_pc, self.point_cloud), dtype=np.float64)

        return self._point_cloud

    def get_children(self) -> list[GraphNode]:
        return self.child_nodes
        
    
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

    def set_is_meronomy_created_or_altered(self, is_created_or_altered: bool) -> None:
        self.is_meronomy_created_or_altered = is_created_or_altered

    # ==================== Updating / Adding ====================
    def add_child(self, new_child: GraphNode) -> None:
        if new_child in self.child_nodes:
            return # Shouldn't add children more than once
        
        self.child_nodes.append(new_child)
        self.reset_saved_inheritance_vars()
        self.reset_saved_point_vars_safe()

    def add_children(self, new_children: list[GraphNode]) -> None:
        for new_child in new_children:
            self.add_child(new_child)

    def update_point_cloud(self, new_points: np.ndarray) -> set[GraphNode]:
        """ Returns nodes that might need to be deleted due to cleanup removing points..."""
        
        if new_points.shape[0] != 0: 

            # Check the input array is the shape we expect
            if new_points.shape[1] != 3: raise ValueError(f"Point array in a non-supported shape: {new_points.shape()}")

            # Append them to our point cloud
            self.point_cloud = np.concatenate((self.point_cloud, new_points), axis=0)

            # Reset point cloud dependent saved variables and return nodes to delete
            return self.reset_saved_point_vars()
        
        else: return set()

    # ==================== ID Management ====================
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

    # ==================== Conversions ====================
    def to_segment(self) -> Segment:
        """Returns a segment representation of this graph node"""

        # Create a Segment
        obs = Observation(0.0, np.eye(4), None, None, None, None, None, None, None)
        seg = Segment(obs, None, self.get_id(), None)

        # Update internal values of Segment so it matches Graph Node
        seg.last_seen = None
        seg.num_sightings = None
        seg.points = self.get_point_cloud()
        seg.semantic_descriptor = self.get_semantic_descriptor()
        seg.semantic_descriptor_cnt = None

        return seg
    
    @staticmethod
    def from_segment(seg: Segment) -> GraphNode | None:
        """ Creates a GraphNode from a Segment if possible, else returns None. """
        new_node: GraphNode | None = GraphNode.create_node_if_possible(seg.id, None, seg.semantic_descriptor, seg.points, [])
        if new_node is None: return None # Node creation failed
        return new_node

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

    def remove_child(self, child: GraphNode) -> set[GraphNode]:
        if child in self.child_nodes:
            self.child_nodes.remove(child)
            self.reset_saved_inheritance_vars()
            return self.reset_saved_point_vars()
        else:
            raise ValueError(f"Tried to remove {child} from {self}, but {child} not in self.child_nodes: {self.child_nodes}")
        
    def remove_children(self, children: list[GraphNode]) -> set[GraphNode]:
        nodes_to_delete = set()
        for child in children[:]:
            nodes_to_delete.update(self.remove_child(child))
        return nodes_to_delete

    # ==================== Merging ====================
    def merge_with_node_meronomy(self, other: GraphNode, keep_children=True, new_id: int | None = None) -> GraphNode | None:
        """
        As opposed to merge_with_observation (which can just be called), this method 
        will take out self and other from the graph and return a new node. This new
        node needs to be inserted back into the graph by the SceneGraph3D.

        NOTE: other cannot be a descendent or ascendent of self!
        NOTE: If the new node is invalid, then will just return None.
        NOTE: Keeps descriptor of self, throws away descriptor of other.
        """

        # Make sure they are not related
        if  self.is_descendent_or_ascendent(other) and keep_children:
            # Keep children not possible in this situation, so turn it off
            keep_children = False

        # Remove both nodes (and all descendants) from the graph
        self.remove_from_graph_complete(keep_children)
        other.remove_from_graph_complete(keep_children)

        # Do the same with point clouds specific to these two nodes, not children
        combined_pc = np.concatenate((self.point_cloud, other.point_cloud), axis=0)

        # Make a list of children
        combined_children = (self.get_children() + other.get_children())

        # Create a new node representing the merge
        if new_id is None:
            new_id = self.get_id() if self.get_id() < other.get_id() else other.get_id()
        new_node = GraphNode.create_node_if_possible(new_id, None, self.get_semantic_descriptor(), combined_pc, combined_children, run_dbscan=False)
        if new_node is None:
            return None
 
        # Tell our children who their new parent is
        for child in combined_children:
            child.set_parent(new_node)
        return new_node
            
    def merge_parent_and_child(self, other: GraphNode, new_id: int | None = None) -> GraphNode:
        """ Merge child into parent and keep parent, return parent node. """

        # Determine which node is the parent
        if self.is_parent(other): 
            parent_node = self
            child_node = other
        else: 
            parent_node = other
            child_node = self

        # Conduct the merge
        parent_node.merge_child_with_self(child_node, new_id=new_id)
        return parent_node

    def merge_child_with_self(self, other: GraphNode, new_id: int | None = None) -> None:
        """ NOTE: Descriptors of child thrown away, keeping descriptors of parent only. """
        
        # Make sure other is a child of self
        if not self.is_parent(other) or not other in self.get_children():
            raise ValueError("Cannot merge_child_with_self; node {other} is not a child of self!")
           
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
        if self.parent_node is not None:
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
        if self.parent_node is not None:
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

    def reset_saved_inheritance_vars(self) -> None:
        """ Wipes saved variables that depend on inheritance from children. """

        self._descendents = None
        if self.parent_node is not None:
            self.parent_node.reset_saved_inheritance_vars()

    # ==================== Iterator ====================
    def __iter__(self) -> Iterator[GraphNode]:
        stack: list[GraphNode] = [self]
        while stack:            
            node = stack.pop()
            yield node
            stack.extend(node.get_children())

