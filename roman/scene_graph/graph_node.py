from __future__ import annotations

from enum import Enum
from ..map.observation import Observation
import numpy as np
from numpy.typing import NDArray
from ..object.segment import Segment
import trimesh
from .scene_graph_3D import SceneGraph3D
from scipy.spatial import ConvexHull
from typeguard import typechecked

class GraphNode():
    parent_node: GraphNode | None
    dummy_parent_node: GraphNode | None
    id: int
    is_dummy: bool

    class State(Enum):
        Nursery = 0
        Current = 1
        Inactive = 2
        Graveyard = 3 

    @typechecked
    def __init__(self, id: int, parent_node: GraphNode | None, is_dummy: bool):
        self.id = id
        self.parent_node = parent_node
        self.is_dummy = is_dummy
        self.reset_similarity_holders()

    # Class Methods
    @classmethod
    def create_dummy_node(cls, id: int, parent_node: GraphNode | None):
        return cls(id, parent_node, True)

    # Getters 
    def noParent(self) -> bool:
        if self.parent_node is None: return True
        else: return False
    
    def get_convex_hull(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    def get_semantic_descriptors(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    @typechecked
    def get_weighted_semantic_descriptor(self) -> np.ndarray:
        return self.calculate_weighted_semantic_descriptor(self.get_semantic_descriptors())
    
    def get_point_cloud(self):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    # Calculations
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

    # Setters 
    @typechecked
    def setParent(self, node: GraphNode):
        self.parent_node = node

    @typechecked
    def setDummyParent(self, node: GraphNode):
        self.dummy_parent_node = node

    # Manipulators
    @typechecked
    def addChild(self, node: GraphNode):
        raise NotImplementedError("Use a child class, not GraphNode itself!")
    
    # Iterator
    def __iter__(self):
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            if isinstance(node, ParentGraphNode):
                stack.extend(node.child_nodes)
    
    # def reset_similarity_holders(self):
    #     self.sem_con = None
    #     self.iou = None
    #     self.parent_enclosure = None
    #     self.child_enclosures = []

    # def get_final_similiarity_score(self) -> float:
    #     # Deal with edge case of no parents/children
    #     if len(self.enclosure) == 0:
    #         self.enclosure.append(0.5)
    #     np.mean(self.enclosure)

    # @typechecked
    # def apply_enclosure_parents(self, enclosure: float):
    #     if not self.is_dummy:
    #         if self.parent_node is not None:
    #             self.parent_node.child_enclosures.append(enclosure)
    #         if self.dummy_parent_node is not None:
    #             self.dummy_parent_node.child_enclosures.append(enclosure)

    # def apply_enclosure_children(self, enclosure):
    #     raise NotImplementedError("Use a child class, not GraphNode itself!")

class ParentGraphNode(GraphNode):
    child_nodes: list[GraphNode]

    # === Embeddings and Points specific to this level of the hierarchhy ===

    # As Clip embeddings can't be split and passed to children,
    # track embeddings specific to this parent here. Will be combined
    # with child embeddings before returning the final embedding.
    semantic_descriptors: list[tuple[np.ndarray, float]] = []

    # Points that relate only to this object and not any children.
    # Ex. points on a wall, whose children are a painting and a window.
    specific_pc: np.ndarray = np.zeros((0, 3), dtype=np.float64)

    @typechecked
    def __init__(self, id: int, parent_node: GraphNode | None, is_dummy: bool, child_nodes: list[GraphNode]):
        super().__init__(id, parent_node, is_dummy)
        self.child_nodes = child_nodes
        self.curr_segment = None
    
    @typechecked
    def get_number_of_nodes(self) -> int:
        num = 1
        for child in self.child_nodes:
            num += child.get_number_of_nodes(child)
        return num
    
    @typechecked
    def get_convex_hull(self)-> trimesh.Trimesh:
        pc = self.get_point_cloud()
        hull = ConvexHull(pc)
        mesh = trimesh.Trimesh(vertices=pc, faces=hull.simplices, process=True)
        mesh.fix_normals()
        return mesh
    
    @typechecked
    def get_semantic_descriptors(self):
        descriptors = []
        for child in self.child_nodes:
            descriptors += child.get_semantic_descriptors()
        descriptors += self.semantic_descriptors
        return descriptors
    
    @typechecked
    def get_point_cloud(self) -> np.ndarray:
        full_pc = np.zeros((0, 3), dtype=np.float64)
        for child in self.child_nodes:
            full_pc = np.concatenate((full_pc, child.get_point_cloud()), dtype=np.float64)
        full_pc = np.concatenate((full_pc, self.specific_pc), dtype=np.float64)
        return full_pc
    
    # Manipulators 
    @typechecked
    def update_with_observation(self, obs: Observation):

        # Save the semantic embedding into this parent graph node
        self.semantic_descriptors.append([(obs.clip_embedding, ConvexHull(obs.point_cloud).volume)])

        # Get convex hulls of each child
        hulls: list[trimesh.Trimesh] = []
        for child in self.child_nodes:
            hulls.append(child.get_convex_hull())
        
        # Get masks of which points fall into which hulls
        contain_masks = SceneGraph3D.find_point_overlap_with_hulls(obs.point_cloud, hulls, fail_on_multi_assign=True)

        # Based on point assignments, create new observations for each child to be updated with
        for i, child in enumerate(self.child_nodes):
            child_pc = obs.point_cloud[contain_masks[i],:]
            child_obs = Observation(time=obs.time, pose=obs.pose, mask=None, mask_downsampled=None,
                                    point_cloud=child_pc, clip_embedding=None)
            child.update_with_observation(child_obs)
            
        # Find points that had no assignment
        num_mask_assignments = np.sum(contain_masks, axis=0)
        orphan_mask = np.where(num_mask_assignments == 0)[0]
        orphan_pc = obs.point_cloud[orphan_mask,:]

        # If there are at least one point in this orphan point cloud
        if orphan_pc.shape[0] > 1:
            # Convert observed points to global frame
            R = obs.pose[:3,:3]
            T = obs.pose[:3,3].reshape((3,1))
            points_obs_body = orphan_pc.T
            num_points_obs = points_obs_body.shape[1]
            points_obs_world = R @ points_obs_body + np.repeat(T, num_points_obs, axis=1)

            # Add it to our own point cloud
            self.specific_pc = np.concatenate((self.specific_pc, points_obs_world), dtype=np.float64)

    @typechecked
    def addChild(self, node: GraphNode):
        raise NotImplementedError("NEEDS REWORKING")
        self.child_nodes.append(node)
        self.curr_segment = None

    # @typechecked
    # def apply_enclosure_children(self, enclosure: float):
    #     if not self.is_dummy:
    #         for child in self.child_nodes:
    #             child.parent_enclosure.append(enclosure)
    

class LeafGraphNode(GraphNode):
    """
    Basically just a wrapper for a segment.
    """
    segment: Segment

    @typechecked
    def __init__(self, id: int, parent_node: GraphNode | None, is_dummy: bool, segment: Segment):
        super().__init__(id, parent_node, is_dummy)
        self.segment = segment

    # Getters
    def get_number_of_nodes(self) -> int:
        return 1
    
    def get_convex_hull(self)-> trimesh.Trimesh:
        return self.segment.get_convex_hull()
    
    def get_semantic_descriptors(self):
        return self.segment.semantic_descriptors
    
    def get_point_cloud(self) -> np.ndarray:
        return self.segment.points

    # Manipulators
    @typechecked
    def update_with_observation(self, obs: Observation):
        self.segment.update(obs, integrate_points=True)

    @typechecked
    def addChild(self, node: GraphNode):
        raise NotImplementedError("NEEDS REWORKING")
        self = ParentGraphNode(self.id, self.parent_node, self.is_dummy, [node])