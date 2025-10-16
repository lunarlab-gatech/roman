from __future__ import annotations

import copy
from .graph_node import GraphNode
from ..logger import logger
from ..map.map import SubmapParams
import numpy as np
from robotdatapy.data.pose_data import PoseData
from robotdatapy.transform import transform
from roman.utils import transform_rm_roll_pitch
from .scene_graph_3D import SceneGraph3D
from typeguard import typechecked

@typechecked
class SceneGraph3DSubmap():

    def __init__(self, id: int, time: float, inactive_nodes: list[GraphNode], pose: np.ndarray, pose_gt: np.ndarray):
        # Save ID of this submap
        self.id: int = id

        # Time of the submap
        self.time: float = time

        # GraphNodes that fall within this submap
        self.inactive_nodes: list[GraphNode] = inactive_nodes

        # Pose of the robot wrt world (estimated & GT) for this submap
        self.pose: np.ndarray = pose
        self.pose_gt: np.ndarray = pose_gt

    @property
    def pose_gravity_aligned(self) -> np.ndarray:
        """ Returns the pose of the robot wrt world (estimated) with roll & pitch set to zero. """
        return transform_rm_roll_pitch(self.pose)
    
    @property
    def pose_gravity_aligned_gt(self):
        """ Returns the pose of the robot wrt world (GT) with roll & pitch set to zero. """
        return transform_rm_roll_pitch(self.pose_gt)
    
    @property
    def has_gt(self) -> bool:
        return self.pose_gt is not None
    
    @property
    def position(self) -> np.ndarray:
        return self.pose[:3,3]
    
    @property
    def position_gt(self) -> np.ndarray:
        return self.pose_gt[:3,3]

    @staticmethod
    def generate_submaps(map: SceneGraph3D, submap_params: SubmapParams, 
                            gt_flu_pose_data: PoseData | None = None) -> list[SceneGraph3DSubmap]:
        """ Breaks a SceneGraph3D into submaps. """
        
        logger.info(f"Total number of top-level nodes in this SceneGraph3D: {len(map.inactive_nodes)}")

        # Create number of submaps based on distance & time thresholds
        submaps: list[SceneGraph3DSubmap] = []
        for i, (pose, t) in enumerate(zip(map.poses, map.times)):
            if i == 0 or np.linalg.norm(pose[:-1,-1] - submaps[-1].pose[:-1,-1]) > submap_params.distance \
                or (t - submaps[-1].time > submap_params.time_threshold):
                submaps.append(SceneGraph3DSubmap(id=len(submaps), time=t, segments=[],
                    pose_flu=pose, pose_flu_gt=gt_flu_pose_data.pose(t) if gt_flu_pose_data is not None else None
                ))

        # Define helper function for checking time constraints
        def meets_time_constraints(node: GraphNode, time_submap_before: float, time_submap_after: float) -> bool:
            after_next_submap: bool = node.get_time_first_seen() > time_submap_after + submap_params.time_threshold
            before_prev_submap: bool = node.get_time_last_updated() < time_submap_before - submap_params.time_threshold
            return not (after_next_submap or before_prev_submap)

        # Add GraphNodes to submaps
        for i, submap in enumerate(submaps):
            
            # Calculate times of prev and next submaps
            time_submap_before: float = submaps[i-1].time if i > 0 else -np.inf
            time_submap_after: float = submaps[i+1].time if i < len(submaps) - 1 else np.inf

            # For each top-level node, check if within submap geometrically and in time.
            num_segments_in_submap_bounds: int = 0
            for node in map.inactive_nodes:
                if np.linalg.norm(node.get_centroid().flatten() - submap.pose[:3,3]) < submap_params.radius:
                    num_segments_in_submap_bounds += 1
                    if meets_time_constraints(node, time_submap_before, time_submap_after):
                        submap.inactive_nodes.append(copy.deepcopy(node))

            logger.info(f"Num of top-level nodes in submap bounds: {num_segments_in_submap_bounds}")
            logger.info(f"Num of top-level nodes additionally meeting time constraints:{len(submap.segments)}")

            # Calculate pose of world wrt robot's pose aligned with gravity (estimated)
            H_world_wrt_robot_aligned = np.linalg.inv(submap.pose_gravity_aligned)

            # Calculate points from the world frame to the robot frame aligned to gravity
            for node in submap.inactive_nodes:
                node.calculate_point_cloud_in_robot_frame_aligned(H_world_wrt_robot_aligned)

            # If submap has a max size, prune top-level nodes (and children) furthest from the center
            if submap_params.max_size is not None:
                nodes_sorted_by_dist = sorted(submap.inactive_nodes, 
                        key=lambda node: np.linalg.norm(node.get_centroid_robot_aligned().flatten()))
                submap.inactive_nodes = nodes_sorted_by_dist[:submap_params.max_size]
            
            logger.info(f"Num of segments after pruning to max_size: {len(submap.inactive_nodes)}")

        # Return the resulting submaps
        return submaps