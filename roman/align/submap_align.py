import argparse
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List
import trimesh
from trimesh.collision import CollisionManager
import open3d as o3d
import clipperpy
import time
from copy import deepcopy
import yaml

from robotdatapy.data.pose_data import PoseData
from robotdatapy.transform import transform, transform_to_xytheta, transform_to_xyzrpy
from robotdatapy.camera import CameraParams

from roman.rerun_wrapper.rerun_wrapper_window_alignment import RerunWrapperWindowAlignment
from roman.map.map import Submap, SubmapParams, submaps_from_roman_map
from roman.align.object_registration import InsufficientAssociationsException, ObjectRegistration
from roman.align.dist_reg_with_pruning import GravityConstraintError
from roman.utils import object_list_bounds, transform_rm_roll_pitch, expandvars_recursive
from roman.params.submap_align_params import SubmapAlignParams, SubmapAlignInputOutput
from roman.params.system_params import SystemParams
from roman.align.results import save_submap_align_results, SubmapAlignResults
from roman.map.map import load_roman_map, ROMANMap
from roman.object.segment import Segment
from roman.scene_graph.graph_node import GraphNode
from roman.scene_graph.scene_graph_3D import SceneGraph3D
from roman.scene_graph.scene_graph_3D_submap import SceneGraph3DSubmap
from .roman_registration import ROMANRegistration

import rerun as rr
import rerun.blueprint as rrb

def build_frustum_mesh(pose_camera_gt: np.ndarray, cam: CameraParams, max_range=50.0):
    """
    Build a trimesh mesh representing the camera frustum.
    """
    # FOV in radians
    hfov = 2 * np.arctan2(cam.width, (2 * cam.fx))
    vfov = 2 * np.arctan2(cam.height, (2 * cam.fy))

    # Extract the camera axes
    R = pose_camera_gt[:3, :3]
    forward = R[:, 2]  # camera z-axis
    right   = R[:, 0]  # camera x-axis
    up      = R[:, 1]  # camera y-axis

    # Extract the position
    pos = pose_camera_gt[:3, 3]

    # Far plane dimensions
    half_width = max_range * np.tan(hfov / 2)
    half_height = max_range * np.tan(vfov / 2)
    fc = pos + forward * max_range

    # 8 vertices of the frustum (4 at camera, 4 at far plane)
    near = 0.01  # near plane distance
    half_width_near = near * np.tan(hfov / 2)
    half_height_near = near * np.tan(vfov / 2)
    nc = pos + forward * near  # near plane center

    vertices = np.array([
        # Near plane corners
        nc + up * half_height_near + right * half_width_near,
        nc + up * half_height_near - right * half_width_near,
        nc - up * half_height_near + right * half_width_near,
        nc - up * half_height_near - right * half_width_near,
        
        # Far plane corners
        fc + up * half_height + right * half_width,
        fc + up * half_height - right * half_width,
        fc - up * half_height + right * half_width,
        fc - up * half_height - right * half_width,
    ])

    # Define faces as triangles (camera to far plane)
    faces = np.array([[0,1,2],[1,3,2], # Near Plane
                      [4,5,6],[5,7,6], # Far Plane
                      [0,1,5],[0,5,4], # In-between
                      [1,3,7],[1,7,5],
                      [3,2,6],[3,6,7],
                      [2,0,4],[2,4,6]])

    # Create convex hull mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh = mesh.convex_hull  # ensure convex
    return mesh

def fov_of_submaps_overlap(submap_a: Submap, submap_b: Submap, cam_params: CameraParams, _T_camera_flu: np.ndarray, max_range=50.0) -> bool:
    """ Check if camera frustums overlap using trimesh collision. """
    if submap_a.pose_flu_gt is None or submap_b.pose_flu_gt is None:
        raise ValueError("Both submaps must have pose_flu_gt defined")

    a_pose_camera_gt: np.ndarray = submap_a.pose_flu_gt @ np.linalg.inv(_T_camera_flu)
    b_pose_camera_gt: np.ndarray  = submap_b.pose_flu_gt @ np.linalg.inv(_T_camera_flu)

    mesh_a: trimesh.Trimesh = build_frustum_mesh(a_pose_camera_gt, cam_params, max_range)
    mesh_b: trimesh.Trimesh = build_frustum_mesh(b_pose_camera_gt, cam_params, max_range)

    # Check intersection
    cm = CollisionManager()
    cm.add_object('mesh_a', mesh_a)
    cm.add_object('mesh_b', mesh_b)
    return cm.in_collision_internal()

def submap_align(system_params: SystemParams, sm_params: SubmapAlignParams, sm_io: SubmapAlignInputOutput, rerun_viewer: RerunWrapperWindowAlignment) -> SubmapAlignResults:
    """
    Breaks maps into submaps and attempts to align each submap from one map with each submap from the second map.

    Args:
        sm_params (SubmapAlignParams): Aignment (loop closure) params.
        sm_io (SubmapAlignInputOutput): Input/output specifications.
    """

    # TODO: Might need to drop mindist restrictions so small nearby objects in our meronomy can still be properly associated.
    # TODO: Maybe use node average size instead of longest size?

    # Load image data so we have camera parameters
    img_data = system_params.data_params.load_img_data()
    
    # Load maps and split into Submaps
    submap_params = SubmapParams.from_submap_align_params(sm_params)
    submap_params.use_minimal_data = True
    if system_params.use_roman_map_for_alignment:
        maps: list[ROMANMap] = [load_roman_map(sm_io.inputs[i]) for i in range(2)]
        submaps: list[list[Submap]] = [submaps_from_roman_map(
            maps[i], submap_params, sm_io.gt_pose_data[i]) for i in range(2)]
    else:
        maps: list[SceneGraph3D] = [SceneGraph3D.load_map_from_pickle(sm_io.inputs[i]) for i in range(2)]
        submaps: list[list[SceneGraph3DSubmap]] = [SceneGraph3DSubmap.generate_submaps(maps[i],
                                                submap_params, sm_io.gt_pose_data[i]) for i in range(2)]
    print("Total Number of Submaps-  ROBOT1: ", len(submaps[0]), " ROBOT2: ", len(submaps[1]))

    # Registration setup
    clipper_angle_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_dist_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_num_associations = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    robots_nearby_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan  # Tracks the distance between robots for these submaps
    clipper_percent_associations = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    submap_yaw_diff_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    timing_list = []
    
    T_ij_mat = np.zeros((len(submaps[0]), len(submaps[1]), 4, 4))*np.nan
    T_ij_hat_mat = np.zeros((len(submaps[0]), len(submaps[1]), 4, 4))*np.nan
    associated_objs_mat = [[[] for _ in range(len(submaps[1]))] for _ in range(len(submaps[0]))] # cannot be numpy array since each element is a different sized array

    # Additional tracking variables for use in MeronomyGraph paper
    aligned_submaps_by_degree: np.ndarray = np.zeros((3), dtype=np.int64) # 0 is 0-60, 1 is 60-120, and 2 is 120-180
    alignment_success_num_by_degree: np.ndarray = np.zeros((3), dtype=np.int64) # Same as above
    overlapping_fov_mat = np.zeros((len(submaps[0]), len(submaps[1])), dtype=bool)*np.nan

    # Registration method
    registration: ObjectRegistration = sm_params.get_object_registration()

    # iterate over pairs of submaps and create registration results
    for i in tqdm(range(len(submaps[0]))):
        for j in (range(len(submaps[1]))):

            # Visualize the submaps (with no associations)
            rerun_viewer.update_associations(submaps[0][i].segments, submaps[1][j].segments, np.zeros((0, 2)))
            
            # Calculate distances between robots when at the respective submaps (using GT if available)
            if submaps[0][i].has_gt and submaps[1][j].has_gt:
                submap_distance = norm(submaps[0][i].position_gt - submaps[1][j].position_gt)
            else:
                submap_distance = norm(submaps[0][i].position - submaps[1][j].position)

            # If there is any potential overlap between submaps, track their distances for visualization later
            if submap_distance < sm_params.submap_radius*2:
                robots_nearby_mat[i, j] = submap_distance

            # Make a deep copy of the submaps
            submap_i = deepcopy(submaps[0][i])
            submap_j = deepcopy(submaps[1][j])

            # If single_robot_lc, then delete segments from submaps that are shared between them?
            # I think single_robot_lc means that it will try to avoid doing loop closures at all for a single robot with itself.
            # TODO: Ask advisors; why would they do this?
            if sm_params.single_robot_lc:

                if system_params.use_roman_map_for_alignment:
                    ids_i = set([seg.id for seg in submap_i.segments])
                    ids_j = set([seg.id for seg in submap_j.segments])
                    common_ids = ids_i.intersection(ids_j)
                    for sm in [submap_i, submap_j]:
                        to_rm = [seg for seg in sm.segments if seg.id in common_ids]
                        for seg in to_rm:
                            sm.segments.remove(seg)
                else:
                    ids_i = set([node.get_id() for node in submap_i.inactive_nodes])
                    ids_j = set([node.get_id() for node in submap_j.inactive_nodes])
                    common_ids = ids_i.intersection(ids_j)
                    for submap in [submap_i, submap_j]:
                        to_rm: list[GraphNode] = [node for node in submap.inactive_nodes if node.get_id() in common_ids]
                        for node in to_rm:
                            submap.inactive_nodes.remove(node)

            # Extract poses of robots with respect to the world (removing roll & pitch), using GT if available
            if sm_io.gt_pose_data[0] is not None:
                H_i_wrt_w = submaps[0][i].pose_gravity_aligned_gt
            else:
                H_i_wrt_w = submaps[0][i].pose_gravity_aligned

            if sm_io.gt_pose_data[1] is not None:
                H_j_wrt_w = submaps[1][j].pose_gravity_aligned_gt
            else:
                H_j_wrt_w = submaps[1][j].pose_gravity_aligned

            # Calculate the Pose of robot j with respect to robot i
            H_j_wrt_i = np.linalg.inv(H_i_wrt_w) @ H_j_wrt_w

            # If there is overlap between these submaps...
            if not np.isnan(robots_nearby_mat[i, j]):

                # Get the absolute difference in yaw and record for visualization later
                relative_yaw_angle = transform_to_xyzrpy(H_j_wrt_i)[5]
                submap_yaw_diff_mat[i, j] = np.abs(np.rad2deg(relative_yaw_angle))
                
            # Attempt to register the submaps
            try:   
                # Call the registration routine
                start_t = time.time()
                if not system_params.use_roman_map_for_alignment:
                    raise NotImplementedError
                associations: np.ndarray = registration.register(submap_i.segments, submap_j.segments)
                timing_list.append(time.time() - start_t)
                
                if sm_params.dim == 2:
                    T_ij_hat = registration.T_align(submap_i.segments, submap_j.segments, associations)
                    T_error = np.linalg.inv(T_ij_hat) @ H_j_wrt_i
                    _, _, theta = transform_to_xytheta(T_error)
                    dist = np.linalg.norm(T_error[:sm_params.dim, 3])

                elif sm_params.dim == 3:
                    T_ij_hat = registration.T_align(submap_i.segments, submap_j.segments, associations)
                    if sm_params.force_rm_upside_down:
                        xyzrpy = transform_to_xyzrpy(T_ij_hat)
                        if np.abs(xyzrpy[3]) > np.deg2rad(90.) or np.abs(xyzrpy[4]) > np.deg2rad(90.):
                            raise GravityConstraintError
                    if sm_params.force_rm_lc_roll_pitch:
                        T_ij_hat = transform_rm_roll_pitch(T_ij_hat)
                    T_error = np.linalg.inv(T_ij_hat) @ H_j_wrt_i
                    theta = Rot.from_matrix(T_error[:3, :3]).magnitude()
                    dist = np.linalg.norm(T_error[:sm_params.dim, 3])
                else:
                    raise ValueError("Invalid dimension")
                
            except (InsufficientAssociationsException, GravityConstraintError) as ex:
                timing_list.append(time.time() - start_t)
                T_ij_hat = np.zeros((4, 4))*np.nan
                theta = 180.0
                dist = 1e6
                associations = []
            
            if not np.isnan(robots_nearby_mat[i, j]):
                clipper_angle_mat[i, j] = np.abs(np.rad2deg(theta))
                clipper_dist_mat[i, j] = dist
            else:
                clipper_angle_mat[i, j] = np.nan
                clipper_dist_mat[i, j] = np.nan

            clipper_num_associations[i, j] = len(associations)
            avg_num_objs = np.mean([len(submap_i), len(submap_j)])
            if avg_num_objs > 0: clipper_percent_associations[i, j] = len(associations) / avg_num_objs
            else: clipper_percent_associations[i, j] = 0
            
            T_ij_mat[i, j] = H_j_wrt_i
            T_ij_hat_mat[i, j] = T_ij_hat
            associated_objs_mat[i][j] = associations
            overlapping_fov_mat[i, j] = fov_of_submaps_overlap(submap_i, submap_j, img_data.camera_params, system_params.data_params.pose_data_params.T_camera_flu)

            # Visualize the associations
            rerun_viewer.update_associations(submaps[0][i].segments, submaps[1][j].segments, associations)

            # Track metrics to match ROMAN paper
            if submap_distance <= 10 and overlapping_fov_mat[i, j]:

                # Calculate heading difference
                heading_diff = Rot.from_matrix(H_j_wrt_i[:3, :3]).magnitude() # radians

                # Determine if there is alignment success
                alignment_success: bool = False
                if clipper_num_associations[i, j] >= sm_io.lc_association_thresh and dist < 1 and theta < np.deg2rad(5):
                    alignment_success = True

                # Track the differences and rates of success
                if heading_diff <= np.deg2rad(60):
                    aligned_submaps_by_degree[0] += 1
                    alignment_success_num_by_degree[0] += int(alignment_success)
                elif heading_diff <= np.deg2rad(120):
                    aligned_submaps_by_degree[1] += 1
                    alignment_success_num_by_degree[1] += int(alignment_success)
                else:
                    aligned_submaps_by_degree[2] += 1
                    alignment_success_num_by_degree[2] += int(alignment_success)

    # save results
    results = SubmapAlignResults(
        robots_nearby_mat=robots_nearby_mat,
        clipper_angle_mat=clipper_angle_mat,
        clipper_dist_mat=clipper_dist_mat,
        clipper_num_associations=clipper_num_associations,
        submap_yaw_diff_mat=submap_yaw_diff_mat,
        T_ij_mat=T_ij_mat,
        T_ij_hat_mat=T_ij_hat_mat,
        associated_objs_mat=associated_objs_mat,
        timing_list=timing_list,
        submap_align_params=sm_params,
        submap_io=sm_io,
        overlapping_fov_mat=overlapping_fov_mat,
        aligned_submaps_by_degree=aligned_submaps_by_degree,
        alignment_success_num_by_degree=alignment_success_num_by_degree
    )
    save_submap_align_results(results, submaps, maps)
    return results