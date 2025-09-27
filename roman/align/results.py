import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass
from typing import List
import json
import numpy as np

from robotdatapy.transform import transform_to_xytheta, transform_to_xyz_quat, \
    transform_to_xyzrpy
from robotdatapy.data.pose_data import PoseData
from roman.utils import transform_rm_roll_pitch
from roman.map.map import ROMANMap, SubmapParams
from roman.params.submap_align_params import SubmapAlignInputOutput, SubmapAlignParams
from roman.object.segment import Segment
from scipy.spatial.transform import Rotation

@dataclass
class SubmapAlignResults:
    robots_nearby_mat: np.array
    clipper_angle_mat: np.array
    clipper_dist_mat: np.array
    clipper_num_associations: np.array
    submap_yaw_diff_mat: np.array
    associated_objs_mat: np.array
    T_ij_mat: np.array
    T_ij_hat_mat: np.array
    timing_list: List[float]
    submap_align_params: SubmapAlignParams
    submap_io: SubmapAlignInputOutput
    
    def save(self):
        pkl_file = open(self.submap_io.output_pkl, 'wb')
        pickle.dump(self, pkl_file)
        pkl_file.close()
        
    @classmethod
    def load(self, file_path):
        pkl_file = open(file_path, 'rb')
        return pickle.load(pkl_file)
        

def time_to_secs_nsecs(t, as_dict=False):
    seconds = int(t)
    nanoseconds = int((t - int(t)) * 1e9)
    if not as_dict:
        return seconds, nanoseconds
    else:
        return {'seconds': seconds, 'nanoseconds': nanoseconds}

def plot_align_results(results: SubmapAlignResults, dpi=500):
    # TODO: Update this to solve issues with overlapping portions, maybe seperate figures?

    # Create plots
    fig, ax = plt.subplots(1, 5, figsize=(20, 5), dpi=dpi)
    fig.subplots_adjust(wspace=.3)
    fig.suptitle(results.submap_io.run_name)

    mp = ax[0].imshow(results.robots_nearby_mat, cmap='viridis', vmin=0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[0].set_title("Submaps Center Distance (m)")

    mp = ax[1].imshow(-results.clipper_angle_mat, cmap='viridis', vmax=0, vmin=-10)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[1].set_title("Registration Error (deg)")

    mp = ax[2].imshow(-results.clipper_dist_mat, cmap='viridis', vmax=0, vmin=-5.0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[2].set_title("Registration Distance Error (m)")

    mp = ax[3].imshow(results.clipper_num_associations, cmap='viridis', vmin=0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[3].set_title("Number of CLIPPER Associations")

    mp = ax[4].imshow(results.submap_yaw_diff_mat, cmap='viridis', vmin=0)
    fig.colorbar(mp, fraction=0.04, pad=0.04)
    ax[4].set_title("Submap Yaw Difference (deg)")

    for i in range(len(ax)):
        ax[i].set_xlabel("submap index (robot 2)")
        ax[i].set_ylabel("submap index (robot 1)")
        ax[i].grid(False)

def save_submap_align_results(results: SubmapAlignResults, submaps, roman_maps: List[ROMANMap]):
    plot_align_results(results)

    plt.savefig(results.submap_io.output_img)
        
    # for saving matrix results instead of image
    pkl_file = open(results.submap_io.output_matrix, 'wb')
    pickle.dump([results.robots_nearby_mat, results.clipper_angle_mat, results.clipper_dist_mat, 
                 results.clipper_num_associations, results.submap_yaw_diff_mat], pkl_file)
    pkl_file.close()
        
    # stores the submaps, associated objects, ground truth object overlap, and ground truth and estimated submap transformations
    # TODO: save non-minimal data representation of segments
    pkl_file = open(results.submap_io.output_pkl, 'wb')
    pickle.dump(results, pkl_file)
    pkl_file.close()

    with open(results.submap_io.output_timing, 'w') as f:
        f.write(f"Total number of submaps: {len(submaps[0])} x {len(submaps[1])} = {len(submaps[0])*len(submaps[1])}\n")
        f.write(f"Average time per registration: {np.mean(results.timing_list):.4f} seconds\n")
        f.write(f"Total time: {np.sum(results.timing_list):.4f} seconds\n")
        f.write(f"Total number of objects: {np.sum([len(submap) for submap in submaps[0] + submaps[1]])}\n")
        f.write(f"Average number of obects per map: {np.mean([len(submap) for submap in submaps[0] + submaps[1]]):.2f}\n")
    
    with open(results.submap_io.output_params, 'w') as f:
        f.write(f"{results.submap_align_params}")

    I_t = 1 / (results.submap_io.g2o_t_std**2)
    I_r = 1 / (results.submap_io.g2o_r_std**2)
    I = np.diag([I_t, I_t, I_t, I_r, I_r, I_r])
    
    json_output = []
    pose_data = [PoseData.from_times_and_poses(rm.times, rm.trajectory) for rm in roman_maps]

    with open(results.submap_io.output_g2o, 'w') as f:
        for i in range(len(submaps[0])):
            for j in range(len(submaps[1])):
                if results.clipper_num_associations[i, j] < results.submap_io.lc_association_thresh:
                    continue
                if (np.abs(submaps[0][i].time - submaps[1][j].time) < 
                    results.submap_align_params.single_robot_lc_time_thresh and results.submap_align_params.single_robot_lc):
                    continue
                T_ci_cj = results.T_ij_hat_mat[i, j] # transform from center_j to center_i
                T_odomi_ci = submaps[0][i].pose_gravity_aligned # center i in odom frame
                T_odomj_cj = submaps[1][j].pose_gravity_aligned # center i in odom frame
                T_odomi_pi = submaps[0][i].pose_flu # pose i in odom frame
                T_odomj_pj = submaps[1][j].pose_flu # pose j in odom frame
                T_pi_pj = ( # pose j in pose i frame, the desired format for our loop closure
                    np.linalg.inv(T_odomi_pi) @ T_odomi_ci @ T_ci_cj @ np.linalg.inv(T_odomj_cj) @ T_odomj_pj
                )
                t, q = transform_to_xyz_quat(T_pi_pj, separate=True)
                json_output.append({
                    'seconds': [int(submaps[0][i].time), int(submaps[1][j].time)],
                    'nanoseconds': [int((submaps[0][i].time % 1) * 1e9), int((submaps[1][j].time % 1) * 1e9)],
                    'names': results.submap_io.robot_names,
                    'translation': t.tolist(),
                    'rotation': q.tolist(),
                    'rotation_convention': 'xyzw',
                })

                idx_a = pose_data[0].idx(submaps[0][i].time, force_single=True)
                idx_b = pose_data[1].idx(submaps[1][j].time, force_single=True)
                f.write(f"# LC: {int(results.clipper_num_associations[i, j])}\n")
                f.write(f"EDGE_SE3:QUAT a{idx_a} b{idx_b} \t")
                f.write(f"{t[0]} {t[1]} {t[2]} \t")
                f.write(f"{q[0]} {q[1]} {q[2]} {q[3]} \t")
                for ii in range(6):
                    for jj in range(6):
                        if jj < ii:
                            continue
                        f.write(f"{I[ii, jj]} ")
                    f.write("\t")
                f.write("\n")
        f.close()
            
        with open(results.submap_io.output_lc_json, 'w') as f:
            json.dump(json_output, f, indent=4)
            f.close()
            
        for i, output_sm in enumerate(results.submap_io.output_submaps):
            roman_map = roman_maps[i]
            if output_sm is not None:
                with open(output_sm, 'w') as f:
                    sm_json = dict()
                    sm_json['segments'] = []
                    sm_json['submaps'] = []
                    
                    segment: Segment
                    for segment in roman_map.segments:
                        try:
                            segment_json = {}
                            segment_json['robot_name'] = results.submap_io.robot_names[i]
                            segment_json['segment_index'] = segment.id
                            segment_json['centroid_odom'] = np.mean(segment.points, axis=0).tolist()
                            e = segment.normalized_eigenvalues()
                            segment_json['shape_attributes'] = {'volume': segment.volume, 
                                                                'linearity': segment.linearity(e), 
                                                                'planarity': segment.planarity(e), 
                                                                'scattering': segment.scattering(e)}
                            segment_json['first_seen'] = time_to_secs_nsecs(segment.first_seen, as_dict=True)
                            segment_json['last_seen'] = time_to_secs_nsecs(segment.last_seen, as_dict=True)
                            sm_json['segments'].append(segment_json)
                        except:
                            continue
                        
                    for j in range(len(submaps[i])):
                        t_j = submaps[i][j].time
                        xyzquat_submap = transform_to_xyz_quat(submaps[i][j].pose_gravity_aligned, separate=False)
                        sm_json['submaps'].append({
                            'submap_index': j,
                            'T_odom_submap': {
                                'tx': xyzquat_submap[0],
                                'ty': xyzquat_submap[1],
                                'tz': xyzquat_submap[2],
                                'qx': xyzquat_submap[3],
                                'qy': xyzquat_submap[4],
                                'qz': xyzquat_submap[5],
                                'qw': xyzquat_submap[6],
                            },
                            'robot_name': results.submap_io.robot_names[i],
                            'seconds': int(t_j),
                            'nanoseconds': int((t_j % 1) * 1e9),
                            'segment_indices': [segment.id for segment in submaps[i][j].segments]
                        })
                    json.dump(sm_json, f, indent=4)
                    f.close()

def calculate_loop_closure_error(json_file: str, gt_pose0: PoseData, gt_pose1: PoseData) -> tuple[float, float, float, float]:

    def extract_T_R_from_transform(H: np.ndarray) -> tuple[np.ndarray, Rotation]:
        T = H[:3, 3] 
        R = Rotation.from_matrix(H[:3, :3])
        return T, R
    
    def relative_transform(t1: np.ndarray, r1: Rotation, t2: np.ndarray, r2: Rotation):
        r_rel = r1.inv() * r2
        t_rel = r1.inv().apply(t2 - t1)
        return t_rel, r_rel

    # Load the calculated loop closure data
    with open(json_file, 'r') as f:
        loops = json.load(f)

    # If there are no loops, return infinite error
    if len(loops) == 0:
        return np.inf, np.inf, np.inf, np.inf

    # Create lists to store errors
    trans_errors: list[float] = []
    rot_errors: list[float] = []

    # For each loop closure
    for loop in loops:
        # Calculate the time in both robots maps
        t0_sec, t1_sec = loop["seconds"]
        t0_ns, t1_ns = loop["nanoseconds"]
        t0 = t0_sec + t0_ns * 1e-9
        t1 = t1_sec + t1_ns * 1e-9

        # Get GT poses at those times
        H_gt0 = gt_pose0.T_WB(t0)
        H_gt1 = gt_pose1.T_WB(t1)
        t_gt0, r_gt0 = extract_T_R_from_transform(H_gt0)
        t_gt1, r_gt1 = extract_T_R_from_transform(H_gt1)

        # Compute GT relative transform
        gt_t_rel, gt_r_rel = relative_transform(t_gt0, r_gt0, t_gt1, r_gt1)

        print("GT Tranlsation: ", gt_t_rel)
        print("GT Rotation: ", gt_r_rel.as_matrix())

        # Predicted transform
        pred_t = np.array(loop["translation"])
        pred_r = Rotation.from_quat(loop["rotation"])

        print("Predicted Translation: ", pred_t)
        print("Predicted Rotation: ", pred_r.as_matrix())

        # Compute errors
        trans_error = np.linalg.norm(pred_t - gt_t_rel)
        r_diff = pred_r.inv() * gt_r_rel
        angle_error = r_diff.magnitude() * 180/np.pi
        trans_errors.append(trans_error)
        rot_errors.append(angle_error)

        print("Translation error:", trans_error)
        print("Rotation error (deg):", angle_error)

    mean_trans_error = np.mean(trans_errors)
    print(trans_error)
    std_trans_error = np.std(trans_error)
    mean_rot_error_deg = np.mean(rot_errors)
    std_rot_error_deg = np.std(rot_errors)
    print("Mean translation error:",  mean_trans_error)
    print("Standard Deviation translation error: ", std_trans_error)
    print("Mean rotation error (deg):", mean_rot_error_deg)
    print("Standard Deviation rotation error (deg):", std_rot_error_deg)
    return mean_trans_error, std_trans_error, mean_rot_error_deg, std_rot_error_deg