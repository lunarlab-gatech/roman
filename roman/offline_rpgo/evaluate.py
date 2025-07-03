import colorsys
import copy
from evo.core import metrics
from evo.core import sync
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from roman.offline_rpgo.g2o_and_time_to_pose_data import gt_csv_est_g2o_to_pose_data
import seaborn as sns
from typing import Dict

def make_lightness_palette(base_color, n_colors=20, light_range=(0.3, 0.8)):
    # Convert base color to HLS
    rgb = mcolors.to_rgb(base_color)
    h, _, s = colorsys.rgb_to_hls(*rgb)

    # Generate colors with varying lightness
    lightnesses = np.linspace(light_range[0], light_range[1], n_colors)
    palette = [colorsys.hls_to_rgb(h, li, s) for li in lightnesses]
    return palette

def draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_robot0,
        traj_gt_robot0, traj_est_aligned_robot1, traj_gt_robot1, run_names, file_path,
        no_background=False, robot0color="#B3A369", robot1color="#003057",
        linewidth=1.0, plot_gt=True, aspect='equal'):
        
    # Draw a secondary plot where the trajectories from different robots have slightly different color
    fig, axs = plt.subplots(1, 1)
    if no_background:
        fig.patch.set_facecolor('white')
        axs.set_facecolor('none')
    else:
        fig.patch.set_facecolor('white')
        axs.set_facecolor("#F0F0F0")
    robot0_name = str(list(run_names.values())[0])
    robot1_name = str(list(run_names.values())[1])
    
    # Setup custom color palette based on GT colors
    gold_palette = make_lightness_palette(robot0color)
    blue_palette = make_lightness_palette(robot1color)

    axs.plot(traj_est_aligned_robot0.positions_xyz[:,0], traj_est_aligned_robot0.positions_xyz[:,1], 
                label=robot0_name + " (Est.)", color=gold_palette[10], linewidth=linewidth)
    if plot_gt:
        axs.plot(traj_gt_robot0.positions_xyz[:,0], traj_gt_robot0.positions_xyz[:,1], 
                    label=robot0_name + " (GT)", color=gold_palette[4], linewidth=linewidth,
                    linestyle="dotted")
    axs.plot(traj_est_aligned_robot1.positions_xyz[:,0], traj_est_aligned_robot1.positions_xyz[:,1], 
                label=robot1_name + " (Est.)", color=blue_palette[8], linewidth=linewidth)
    if plot_gt:
        axs.plot(traj_gt_robot1.positions_xyz[:,0], traj_gt_robot1.positions_xyz[:,1], 
                    label=robot1_name + " (GT)", color=blue_palette[2], linewidth=linewidth,
                    linestyle="dotted")    
    axs.set_aspect(aspect, adjustable='box')

    for spine in axs.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    axs.set_title("Ground Truth vs. ROMAN Estimated Trajectories (Aligned)")
    axs.set_xlabel("X (meters)")
    axs.set_ylabel("Y (meters)")
    axs.legend()
    plt.savefig(file_path, dpi=300)

def evaluate(est_g2o_file: str, est_time_file: str, gt_files: Dict[int, str], 
             run_names: Dict[int, str] = None, run_env: str = None, output_dir: str = None):
    pd_est, pd_gt = gt_csv_est_g2o_to_pose_data(
        est_g2o_file, est_time_file, gt_files, run_names, run_env)
        
    traj_ref = pd_gt.to_evo()
    traj_est = pd_est.to_evo()

    max_diff = 0.1

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

    pose_relation = metrics.PoseRelation.translation_part
    use_aligned_trajectories = True
    
    if use_aligned_trajectories:
        data = (traj_ref, traj_est_aligned) 
    else:
        data = (traj_ref, traj_est)
    
    if output_dir is not None:
        # evo/pyqt/opencv do not play well together - 
        # only make this plot of everything is importing okay
        try:
            from evo.tools import plot
            
            fig = plt.figure()
            traj_by_label = {
                "estimate (aligned)": traj_est_aligned,
                "reference": traj_ref
            }
            plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            plt.savefig(f"{output_dir}/offline_rpgo/aligned_gt_est.png")
        except:
            print("WARNING: loading evo plotting failed, likely due to qt issues.")

        try:
            from evo.tools import plot

            # If number of robots is two, draw plot with colors per robot
            if len(list(run_names.keys())) == 2:

                # Get PoseData objects not merged between robots
                list_est, list_gt = gt_csv_est_g2o_to_pose_data(est_g2o_file, est_time_file, gt_files, run_names, run_env, skip_final_merge=True)

                # Extract evo PoseTrajectory3Ds that corresond to each robot
                time_offset_robot1 = list_est[0].tf - list_est[1].t0 + 1.0

                traj_est_aligned_robot0 = copy.deepcopy(traj_est_aligned)
                traj_est_aligned_robot1 = copy.deepcopy(traj_est_aligned)
                traj_gt_robot0 = copy.deepcopy(traj_ref)
                traj_gt_robot1 = copy.deepcopy(traj_ref)

                traj_est_aligned_robot0.reduce_to_time_range(list_est[0].t0, list_est[0].tf)
                traj_est_aligned_robot1.reduce_to_time_range(list_est[1].t0 + time_offset_robot1, list_est[1].tf + time_offset_robot1)
                traj_gt_robot0.reduce_to_time_range(list_est[0].t0, list_est[0].tf)
                traj_gt_robot1.reduce_to_time_range(list_est[1].t0 + time_offset_robot1, list_est[1].tf + time_offset_robot1)

                # Draw a secondary plot where the trajectories from different robots have slightly different color
                draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_robot0, traj_gt_robot0,
                    traj_est_aligned_robot1, traj_gt_robot1, run_names, f"{output_dir}/offline_rpgo/aligned_gt_est_per_robot.png",
                    no_background=True, linewidth=4.0, robot0color="#11EE72", robot1color="#7211EE", aspect=1)
                draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_robot0, traj_gt_robot0, 
                    traj_est_aligned_robot1, traj_gt_robot1, run_names, f"{output_dir}/offline_rpgo/aligned_gt_est_per_robot_noBackground.png",
                    no_background=True, linewidth=4.0, robot0color="#11EE72", robot1color="#7211EE", plot_gt=False)
            else:
                print("Skipping plot of trajectories w/color per robot since not implemented for 3+ robots yet.")
        except Exception as e:
            print("Exception: ", e)

    # Calculate the Absolute Pose Error
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    return ape_stat