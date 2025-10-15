import colorsys
import copy
from evo.core import metrics
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from robotdatapy.data.pose_data import PoseData
from roman.offline_rpgo.g2o_and_time_to_pose_data import gt_csv_est_g2o_to_pose_data, make_start_and_end_times_match
import seaborn as sns
from typing import Dict


def make_lightness_palette(base_color, n_colors=20, light_range=(0.3, 0.8)) -> list[tuple[float, float, float]]:
    # Convert base color to HLS
    rgb = mcolors.to_rgb(base_color)
    h, _, s = colorsys.rgb_to_hls(*rgb)

    # Generate colors with varying lightness
    lightnesses = np.linspace(light_range[0], light_range[1], n_colors)
    palette = [colorsys.hls_to_rgb(h, li, s) for li in lightnesses]
    return palette

def draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_copies, traj_gt_copies,
        run_names, file_path, no_background=False, robot_colors=["#B3A369", "#003057"],
        linewidth=1.0, plot_gt=True, aspect='equal'):
        
    # Draw a secondary plot where the trajectories from different robots have slightly different color
    fig, axs = plt.subplots(1, 1)
    if no_background:
        fig.patch.set_facecolor('white')
        axs.set_facecolor('none')
    else:
        fig.patch.set_facecolor('white')
        axs.set_facecolor("#F0F0F0")

    # Get the robot names
    robot_names = [str(name) for name in list(run_names.values())]

    # Setup custom color palette based on provided colors
    for i, color in enumerate(robot_colors):
        robot_colors[i] = make_lightness_palette(color)

    # Plot the trajectories
    for i in range(len(robot_names)):
        traj_est = traj_est_aligned_copies[i]
        traj_gt = traj_gt_copies[i]

        axs.plot(traj_est.positions_xyz[:,0], traj_est.positions_xyz[:,1], 
                    label=robot_names[i] + " (Est.)", color=robot_colors[i][9], linewidth=linewidth)
        if plot_gt:
            axs.plot(traj_gt.positions_xyz[:,0], traj_gt.positions_xyz[:,1], 
                        label=robot_names[i] + " (GT)", color=robot_colors[i][3], linewidth=linewidth,
                        linestyle="dotted")
    axs.set_aspect(aspect, adjustable='box')

    # Adjust the spines
    for spine in axs.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)

    # Add labels
    axs.set_title("Ground Truth vs. ROMAN Estimated Trajectories (Aligned)")
    axs.set_xlabel("X (meters)")
    axs.set_ylabel("Y (meters)")
    axs.legend()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

def evaluate(est_g2o_file: str, est_time_file: str, gt_data: Dict[int, PoseData], 
             run_names: Dict[int, str] = None, run_env: str = None, output_dir: str = None):
    pd_est, pd_gt = gt_csv_est_g2o_to_pose_data(
        est_g2o_file, est_time_file, gt_data, run_names, run_env)
        
    traj_ref: PoseTrajectory3D = pd_gt.to_evo()
    traj_est: PoseTrajectory3D = pd_est.to_evo()

    max_diff = 0.1

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

    traj_est_aligned: PoseTrajectory3D = copy.deepcopy(traj_est)
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
            plt.close(fig)
        except:
            print("WARNING: loading evo plotting failed, likely due to qt issues.")

        from evo.tools import plot

        # Get PoseData objects not merged between robots (but with updated times)
        list_est, list_gt = gt_csv_est_g2o_to_pose_data(est_g2o_file, est_time_file, gt_data, run_names, run_env, skip_final_merge=True)
        list_est, list_gt = make_start_and_end_times_match(list_est, list_gt)

        def get_aligned_trajectories_specific_to_each_robot(list_pd: list[PoseData], aligned_traj: PoseTrajectory3D) -> list[PoseTrajectory3D]:
            # Calculate the Start time for the beginning of each robots data
            start_times: list[float] = []
            end_times: list[float] = []
            num_robots = len(list(run_names.keys()))
            for i in range(num_robots): 
                pd: PoseData = list_pd[i]
                if i == 0:
                    start_times.append(pd.t0)
                    end_times.append(pd.tf)
                else:
                    start_times.append(end_times[-1] + 1)
                    end_times.append(pd.tf - pd.t0 + end_times[-1] + 1)
            
            # Make deep copies of trajectories
            aligned_traj_copies: list[PoseTrajectory3D] = [copy.deepcopy(aligned_traj) for x in range(len(list_pd))]

            # Reduce trajectories to the specific time that covers each robot
            for i, traj in enumerate(aligned_traj_copies):
                traj.reduce_to_time_range(start_times[i], end_times[i])
            return aligned_traj_copies

        # Get aligned trajectories for each robot for estimated and GT
        traj_est_aligned_copies = get_aligned_trajectories_specific_to_each_robot(list_est, traj_est_aligned)
        traj_gt_copies = get_aligned_trajectories_specific_to_each_robot(list_gt, traj_ref)

        # Draw a secondary plot where the trajectories from different robots have slightly different color
        robot_colors = ["#11EE72", "#7211EE", "#38C2C7", "#2F3FD0"]
        draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_copies, traj_gt_copies, run_names, f"{output_dir}/offline_rpgo/aligned_gt_est_per_robot.png",
            no_background=True, linewidth=2.0, robot_colors=robot_colors, aspect=1)
        
        robot_colors = ["#11EE72", "#7211EE", "#38C2C7", "#2F3FD0"]
        draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_copies, traj_gt_copies, run_names, f"{output_dir}/offline_rpgo/aligned_gt_est_per_robot_noBackground.png",
            no_background=True, linewidth=2.0, robot_colors=robot_colors, plot_gt=False)
        
    # Calculate the Absolute Pose Error
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    return ape_stat