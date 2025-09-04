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
    print(base_color)
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
    # axs.set_xlim([-250, 250])
    # axs.set_ylim([-250, 250])
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

        from evo.tools import plot

        # If number of robots is two, draw plot with colors per robot
        num_robots = len(list(run_names.keys()))

        # Get PoseData objects not merged between robots
        list_est, list_gt = gt_csv_est_g2o_to_pose_data(est_g2o_file, est_time_file, gt_files, run_names, run_env, skip_final_merge=True)

        # Extract evo PoseTrajectory3Ds that corresond to each robot
        time_offsets = []
        for i in range(num_robots):
            if i == 0:
                time_offsets.append(0)
            else:
                time_offsets.append(list_est[i-1].tf - list_est[i].t0 + 1.0) # TODO: Why +1

        # Make deep copies of trajectories
        traj_est_aligned_copies = [copy.deepcopy(traj_est_aligned) for x in range(num_robots)]
        traj_gt_copies = [copy.deepcopy(traj_ref) for x in range(num_robots)]

        # Reduce trajectories to the specific time that covers each robot
        # TODO: THIS TIME OFFSET CODE IS COMPLETELY BUSTED.
        print("WARNING: This code is not fully tested yet!")
        for i, traj in enumerate(traj_est_aligned_copies):
            print(f"Robot {i}: ", list_est[i].t0 + np.sum(time_offsets[0:i+1]), list_est[i].tf + np.sum(time_offsets[0:i+1]))
            traj.reduce_to_time_range(list_est[i].t0 + np.sum(time_offsets[0:i+1]), list_est[i].tf + np.sum(time_offsets[0:i+1]))
        for i, traj in enumerate(traj_gt_copies):
            traj.reduce_to_time_range(list_est[i].t0 + np.sum(time_offsets[0:i+1]), list_est[i].tf + np.sum(time_offsets[0:i+1]))

        # Draw a secondary plot where the trajectories from different robots have slightly different color
        robot_colors = ["#11EE72", "#7211EE", "#38C2C7", "#2F3FD0"]
        draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_copies, traj_gt_copies, run_names, f"{output_dir}/offline_rpgo/aligned_gt_est_per_robot.png",
            no_background=True, linewidth=2.0, robot_colors=robot_colors, aspect=1)
        robot_colors = ["#11EE72", "#7211EE", "#38C2C7", "#2F3FD0"]
        draw_plot_with_robot_trajectories_different_colors(traj_est_aligned_copies, traj_gt_copies, run_names, f"{output_dir}/offline_rpgo/aligned_gt_est_per_robot_noBackground.png",
            no_background=True, linewidth=2.0, robot_colors=robot_colors, plot_gt=False)
        
        # for i in range(num_robots):
        #     _traj_est_aligned_copies = [traj_est_aligned_copies[i]]
        #     _traj_gt_copies = [traj_gt_copies[i]]
        #     _run_names = [run_names[i]]
        #     _robot_colors = [robot_colors[i]]

        #     robot_colors = ["#11EE72", "#7211EE", "#38C2C7", "#2F3FD0"]

        #     draw_plot_with_robot_trajectories_different_colors(_traj_est_aligned_copies, _traj_gt_copies, _run_names, f"{output_dir}/offline_rpgo/{i}_aligned_gt_est_per_robot.png",
        #         no_background=True, linewidth=2.0, robot_colors=_robot_colors, aspect=1)
        #     draw_plot_with_robot_trajectories_different_colors(_traj_est_aligned_copies, _traj_gt_copies, _run_names, f"{output_dir}/offline_rpgo/{i}_aligned_gt_est_per_robot_noBackground.png",
        #         no_background=True, linewidth=2.0, robot_colors=_robot_colors, plot_gt=False)

    # Calculate the Absolute Pose Error
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    return ape_stat