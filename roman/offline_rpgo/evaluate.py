import colorsys
import copy
from dataclasses import dataclass
from evo.core import metrics
from evo.core import sync
from evo.core.metrics import PathPair
from evo.core.trajectory import PoseTrajectory3D
import hashlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import numpy as np
from pathlib import Path
from robotdatapy.data.pose_data import PoseData
from roman.offline_rpgo.g2o_and_time_to_pose_data import gt_csv_est_g2o_to_pose_data, make_start_and_end_times_match
from roman.utils import expandvars_recursive
from typing import Dict


def make_lightness_palette(base_color, n_colors=20, light_range=(0.3, 0.8)) -> list[tuple[float, float, float]]:
    # Convert base color to HLS
    rgb = mcolors.to_rgb(base_color)
    h, _, s = colorsys.rgb_to_hls(*rgb)

    # Generate colors with varying lightness
    lightnesses = np.linspace(light_range[0], light_range[1], n_colors)
    palette = [colorsys.hls_to_rgb(h, li, s) for li in lightnesses]
    return palette

def draw_plot_with_robot_trajectories_different_colors(
        traj_est_aligned_copies: list[PoseTrajectory3D], 
        traj_gt_copies: list[PoseTrajectory3D], 
        run_names: dict, 
        file_path: str, 
        no_background: bool =False,
        linewidth: float = 1.0, 
        plot_gt: bool =True, 
        aspect: str = 'equal', 
        show_grid: bool = False,
        background_image_path: str | None = None,
        background_image_x_edge: np.ndarray | None = None,
        no_border: bool = False,
        legend: bool = True):
    
    # Define a dictionary that maps from robot_name to robot_color
    robot_name_to_color: dict = {
        "Husky1": "#5FF598",
        "Husky2": "#F5756E",
        "Drone1": "#F5DA62",
        "Drone2": "#6262F5",
    }

    # Helper function to map names to colors if one doesn't exist in dictionary
    def name_to_color(name: str) -> str:
        h = hashlib.sha1(name.encode("utf-8")).hexdigest()
        return f"#{h[:6]}"

    # Get the robot names
    robot_names = [run_names[i] for i in range(len(run_names))]

    # Get robot colors based on name
    robot_colors = []
    for robot_name in robot_names:
        if robot_name in robot_name_to_color:
            color = robot_name_to_color[robot_name]
        else:
            color = name_to_color(robot_name)
        robot_colors.append(make_lightness_palette(color))

    # Draw a plot where the trajectories from different robots have slightly different color
    fig, axs = plt.subplots(1, 1)
    if no_background:
        fig.patch.set_facecolor('white')
        axs.set_facecolor('none')
    else:
        fig.patch.set_facecolor('white')
        axs.set_facecolor("#F0F0F0")

    # Draw background image
    if background_image_path is not None:
        img = mpimg.imread(expandvars_recursive(background_image_path))
        if background_image_x_edge:
            x_extent_meters = background_image_x_edge / 100.0
            h, w = img.shape[0], img.shape[1]
            y_extent_meters = x_extent_meters / w * h
            extent = [-x_extent_meters, x_extent_meters, -y_extent_meters, y_extent_meters]
            axs.imshow(img, extent=extent, origin="upper", alpha=1.0, zorder=0)
        else:
            raise ValueError("Extent must be provided with Background image.")

    # Calculate trajectory bounds
    all_x = np.concatenate([traj.positions_xyz[:, 0] for traj in traj_est_aligned_copies])
    all_y = np.concatenate([traj.positions_xyz[:, 1] for traj in traj_est_aligned_copies])
    if plot_gt:
        all_x = np.concatenate([all_x] + [traj.positions_xyz[:, 0] for traj in traj_gt_copies])
        all_y = np.concatenate([all_y] + [traj.positions_xyz[:, 1] for traj in traj_gt_copies])\
        
    padding_x = (all_x.max() - all_x.min()) * 0.05
    padding_y = (all_y.max() - all_y.min()) * 0.05
    x_min, x_max = all_x.min() - padding_x, all_x.max() + padding_x
    y_min, y_max = all_y.min() - padding_y, all_y.max() + padding_y

    # Plot the trajectories
    for i in range(len(robot_names)):
        traj_est: PoseTrajectory3D = traj_est_aligned_copies[i]
        traj_gt: PoseTrajectory3D = traj_gt_copies[i]

        axs.plot(traj_est.positions_xyz[:,0], traj_est.positions_xyz[:,1], 
                    label=robot_names[i] + " (Est.)", color=robot_colors[i][9], linewidth=linewidth)
        if plot_gt:
            axs.plot(traj_gt.positions_xyz[:,0], traj_gt.positions_xyz[:,1], 
                        label=robot_names[i] + " (GT)", color=robot_colors[i][3], linewidth=linewidth,
                        linestyle="dotted")
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)
    axs.set_aspect(aspect, adjustable='box')
    
    # Make the tick spacing match
    yticks = axs.get_yticks()
    if len(yticks) > 1:
        y_spacing = yticks[1] - yticks[0]
        axs.xaxis.set_major_locator(MultipleLocator(y_spacing))

    # Adjust the spines
    for spine in axs.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)

    # Add Grid if Desired
    if show_grid:
        axs.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    else:
        axs.grid(False)

    # Add labels
    axs.set_xlabel("X (meters)")
    axs.set_ylabel("Y (meters)")

    if legend:
        axs.legend()
    if no_border:
        axs.set_axis_off()

    plt.savefig(file_path, dpi=300)
    plt.close(fig)

def evaluate(est_g2o_file: str, est_time_file: str, gt_data: Dict[int, PoseData], 
             run_names: Dict[int, str] = None, run_env: str = None, output_dir: str = None,
             background_image_path: Path | None = None,
             background_image_x_edge: float | None = None) -> dict:
    
    pd_est, pd_gt = gt_csv_est_g2o_to_pose_data(
        est_g2o_file, est_time_file, gt_data, run_names, run_env)
        
    traj_ref: PoseTrajectory3D = pd_gt.to_evo()
    traj_est: PoseTrajectory3D = pd_est.to_evo()

    max_diff = 0.1

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

    traj_est_aligned: PoseTrajectory3D = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

    data: PathPair = (traj_ref, traj_est_aligned) 
    
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


        # Draw Plots
        draw_plot_with_robot_trajectories_different_colors(
            traj_est_aligned_copies,
            traj_gt_copies, 
            run_names, 
            f"{output_dir}/offline_rpgo/plot_white.png", 
            no_background=True, 
            linewidth=2.0, 
            aspect=1)
        
        draw_plot_with_robot_trajectories_different_colors(
            traj_est_aligned_copies,
            traj_gt_copies, 
            run_names, 
            f"{output_dir}/offline_rpgo/plot_grid.png", 
            no_background=True, 
            linewidth=2.0, 
            aspect=1,
            show_grid=True)
        
        draw_plot_with_robot_trajectories_different_colors(
            traj_est_aligned_copies, 
            traj_gt_copies, 
            run_names, 
            f"{output_dir}/offline_rpgo/plot_overlaid.png", 
            no_background=True, 
            linewidth=2.0, 
            no_border=True,
            plot_gt=False, 
            background_image_path=background_image_path, 
            background_image_x_edge=background_image_x_edge)
        
    # Calculate various error metrics using evo, including APE and RPE
    all_pose_relations: list[metrics.PoseRelation] = [metrics.PoseRelation.full_transformation, # dimensionless
                                                      metrics.PoseRelation.translation_part, # meters
                                                      metrics.PoseRelation.rotation_part, # dimensionless
                                                      metrics.PoseRelation.rotation_angle_deg, # degrees
                                                      metrics.PoseRelation.rotation_angle_rad, # radians
                                                      metrics.PoseRelation.point_distance, # meters
                                                      metrics.PoseRelation.point_distance_error_ratio] # percent
    all_statistic_types: list[metrics.StatisticsType] = [metrics.StatisticsType.rmse,
                                                         metrics.StatisticsType.mean,
                                                         metrics.StatisticsType.median,
                                                         metrics.StatisticsType.std,
                                                         metrics.StatisticsType.min,
                                                         metrics.StatisticsType.max,
                                                         metrics.StatisticsType.sse]
    all_metrics: list[metrics.PE] = [metrics.APE, metrics.RPE]
    dict_all_results: dict = {}
    for metric in all_metrics:
        dict_metric: dict = {}

        for pose_relation in all_pose_relations:
            dict_relation: dict = {}

            # Skip uncompatible relation with metric
            if metric is metrics.APE and pose_relation == metrics.PoseRelation.point_distance_error_ratio:
                continue

            data_copied = copy.deepcopy(data)
            metric_with_relation: metrics.PE = metric(pose_relation)
            metric_with_relation.process_data(data_copied)

            for stat in all_statistic_types:
                final_stat = metric_with_relation.get_statistic(stat)
                dict_relation[stat.name] = final_stat

            dict_metric[pose_relation.name] = dict_relation
        
        dict_all_results[metric.__name__] = dict_metric

    return dict_all_results