###########################################################
#
# demo.py
#
# Demo code for running full ROMAN SLAM pipeline including 
# mapping, loop closure, and pose-graph optimization
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia
#
# Dec. 21, 2024
#
###########################################################

import matplotlib.pyplot as plt
import argparse
import json
import os
import yaml
import wandb
import loop_closure_viz
from pathlib import Path

from roman.params.submap_align_params import SubmapAlignInputOutput, SubmapAlignParams
from roman.align.submap_align import submap_align
from roman.offline_rpgo.extract_odom_g2o import roman_map_pkl_to_g2o
from roman.offline_rpgo.g2o_file_fusion import create_config, g2o_file_fusion
from roman.offline_rpgo.combine_loop_closures import combine_loop_closures
from roman.offline_rpgo.plot_g2o import plot_g2o, DEFAULT_TRAJECTORY_COLORS, G2OPlotParams
from roman.offline_rpgo.g2o_and_time_to_pose_data import g2o_and_time_to_pose_data
from roman.offline_rpgo.evaluate import evaluate
from roman.offline_rpgo.edit_g2o_edge_information import edit_g2o_edge_information
from roman.params.offline_rpgo_params import OfflineRPGOParams
from roman.params.data_params import DataParams
from roman.params.scene_graph_3D_params import SceneGraph3DParams, GraphNodeParams
from roman.params.system_params import SystemParams
from roman.utils import expandvars_recursive

import mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, help='Path to params directory', required=True, default=None)
    parser.add_argument('-o', '--output-dir', type=str, help='Path to output directory', required=True, default=None)
    
    parser.add_argument('--max-time', type=float, default=None, help='If the input data is too large, this allows a maximum time' +
                        'to be set, such that if the mapping will be chunked into max_time increments and fused together')

    parser.add_argument('--skip-map', action='store_true', help='Skip mapping')
    parser.add_argument('--skip-align', action='store_true', help='Skip alignment')
    parser.add_argument('--skip-rpgo', action='store_true', help='Skip robust pose graph optimization')
    parser.add_argument('--skip-indices', type=int, nargs='+', help='Skip specific runs in mapping and alignment', default=[])
    args = parser.parse_args()

    # Setup parameters
    system_params = SystemParams.from_param_dir(args.params)

    # Setup WandB to track this run
    config_dict = {'system_params': system_params.model_dump()}
    run = wandb.init(project='ROMAN + MeronomyMapping Ablations',
                     config=config_dict)

    # Create output directories
    params_path = Path(args.params)
    output_dir = params_path.parent() / "demo_output" / params_path.name / run.name
    os.makedirs(os.path.join(output_dir, "map"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "align"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "offline_rpgo"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "offline_rpgo/sparse"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "offline_rpgo/dense"), exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    
    # Run the Mapping step
    if not args.skip_map:

        for i, run in enumerate(system_params.data_params.runs):
            if i in args.skip_indices: continue
                
            # mkdir $output_dir/map
            args.output = os.path.join(output_dir, "map", f"{run}")

            # shell: export RUN=run
            if system_params.data_params.run_env is not None:
                os.environ[system_params.data_params.run_env] = run
            
            print(f"Mapping: {run}")
            mapping.mapping(
                system_params,
                output_path=args.output,
                run_name=run,
                max_time=args.max_time
            )
    
    # Iterate through each pair of runs and do alignment
    if not args.skip_align:
        for i in range(len(system_params.data_params.runs)):
            if args.skip_indices and i in args.skip_indices: continue
            for j in range(i, len(system_params.data_params.runs)):
                if args.skip_indices and j in args.skip_indices: continue

                print(f"Running alignment for {system_params.data_params.runs[i]}_{system_params.data_params.runs[j]}")
                
                # Make the output directory to store the alignment results
                output_dir = os.path.join(output_dir, "align", f"{system_params.data_params.runs[i]}_{system_params.data_params.runs[j]}")
                os.makedirs(output_dir, exist_ok=True)

                # Load the two pickle files containing the maps from the first step
                input_files: list[str] = [os.path.join(output_dir, "map", f"{system_params.data_params.runs[i]}.pkl"),
                            os.path.join(output_dir, "map", f"{system_params.data_params.runs[j]}.pkl")]

                # Create the Input/Output parameters
                sm_io = SubmapAlignInputOutput(
                    inputs=input_files,
                    output_dir=output_dir,
                    run_name="align",
                    lc_association_thresh=args.num_req_assoc,
                    input_gt_pose_yaml=[system_params.gt_file, system_params.gt_file],
                    robot_names=[system_params.data_params.runs[i], system_params.data_params.runs[j]],
                    robot_env=system_params.data_params.run_env,
                )

                # If the same robot is being aligned to itself, enable single_robot_lc.
                # This avoids doing loop closures with itself, not sure why they do this.
                system_params.submap_align_params.single_robot_lc = (i == j)

                # Run the alignment process
                submap_align(sm_params=system_params.submap_align_params, sm_io=sm_io)

                # Calculate loop closure errors
                json_path = os.path.join(output_dir, "align.json")

                gt_csvs = [None, None]
                for index, yaml_file in enumerate([system_params.gt_file, system_params.gt_file]):
                    if yaml_file is not None:
                        if sm_io.robot_env is not None:
                            os.environ[sm_io.robot_env] = sm_io.robot_names[index]
                        with open(os.path.expanduser(yaml_file), 'r') as f:
                            gt_pose_args = yaml.safe_load(f)
                            for k, v in gt_pose_args.items():
                                if type(gt_pose_args[k]) == str:
                                    gt_pose_args[k] = expandvars_recursive(gt_pose_args[k])
                            gt_csvs[index] = gt_pose_args["path"]
                      
                mean_trans_error, mean_rot_error = loop_closure_viz.calculate_loop_closure_error(json_path, gt_csvs[i], gt_csvs[j])
                if i != j:
                    run.log({"LC: Mean Translation Error": mean_trans_error, "LC: Mean Rotation Angle Error": mean_rot_error})
                       
    if not args.skip_rpgo:
        min_keyframe_dist = 0.01 if not system_params.offline_rpgo_params.sparsified else 2.0
        # Create g2o files for odometry
        for i, run in enumerate(system_params.data_params.runs):
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(output_dir, "map", f"{run}.pkl"),
                g2o_file=os.path.join(output_dir, "offline_rpgo/sparse", f"{run}.g2o"),
                time_file=os.path.join(output_dir, "offline_rpgo/sparse", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=min_keyframe_dist,
                t_std=system_params.offline_rpgo_params.odom_t_std,
                r_std=system_params.offline_rpgo_params.odom_r_std,
                verbose=True
            )
            
            # create dense g2o file
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(output_dir, "map", f"{run}.pkl"),
                g2o_file=os.path.join(output_dir, "offline_rpgo/dense", f"{run}.g2o"),
                time_file=os.path.join(output_dir, "offline_rpgo/dense", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=None,
                t_std=system_params.offline_rpgo_params.odom_t_std,
                r_std=system_params.offline_rpgo_params.odom_r_std,
                verbose=True
            )
        
        # Combine timing files
        odom_sparse_all_time_file = os.path.join(output_dir, "offline_rpgo/sparse", "odom_all.time.txt")
        odom_dense_all_time_file = os.path.join(output_dir, "offline_rpgo/dense", "odom_all.time.txt")
        with open(odom_sparse_all_time_file, 'w') as f:
            for i, run in enumerate(system_params.data_params.runs):
                with open(os.path.join(output_dir, "offline_rpgo/sparse", f"{run}.time.txt"), 'r') as f2:
                    f.write(f2.read())
                f2.close()
        with open(odom_dense_all_time_file, 'w') as f:
            for i, run in enumerate(system_params.data_params.runs):
                with open(os.path.join(output_dir, "offline_rpgo/dense", f"{run}.time.txt"), 'r') as f2:
                    f.write(f2.read())
                f2.close()
        
        # Fuse all odometry g2o files
        odom_sparse_all_g2o_file = os.path.join(output_dir, "offline_rpgo/sparse", "odom_all.g2o")
        g2o_fusion_config = create_config(robots=system_params.data_params.runs, 
            odometry_g2o_dir=os.path.join(output_dir, "offline_rpgo/sparse"))
        g2o_file_fusion(g2o_fusion_config, odom_sparse_all_g2o_file, thresh=args.num_req_assoc)
        
        # Fuse dense g2o file including loop closures
        dense_g2o_file = os.path.join(output_dir, "offline_rpgo/dense", "odom_and_lc.g2o")
        g2o_fusion_config = create_config(robots=system_params.data_params.runs, 
            odometry_g2o_dir=os.path.join(output_dir, "offline_rpgo/dense"),
            submap_align_dir=os.path.join(output_dir, "align"), align_file_name="align")
        g2o_file_fusion(g2o_fusion_config, dense_g2o_file, thresh=args.num_req_assoc)

        # Add loop closures to odometry g2o files
        if system_params.offline_rpgo_params.sparsified:
            final_g2o_file = os.path.join(output_dir, "offline_rpgo", "odom_and_lc.g2o")
            combine_loop_closures(
                g2o_reference=odom_sparse_all_g2o_file, 
                g2o_extra_lc=dense_g2o_file, 
                vertex_times_reference=odom_sparse_all_time_file,
                vertex_times_extra_lc=odom_dense_all_time_file,
                output_file=final_g2o_file,
            )
        else:
            final_g2o_file = dense_g2o_file
        
        # change lc covar
        with open(os.path.expanduser(final_g2o_file), 'r') as f:
            g2o_lines = f.readlines()
        g2o_lines = edit_g2o_edge_information(g2o_lines, system_params.offline_rpgo_params.lc_t_std, 
                                              system_params.offline_rpgo_params.lc_r_std, loop_closures=True)
        with open(os.path.expanduser(final_g2o_file), 'w') as f:
            for line in g2o_lines:
                f.write(line + '\n')
            f.close()
            
        # run kimera centralized robust pose graph optimization
        result_g2o_file = os.path.join(output_dir, "offline_rpgo", "result.g2o")
        ros_launch_command = f"roslaunch kimera_centralized_pgmo offline_g2o_solver.launch \
            g2o_file:={final_g2o_file} \
            output_path:={os.path.join(output_dir, 'offline_rpgo')}"
        roman_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        rpgo_read_g2o_executable = \
            f"{roman_path}/dependencies/Kimera-RPGO/build/RpgoReadG2o"
        rpgo_command = f"{rpgo_read_g2o_executable} 3d {final_g2o_file}" \
            + f" -1.0 -1.0 {system_params.offline_rpgo_params.gnc_inlier_threshold} {os.path.join(output_dir, 'offline_rpgo')} v"
        os.system(rpgo_command)
        # os.system(ros_launch_command)
        
        # plot results
        g2o_symbol_to_name = {chr(97 + i): system_params.data_params.runs[i] for i in range(len(system_params.data_params.runs))}
        g2o_plot_params = G2OPlotParams()
        fig, ax = plt.subplots(3, 1, figsize=(7,14), gridspec_kw={'height_ratios': [3, 1, 1]})
        for i in range(3):
            g2o_plot_params.axes = [(0, 1), (0, 2), (1, 2)][i]
            g2o_plot_params.legend = (i == 0)
            plot_g2o(
                g2o_path=result_g2o_file,
                g2o_symbol_to_name=g2o_symbol_to_name,
                g2o_symbol_to_color=DEFAULT_TRAJECTORY_COLORS,
                ax=ax[i],
                params=g2o_plot_params
            )
        plt.savefig(os.path.join(output_dir, "offline_rpgo", "result.png"))
        print(f"Results saved to {os.path.join(output_dir, 'offline_rpgo', 'result.png')}")
        
        # Save csv files with resulting trajectories
        for i, run in enumerate(system_params.data_params.runs):
            pose_data = g2o_and_time_to_pose_data(result_g2o_file, 
                                                  odom_sparse_all_time_file if system_params.offline_rpgo_params.sparsified else odom_dense_all_time_file, 
                                                  robot_id=i)
            pose_data.to_csv(os.path.join(output_dir, "offline_rpgo", f"{run}.csv"))
            print(f"Saving {run} pose data to {os.path.join(output_dir, 'offline_rpgo', f'{run}.csv')}")

        # Report ATE results
        if system_params.gt_file is not None:
            ate_rmse = evaluate(
                result_g2o_file, 
                odom_sparse_all_time_file  if system_params.offline_rpgo_params.sparsified else odom_dense_all_time_file, 
                {i: system_params.gt_file for i in range(len(system_params.data_params.runs))},
                {i: system_params.data_params.runs[i] for i in range(len(system_params.data_params.runs))},
                system_params.data_params.run_env,
                output_dir=output_dir
            )
            print("ATE results:")
            print("============")
            print(ate_rmse)
            run.log({"RMS ATE": ate_rmse})
            with open(os.path.join(output_dir, "offline_rpgo", "ate_rmse.txt"), 'w') as f:
                print(ate_rmse, file=f)
                f.close()
            