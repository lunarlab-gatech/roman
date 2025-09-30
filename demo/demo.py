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
import numpy as np
import yaml
import wandb
from pathlib import Path
import torch
import random
import mapping

from robotdatapy.data.pose_data import PoseData
from roman.params.submap_align_params import SubmapAlignInputOutput, SubmapAlignParams
from roman.align.submap_align import submap_align
from roman.align.results import calculate_loop_closure_error, extract_num_loop_closures
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

# Try to enforce determinism in the algorithm
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, help='Path to params directory', required=True, default=None)
    parser.add_argument('--max-time', type=float, default=None, help='If the input data is too large, this allows a maximum time' +
                        'to be set, such that if the mapping will be chunked into max_time increments and fused together')
    parser.add_argument('--skip-map', action='store_true', help='Skip mapping')
    parser.add_argument('--skip-align', action='store_true', help='Skip alignment')
    parser.add_argument('--skip-rpgo', action='store_true', help='Skip robust pose graph optimization')
    parser.add_argument('--skip-indices', type=int, nargs='+', help='Skip specific runs in mapping and alignment', default=[])
    parser.add_argument('--disable-wandb', action='store_true', help='Skip logging to W&B')
    parser.add_argument('--use-map', type=str, help='Run name with map we want to use', default=None)
    args = parser.parse_args()
    if not args.skip_map and args.use_map is not None:
        raise ValueError("Can't set --use-map without --skip-map")
    if args.skip_map and args.use_map is None:
        raise ValueError("User specified --skip-map but provided no map to use instead with --use-map")

    # Setup parameters
    system_params = SystemParams.from_param_dir(args.params)

    # Setup WandB to track this run
    if not args.disable_wandb:
        def shorten(d):
            return {(''.join([w[0] for w in k.split('_')]) if isinstance(v, dict) else k): (shorten(v) if isinstance(v, dict) else v) for k, v in d.items()}
        config_dict = shorten({'system_params': system_params.model_dump()})
        config_dict['skip_map'] = args.skip_map
        config_dict['use_map'] = args.use_map
        wandb_run = wandb.init(project='HERCULES Experiment',
                        config=config_dict)
        run_name = wandb_run.name
    else:
        run_name = 'latest'

    # Create output directories
    params_path = Path(args.params)
    output_path = params_path.parent.parent.parent / "demo_output" / params_path.name / run_name
    os.makedirs(os.path.join(output_path, "map"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "align"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "offline_rpgo"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "offline_rpgo/sparse"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "offline_rpgo/dense"), exist_ok=True)
    params_output_path = output_path / "params.yaml"
    params_output_path.write_text(yaml.safe_dump(convert_paths(system_params.model_dump())))

    # Set the directory with the map to use
    input_map_path = output_path
    if args.use_map is not None:
        input_map_path = input_map_path.parent / args.use_map

    # Extract GT Pose Data
    gt_pose_data: list[PoseData] = system_params.pose_data_gt_params.get_pose_data(system_params.data_params)

    # Run the Mapping step
    if not args.skip_map:

        for i, run in enumerate(system_params.data_params.runs):
            if i in args.skip_indices: continue
                
            # mkdir $output_path/map
            args.output = os.path.join(output_path, "map", f"{run}")

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
                align_path = os.path.join(output_path, "align", f"{system_params.data_params.runs[i]}_{system_params.data_params.runs[j]}")
                os.makedirs(align_path, exist_ok=True)

                # Load the two pickle files containing the maps from the first step
                input_files: list[str] = [os.path.join(input_map_path, "map", f"{system_params.data_params.runs[i]}.pkl"),
                            os.path.join(input_map_path, "map", f"{system_params.data_params.runs[j]}.pkl")]

                # Create the Input/Output parameters
                sm_io = SubmapAlignInputOutput(
                    inputs=input_files,
                    output_dir=align_path,
                    run_name="align",
                    lc_association_thresh=system_params.num_req_assoc,
                    gt_pose_data=[gt_pose_data[i], gt_pose_data[j]],
                    robot_names=[system_params.data_params.runs[i], system_params.data_params.runs[j]],
                    robot_env=system_params.data_params.run_env,
                )

                # If the same robot is being aligned to itself, enable single_robot_lc.
                # This avoids doing loop closures with itself, not sure why they do this.
                system_params.submap_align_params.single_robot_lc = (i == j)

                # Run the alignment process
                submap_align(system_params=system_params, sm_params=system_params.submap_align_params, sm_io=sm_io)

                # Calculate loop closure errors
                json_path = os.path.join(align_path, "align.json")      
                num_loop_closures = extract_num_loop_closures(json_path)       
                print(f"Total Number of Loop Closures: {num_loop_closures}")
                errors = calculate_loop_closure_error(json_path, gt_pose_data[i], gt_pose_data[j])
                if not args.disable_wandb and i != j:
                    wandb_run.log({"LC: Total number": num_loop_closures,
                                   "LC: Mean Translation Error": errors[0], 
                                   "LC: Median Translation Error": errors[1],
                                   "LC: Std Translation Error": errors[2],
                                   "LC: Mean Rotation Angle Error": errors[3],
                                   "LC: Median Rotation Angle Error": errors[4],
                                   "LC: Std Rotation Angle Error":errors[5]})
                       
    if not args.skip_rpgo:
        min_keyframe_dist = 0.01 if not system_params.offline_rpgo_params.sparsified else 2.0
        # Create g2o files for odometry
        for i, run in enumerate(system_params.data_params.runs):
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(input_map_path, "map", f"{run}.pkl"),
                g2o_file=os.path.join(output_path, "offline_rpgo/sparse", f"{run}.g2o"),
                time_file=os.path.join(output_path, "offline_rpgo/sparse", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=min_keyframe_dist,
                t_std=system_params.offline_rpgo_params.odom_t_std,
                r_std=system_params.offline_rpgo_params.odom_r_std,
                verbose=True
            )
            
            # create dense g2o file
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(input_map_path, "map", f"{run}.pkl"),
                g2o_file=os.path.join(output_path, "offline_rpgo/dense", f"{run}.g2o"),
                time_file=os.path.join(output_path, "offline_rpgo/dense", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=None,
                t_std=system_params.offline_rpgo_params.odom_t_std,
                r_std=system_params.offline_rpgo_params.odom_r_std,
                verbose=True
            )
        
        # Combine timing files
        odom_sparse_all_time_file = os.path.join(output_path, "offline_rpgo/sparse", "odom_all.time.txt")
        odom_dense_all_time_file = os.path.join(output_path, "offline_rpgo/dense", "odom_all.time.txt")
        with open(odom_sparse_all_time_file, 'w') as f:
            for i, run in enumerate(system_params.data_params.runs):
                with open(os.path.join(output_path, "offline_rpgo/sparse", f"{run}.time.txt"), 'r') as f2:
                    f.write(f2.read())
                f2.close()
        with open(odom_dense_all_time_file, 'w') as f:
            for i, run in enumerate(system_params.data_params.runs):
                with open(os.path.join(output_path, "offline_rpgo/dense", f"{run}.time.txt"), 'r') as f2:
                    f.write(f2.read())
                f2.close()
        
        # Fuse all odometry g2o files
        odom_sparse_all_g2o_file = os.path.join(output_path, "offline_rpgo/sparse", "odom_all.g2o")
        g2o_fusion_config = create_config(robots=system_params.data_params.runs, 
            odometry_g2o_dir=os.path.join(output_path, "offline_rpgo/sparse"))
        g2o_file_fusion(g2o_fusion_config, odom_sparse_all_g2o_file, thresh=system_params.num_req_assoc)
        
        # Fuse dense g2o file including loop closures
        dense_g2o_file = os.path.join(output_path, "offline_rpgo/dense", "odom_and_lc.g2o")
        g2o_fusion_config = create_config(robots=system_params.data_params.runs, 
            odometry_g2o_dir=os.path.join(output_path, "offline_rpgo/dense"),
            submap_align_dir=os.path.join(output_path, "align"), align_file_name="align")
        g2o_file_fusion(g2o_fusion_config, dense_g2o_file, thresh=system_params.num_req_assoc)

        # Add loop closures to odometry g2o files
        if system_params.offline_rpgo_params.sparsified:
            final_g2o_file = os.path.join(output_path, "offline_rpgo", "odom_and_lc.g2o")
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
        result_g2o_file = os.path.join(output_path, "offline_rpgo", "result.g2o")
        ros_launch_command = f"roslaunch kimera_centralized_pgmo offline_g2o_solver.launch \
            g2o_file:={final_g2o_file} \
            output_path:={os.path.join(output_path, 'offline_rpgo')}"
        roman_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        rpgo_read_g2o_executable = \
            f"{roman_path}/dependencies/Kimera-RPGO/build/RpgoReadG2o"
        rpgo_command = f"{rpgo_read_g2o_executable} 3d {final_g2o_file}" \
            + f" -1.0 -1.0 {system_params.offline_rpgo_params.gnc_inlier_threshold} {os.path.join(output_path, 'offline_rpgo')} v"
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
        plt.savefig(os.path.join(output_path, "offline_rpgo", "result.png"))
        print(f"Results saved to {os.path.join(output_path, 'offline_rpgo', 'result.png')}")
        
        # Save csv files with resulting trajectories
        for i, run in enumerate(system_params.data_params.runs):
            pose_data = g2o_and_time_to_pose_data(result_g2o_file, 
                                                  odom_sparse_all_time_file if system_params.offline_rpgo_params.sparsified else odom_dense_all_time_file, 
                                                  robot_id=i)
            pose_data.to_csv(os.path.join(output_path, "offline_rpgo", f"{run}.csv"))
            print(f"Saving {run} pose data to {os.path.join(output_path, 'offline_rpgo', f'{run}.csv')}")

        # Report ATE results
        if len(gt_pose_data) != 0:
            ate_rmse = evaluate(
                result_g2o_file, 
                odom_sparse_all_time_file  if system_params.offline_rpgo_params.sparsified else odom_dense_all_time_file, 
                {i: gt_pose_data[i] for i in range(len(gt_pose_data))},
                {i: system_params.data_params.runs[i] for i in range(len(system_params.data_params.runs))},
                system_params.data_params.run_env,
                output_dir=str(output_path)
            )
            print("ATE results:")
            print("============")
            print(ate_rmse)
            if not args.disable_wandb:
                wandb_run.log({"RMS ATE": ate_rmse})
            with open(os.path.join(output_path, "offline_rpgo", "ate_rmse.txt"), 'w') as f:
                print(ate_rmse, file=f)
                f.close()
            