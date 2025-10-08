###########################################################
#
# mapping.py
#
# Demo code for running ROMAN mapping
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
import pickle
import time
import logging
import os
import tqdm
from os.path import expandvars

from roman.map.run import ROMANMapRunner
from roman.params.system_params import SystemParams

from merge_demo_output import merge_demo_output


def run(system_params: SystemParams, output_path: str, robot_index: int):
    
    runner = ROMANMapRunner(system_params, robot_index, verbose=True)

    # Setup logging
    # TODO: add support for logfile
    logging.basicConfig(
        level=logging.ERROR, 
        format='%(asctime)s %(message)s', 
        datefmt='%m-%d %H:%M:%S', 
    )

    print("Running segment tracking! Start time {:.1f}, end time {:.1f}".format(runner.t0, runner.tf))
    wc_t0 = time.time()

    bar = tqdm.tqdm(total=len(runner.times()), desc="Frame Processing")
    for t in runner.times():
        runner.update(t)
        bar.update()

    print(f"Segment tracking took {time.time() - wc_t0:.2f} seconds")
    print(f"Run duration was {runner.tf - runner.t0:.2f} seconds")
    print(f"Compute per second: {(time.time() - wc_t0) / (runner.tf - runner.t0):.2f}")
    if system_params.use_scene_graph:
        print(f"Number of poses: {len(runner.mapper.poses)}.")
    else:
        print(f"Number of poses: {len(runner.mapper.poses_flu_history)}")

    # Get path to output pickle file
    pkl_path = os.path.expanduser(expandvars(output_path)) + ".pkl"
    pkl_file = open(pkl_path, 'wb')

    # Save the file differently depending on if we want to align with ROMANMap or MeronomyGraph
    if system_params.use_roman_map_for_alignment:
        pickle.dump(runner.mapper.get_roman_map(), pkl_file, -1)
    elif system_params.use_scene_graph:
        pickle.dump(runner.mapper, pkl_file, -1)
    else:
        raise ValueError("Cannot set 'use_roman_map_for_alignment' to False when 'use_scene_graph' is also False.")
    pkl_file.close()

    timing_file = os.path.expanduser(expandvars(output_path)) + ".time.txt"
    with open(timing_file, 'w') as f:
        f.write(f"dt: {system_params.data_params.dt}\n\n")
        f.write(f"AVERAGE TIMES\n")
        f.write(f"fastsam: {np.mean(runner.processing_times.fastsam_times):.3f}\n")
        f.write(f"segment_track: {np.mean(runner.processing_times.map_times):.3f}\n")
        f.write(f"total: {np.mean(runner.processing_times.total_times):.3f}\n")
        f.write(f"TOTAL TIMES\n")
        f.write(f"total: {np.sum(runner.processing_times.total_times):.2f}\n")
    
    del runner
    return

def mapping(system_params: SystemParams, output_path: str, robot_index: int,  max_time: float = None) -> None:

    if max_time is not None:
        try:
            mapping_iter = 0
            while True:
                
                system_params.data_params.time_params = {
                    't0': max_time * mapping_iter, 
                    'tf': max_time * (mapping_iter + 1),
                    'relative': True}
                
                run(system_params, output_path=f"{output_path}_{mapping_iter}")
                mapping_iter += 1
        except:
            demo_output_files = [f"{output_path}_{mi}.pkl" for mi in range(mapping_iter)]
            merge_demo_output(demo_output_files, f"{output_path}.pkl")
    
    else:
        run(system_params, output_path, robot_index)