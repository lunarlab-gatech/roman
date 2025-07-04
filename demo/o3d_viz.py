# python3 ./demo/o3d_viz.py /home/masonbp/results/west_point_2023/segment_tracking/rgbd_2/sparkal2.pkl
# helpful for figuring out open3d visualization: 
# https://github.com/isl-org/Open3D/pull/3233/files#diff-4bf889278ab3bd4bb37dafdc436b6b6bc772b5e730dcd67a5e7e3f369cee694c

import numpy as np
import open3d as o3d
import pickle
import argparse
import os

from roman.viz import visualize_3d
from roman.map.map import ROMANMap

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file', type=str, help='Path to pickle file(s)')
    parser.add_argument('--no-text', action='store_true', help='Do not display text labels')
    parser.add_argument('--no-orig', action='store_true', help='Do not display origin')
    parser.add_argument('-t', '--time-range', type=float, nargs=2, help='Time range to display')
    args = parser.parse_args()

    pickle_files = args.pickle_file.split(',')
    maps = [ROMANMap.from_pickle(x) for x in pickle_files]
    merged_map = ROMANMap.concatenate(maps)

    visualize_3d(
        roman_map=merged_map,
        time_range=args.time_range,
        show_labels=not args.no_text,
        show_origin=not args.no_orig,
        use_clip_to_find_text=True,
        word_list_for_clip="short"
    )