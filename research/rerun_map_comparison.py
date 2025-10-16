import argparse
from roman.map.map import ROMANMap
from roman.rerun_wrapper.rerun_wrapper_window_map_comparison import RerunWrapperWindowMapComparison
from roman.rerun_wrapper.rerun_wrapper import RerunWrapper


def main():
    parser = argparse.ArgumentParser(description="Load two SceneGraph3D pickle files and visualize them using Rerun.")
    parser.add_argument("pickle_path_1", type=str, help="Path to the first pickle file containing a SceneGraph3D object.")
    parser.add_argument("pickle_path_2", type=str, help="Path to the second pickle file containing a SceneGraph3D object.")
    parser.add_argument("robot_name_1", type=str,help="Name of the Robot in the first pickle map")
    parser.add_argument("robot_name_2", type=str,help="Name of the Robot in the second pickle map")
    args = parser.parse_args()

    # Load both pickle files
    print(f"Loading first scene graph from: {args.pickle_path_1}")
    sg0: ROMANMap = ROMANMap.from_pickle(args.pickle_path_1)

    print(f"Loading second scene graph from: {args.pickle_path_2}")
    sg1: ROMANMap = ROMANMap.from_pickle(args.pickle_path_2)

    # Create a rerun visualizer with MapFinal window and update with graphs
    comparison_window = RerunWrapperWindowMapComparison(True, args.robot_name_1, args.robot_name_2)
    rerun_viewer = RerunWrapper(True, "MeronomyGraph Two Map Comparison", [comparison_window])
    comparison_window.update_curr_time(0)
    comparison_window.set_curr_robot(args.robot_name_1)
    comparison_window.update_segments(sg0.segments)
    comparison_window.set_curr_robot(args.robot_name_2)
    comparison_window.update_segments(sg1.segments)
    
if __name__ == "__main__":
    main()
