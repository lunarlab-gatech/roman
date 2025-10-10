import argparse
from roman.map.map import ROMANMap
from roman.rerun_wrapper import RerunWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Load two SceneGraph3D pickle files and visualize them using Rerun."
    )
    parser.add_argument(
        "pickle_path_1",
        type=str,
        help="Path to the first pickle file containing a SceneGraph3D object."
    )
    parser.add_argument(
        "pickle_path_2",
        type=str,
        help="Path to the second pickle file containing a SceneGraph3D object."
    )

    args = parser.parse_args()

    # Load both pickle files
    print(f"Loading first scene graph from: {args.pickle_path_1}")
    sg0: ROMANMap = ROMANMap.from_pickle(args.pickle_path_1)

    print(f"Loading second scene graph from: {args.pickle_path_2}")
    sg1: ROMANMap = ROMANMap.from_pickle(args.pickle_path_2)

    # Create a rerun visualizer with MapFinal window and update with graphs
    rerun_viewer = RerunWrapper(name="MeronomyGraph Two Map Comparison",
                                windows=[RerunWrapper.RerunWrapperWindow.MapFinal])
    rerun_viewer.update_curr_time(0)
    rerun_viewer.set_curr_robot_index(0)
    rerun_viewer.update_segments(sg0.segments)
    rerun_viewer.set_curr_robot_index(1)
    rerun_viewer.update_segments(sg1.segments)
    
if __name__ == "__main__":
    main()
