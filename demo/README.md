# ROMAN Demo

Using `demo.py`, the full ROMAN pipeline can be run on a multi-robot dataset, or mapping, loop closures, or pose graph optimization can be run separately.

After following the [install instructions](../README.md/#install) and the [demo setup instructions](../README.md/#demo), the demo can be run with the following command:

```
cd roman
mkdir demo_output
python3 demo/demo.py \
    -p params/demo \
    -o demo_output
```

Optionally, the mapping process can be visualized with the `-m` argument to show the map projected on the camera image as it is created or `-3` command to show a 3D visualization of the map. However, these will cause the demo to run slower.

Note that by default, FastSAM and CLIP are run on GPU. 
If your computer does not have a GPU, you can use `-p params/demo_no_gpu` which specifies that CPU should be used for running FastSAM and additionally does not use CLIP semantic embeddings since these are extrememly slow to compute on CPU. 
Even without CLIP semantic embeddings, the demo may run slower than real-time without using GPU.

### Association Visualization

After running the demo, you can create a post-processed visualizations of object matches that have been found (see example [here](https://www.youtube.com/watch?v=y51NDoPpBy8&t=3s)).

To create this visualization, run

```
python3 ./demo/association_vid.py ./demo_output ./demo_output/association_vid.mp4 --runs sparkal1 sparkal2
```

You will be shown the results of aligning object submaps which includes the ground truth distance between submaps,
and the error in the estimated submap alignment poses.
You will then be prompted to enter a submap index to visualize (in order robot 1, along y-axis then robot2, along x-axis).
For, example, try 

```
4 7
```

Then, a video will be created showing the associations that were found between the submaps you entered.
One thing you may notice is that the cars that are parked along the sidewalk in this example have changed (the two sessions were not taken at the same time), but ROMAN is still able to use the geometry of the newly parked cars to successfully align submaps.

## Demo Output

The demo output includes three directories:
- `map`:
    - `<run>.mp4`: mapping visualization video (if demo run with `-m` or `-3` argument)
    - `<run>.pkl`: pickled ROMAN map (see the [ROMANMap class](../roman/map/map.py). This can be visualized using the `o3d_viz.py` script in this directory)
- `align`:
    - `<run1>_<run2>`:
        - `align.png`: global localization results. If ground truth pose data has been provided, this file shows heatmaps showing the results of attempting to perform object registration between each pair of submaps from `<run1>` and `<run2>`. Ground truth submap distance is shown, along with registration translation and rotation errors, the number of associated objects, and the ground truth heading difference between the two submap center poses. This file will be much less meaningful if ground truth pose data is not provided to the demo. 
        - `align.pkl`: pickled global localization results (see the [SubmapAlignResults class](../roman/align/results.py))
- `offline_rpgo`:
    - `result.png`: resulting trajectory estimate plots
    - `result.txt`: absolute trajectory error result if ground truth is provided
    - `<run>.csv`: optimized trajectory estimate using ROMAN loop closures

The output includes map visualization, loop closure accuracy results, and pose graph optimization results including root mean squared absolute trajectory error.

## Custom Data

Up to this point, ROMAN has mostly been used to process ROS 1/2 data (agnostic of ROS version).
This documentation gives instructions for how to image data from a ROS bag and how to load pose data from a ROS bag or from a CSV file.
To run ROMAN on played back data or in real-time, use the [roman_ros](https://github.com/mit-acl/roman_ros) repository. 
If you are interested in loading in another type of data, the [ROMANMapRunner class](../roman/map/run.py) can be edited to load other data types using the `robotdatapy` [PoseData](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/pose_data.py) and [ImgData](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/img_data.py) interfaces.

To run ROMAN on your data, start by copying the [demo params directory](../params/demo).
In most cases, only the `data.yaml` and `fastsam.yaml` file will need to be customized. 
Within [data.yaml](../params/demo/data.yaml), the `img_data`, `depth_data`, and `pose_data` fields will need to be edited to load your data as explained in the following sections.
Within [fastsam.yaml](../params/demo/fastsam.yaml) the `depth_scale` and `max_depth` variables may need to be updated to reflect your depth camera.
Pass the updated params directory in using the `-p` argument.

### Loading ROS Image Data

Update the following subfields within the `img_data`, and `depth_data` fields in your new `data.yaml` file.

```
path: <bag file path>
topic: <topic name>
camera_info_topic: <topic name>
compressed: <true if datatype is compressed>
compressed_rvl: <true if data is compressed using rvl (if you do not know, probably false)>
```

### Loading ROS Pose Data

Update the following subfields within the `pose_data` field in your new `data.yaml` file.

If using a Pose or Odometry topic:

```
type: "bag"
path: <bag file path>
topic: <topic name of type Pose or Odometry>
time_tol: <time in seconds> # Pose data will not be interpolated if there is not a recorded pose within time_tol of a desired time
```

If the pose is recorded using `/tf` topic rather tha a Pose or Odometry message, then use the following:

```
type: bag_tf
path: <bag file path>
parent_frame: <parent frame_id>
child_frame: <child frame_id>
inv: <Whether the transformation from /tf should be inverted or not>
```

Additionally the following subfields should be provided if they are not identity: `T_camera_flu` and `T_odombase_camera`. These are used for determining (1) the pose of the a forward-left-up coordinate frame with respect to the camera frame and (2) the pose of the camera expressed in the odometry base frame. `T_camera_flu` is used to inform ROMAN roughly along which axis gravity will point, since ROMAN uses gravity direction to help guide loop closures. `T_odombase_camera` is necessary if the camera frame is not the child of the pose topic.

These two transformation subfields can be described with the following formats:

```
input_type: "string"
string: <"T_RDFFLU" or "T_FLURDF">
```

or 

```
input_type: "tf"
parent: <parent frame_id>
child: <child frame_id>
inv: <Whether the transformation from /tf_static should be inverted or not>
```

or 

```
input_type: "matrix"
matrix: <4x4 array>
```

### Using Point Cloud Data

First, create a new field called `use_pointcloud` and set it to `True`. Instead of using `depth_data`, create a new field called `pointcloud_data` and fill the following subfields:

```
path: <bag file path>
topic: <topic name>
```

If your bag does not have a correct static TF transform between the camera and lidar frames (or if the image and pointcloud messages are not labelled with the correct frame), you can define the following optional subfield:

```
T_camera_rangesense: <transform source>
```

See the section above for ways to define a transformation.

### Using environment variables in data paths/topics

To enable using the same set of parameters on different robot data, the `run_env` field in your [data.yaml](../demo/data.yaml) can be used to export a desired environment variable with the run names specified in the `runs` field. As an example, if you have two robots, julius and augustus, you can use `runs: [julius, augustus]` and in the path field of `img_data`, specify the path as `"${ROBOT_ENV_VAR}.bag`. Then use `run_env: ROBOT_ENV_VAR` so that ROMAN will correctly grab `julius.bag` for the julius run and `augustus.bag` for the augustus run.

Additionally, `runs` can be used to define run-specific parameters which will override default parameters if they exist. For example, if all robots have the same organization for the image data topic except julius, you can override it:

```
img_data:
    path: ${ROBOT_ENV_VAR}.bag
    topic: ${ROBOT_ENV_VAR}/color/image_raw
    compressed: False

julius:
    img_data:
        topic: ${ROBOT_ENV_VAR}/forward/color/image_raw/compressed
        compressed: True
```

This is especially useful for defining run-specific topic names, data formats, and transforms. The default parameter is not required if it is appropriately defined for every run.


### Additional parameters

`max_time` can be defined to limit the duration of each mapping run, to avoid overflowing memory usage.

`time` can be used to specify the time range of a bag that you want run. For example, if you want to run a bag from 40 to 100 seconds from when it starts, you can do:

```
time:
    t0: 40
    tf: 100
    relative: True
```

Setting `relative` to False means specified times must be the same as in the rosbag (typically unix epoch timestamps).