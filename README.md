# ROMAN

[![Unit Tests](https://github.com/lunarlab-gatech/roman/actions/workflows/python_test.yaml/badge.svg?branch=echelon)](https://github.com/lunarlab-gatech/roman/actions/workflows/python_test.yaml)
[![Coverage Status](https://coveralls.io/repos/github/lunarlab-gatech/roman/badge.svg?branch=echelon)](https://coveralls.io/github/lunarlab-gatech/roman?branch=echelon)

Our fork of the [ROMAN](https://github.com/mit-acl/roman) repository for evaluation as a baseline running on the HERCULES dataset.

## Install

### Docker Setup

Make sure to install:
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

Then, clone this repository into a desired location on your computer.

After that, navigate to the `docker` directory. Log in to the user that you want the docker file to create in the container. Then, edit the `DOCKERFILE` to update these lines:
- `ARG USERNAME=`: Your username
- `ARG USER_UID=`: Output of `echo $UID`
- `ARG USER_GID=`: Output of `id -g`

Edit the `enter_container.sh` script with the following paths:
- `DATA_DIR=`: The directory where the HERCULES dataset is located
- `REPO_DIR=`: The directory of this repository

Now, run the following commands:
```
build_container.sh
run_container.sh
```

The rest of this README **assumes that you are inside the Docker container**. For easier debugging and use, its highly recommended to install the [VSCode Docker extension](https://code.visualstudio.com/docs/containers/overview), which allows you to start/stop the container and additionally attach VSCode to the container by right-clicking on the container and selecting `Attach Visual Studio Code`. If that isn't possible, you can re-enter the container running the following command:
```
enter_container.sh
```

### ROMAN Install

Next install ROMAN by running the following command from the root folder of this repository:
```
./install.sh
pip uninstall matplotlib
```

Finally, run the following to fix `Could not load the Qt platform plug "xcb"` bug:
```
mv ~/.local/lib/python3.10/site-packages/cv2/qt ~/.local/lib/python3.10/site-packages/cv2/qt.bak
```

## Run algorithm
### HERCULES (Australia Environment)

Run the following command to run this demo:

```
mkdir demo_output
export YOLO_VERBOSE=False
python3 demo/demo.py -p demo/params/hercules_AustraliaEnv -o demo_output --skip-align --skip-rpgo
python3 demo/demo.py -p demo/params/hercules_AustraliaEnv -o demo_output --skip-map --skip-rpgo    
python3 demo/demo.py -p demo/params/hercules_AustraliaEnv -o demo_output --skip-map --skip-align
```

If you want to enable visualizations of the mapping step:
```
python3 demo/demo.py -p demo/params/hercules_AustraliaEnv -o demo_output --viz-observations --viz-map --viz-3d --skip-align --skip-rpgo
```

In the output directory, the 'map' folder will contain .mp4 files with visualizations, and .pkl files with the stored ROMAN maps. 

To visualize a map, run the command below:

```
python3 demo/o3d_viz.py demo_output/map/<robot_name>.pkl
```

In the 'align' folder, the file 'align.png' will contain a plot titled "Number of CLIPPER associations". If this plot is a single color, there's a high likelihood that no associations were found, and thus no loop closures. Each detected loop closure can be found in `align.json`.

To visualize the .g2o files in the output, you can use the g2o_viewer binary from (g2o)[https://github.com/RainerKuemmerle/g2o]. If not on the path, you can find it in the `build/bin` directory of the repository after building:

```
g2o_viewer <g2o file>
```

#### Note on Parameters

In order to adapt ROMAN to work successfully on HERCULES, two types of parameters were changed:
- 1. Robot parameters, or those that would always have to change due to differences in the robots we are using. 
- 2. Tunable parameters, or those that don't fall in the category above.

For a fair comparison with ROMAN, ideally we only change parameters in category 1. However, due to the wildly different "Australian Environment", we found that it was necessary to change some parameters in 2 to successfully find a map alignment. Thus, below we document all parameters that were changed in both categories 1 & 2; 1 so that its easy to see for the future what we would need to change to apply other experiments, and 2 so that the reasons for these changes can be well documented:

Category 1:
```
data.yaml: runs, run_env, img_data, depth_data, pose_data, time_tol
fastsam.yaml: depth_scale
gt_pose.yaml: path, csv_options, time_tol
offline_rpgo.yaml: odom_t_std, odom_r_std
```

Category 2:
```
fastsam.yaml: max_depth, voxel_size
mapper.yaml: iou_voxel_size, segment_voxel_size
submap_align.yaml: submap_radius, submap_center_dist
```

For reasons for changes in category 2, see the corresponding .yaml files.

#### TEMP
Profiling command:
```
python3 -m cProfile -o profile.out demo/demo.py -p demo/params/hercules_AustraliaEnv -o demo_output --skip-align --skip-rpgo
```

### GRaCo (Ground-04)

This demo is run in a similar manner to the `HERCULES` demo from above, but using the parameter directory `demo/params/graco_ground04`.
