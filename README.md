# ROMAN

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

## Demos

### Kimera Multi Data

A short demo is available to run ROMAN on small subset of the [Kimera Multi Data](https://github.com/MIT-SPARK/Kimera-Multi-Data) Instructions for running the demo:

1. Download a small portion of the [Kimera Multi Data](https://github.com/MIT-SPARK/Kimera-Multi-Data) that is used for the ROMAN SLAM demo. The data subset is available for download [here](https://drive.google.com/drive/folders/1ANdi4IyroWzJmd85ap1V-IMF8-I9haUB?usp=sharing).

2. Run the following commands, updating paths if needed:

```
mkdir demo_output
export YOLO_VERBOSE=False
export ROMAN_DEMO_DATA=/home/dbutterfield3/roman/datasets/roman_kimera_multi_data-20250415T143326Z-003/roman_kimera_multi_data
export ROMAN_WEIGHTS=/home/dbutterfield3/roman/weights
python3 demo/demo.py -p demo/params/demo -o demo_output --viz-observations --viz-map --viz-3d --skip-align --skip-rpgo    
```

Optionally, the mapping process can be visualized with the `-m` argument to show the map projected on the camera image as it is created or `-3` command to show a 3D visualization of the map.
However, these will cause the demo to run slower. 

The output includes map visualization, loop closure accuracy results, and pose graph optimization results including root mean squared absolute trajectory error. 

## HERCULES

Run the following command to run this demo:

```
mkdir demo_output
export YOLO_VERBOSE=False
python3 demo/demo.py -p demo/params/hercules -o demo_output --skip-align --skip-rpgo
python3 demo/demo.py -p demo/params/hercules -o demo_output --skip-map --skip-rpgo    
python3 demo/demo.py -p demo/params/hercules -o demo_output --skip-map --skip-align
```

If you want to enable visualizations of the mapping step:
```
python3 demo/demo.py -p demo/params/hercules -o demo_output --viz-observations --viz-map --viz-3d --skip-align --skip-rpgo
```

In the output directory, the 'map' folder will contain .mp4 files with visualizations, and .pkl files with the stored ROMAN maps. To visualize a map, run the command below:

```
python3 demo/o3d_viz.py demo_output/map/<robot_name>.pkl
```

In the 'align' folder, the file 'align.png' will contain a plot titled "Number of CLIPPER associations". If this plot is a single color, there's a high likelihood that no associations were found, and thus no loop closures. Each detected loop closure can be found in `align.json`.