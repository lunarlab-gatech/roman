###########################################################
#
# run.py
#
# A class for processing ROMAN mapping
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia
#
# Dec. 21, 2024
#
###########################################################

import numpy as np

from dataclasses import dataclass
import time
from copy import deepcopy

from robotdatapy.data.robot_data import NoDataNearTimeException

from roman.object.segment import Segment
from roman.viz import visualize_map_on_img, visualize_observations_on_img, visualize_3d_on_img
from roman.map.mapper import Mapper
from roman.scene_graph.scene_graph_3D import SceneGraph3D
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.params.system_params import SystemParams
from ..logger import logger

@dataclass
class ProcessingTimes:
    fastsam_times: list
    map_times: list
    total_times: list

class ROMANMapRunner:
    def __init__(self, system_params: SystemParams, robot_index: int, verbose=False):
        self.system_params = system_params
        self.data_params = system_params.data_params
        self.fastsam_params = system_params.fastsam_params
        self.mapper_params = system_params.mapper_params

        if verbose: print("Extracting time range...")
        self.time_range = self.data_params.time_range

        if verbose: 
            print("Loading image data...")
            print(f"Time range: {self.time_range}")

        start_time = time.time()
        self.img_data = self.data_params.load_img_data()
        print(f"Image Data Loading Time: {time.time() - start_time} seconds")
        if verbose:
            self.t0 = self.img_data.t0
            self.tf = self.img_data.tf

        if verbose: print("Loading depth data for time range {}...".format(self.time_range))
        start_time = time.time()
        self.depth_data = self.data_params.load_depth_data()
        print(f"Depth Data Loading Time: {time.time() - start_time} seconds")

        if verbose: print("Loading pose data...")
        start_time = time.time()
        self.camera_pose_data = self.data_params.load_pose_data()
        print(f"Pose Data Loading Time: {time.time() - start_time} seconds")

        if verbose: print("Setting up FastSAM...")
        self.fastsam = FastSAMWrapper.from_params(self.fastsam_params, self.depth_data.camera_params, self.system_params)

        if verbose: print("Setting up mapper...")
        if system_params.use_scene_graph:
            self.mapper = SceneGraph3D(system_params, self.img_data.camera_params, self.data_params.pose_data_params.T_camera_flu, robot_index)
        else: 
            self.mapper = Mapper(self.mapper_params, self.img_data.camera_params)
            if self.data_params.pose_data_params.T_camera_flu is not None:
                self.mapper.set_T_camera_flu(self.data_params.pose_data_params.T_camera_flu)
        
        self.verbose = verbose
        self.processing_times = ProcessingTimes([], [], [])

    def times(self):
        return np.arange(self.t0, self.tf, self.data_params.dt)

    def update(self, t: float) -> None: 
        t0 = self.img_data.t0
        tf = self.img_data.tf

        update_t0 = time.time()

        img_time, observations, pose_odom_camera, img, depth_img = self.update_fastsam(t)
        update_t1 = time.time()
        if observations is not None and pose_odom_camera is not None and img is not None:
            self.update_segment_track(img_time, observations, pose_odom_camera, img, depth_img)
        
        update_t2 = time.time()
        self.processing_times.map_times.append(update_t2 - update_t1)
        self.processing_times.fastsam_times.append(update_t1 - update_t0)
        self.processing_times.total_times.append(update_t2 - update_t0)

    def update_fastsam(self, t):

        try:
            img_t = self.img_data.nearest_time(t)
        except NoDataNearTimeException as e:
            logger.info(f"[red]NoDataNearTimeException[/red]: No time within threshold of time {img_t}.")
            return None, None, None, None, None
        
        try:
            img = self.img_data.img(img_t)
        except NoDataNearTimeException as e:
            logger.info(f"[red]NoDataNearTimeException[/red]: No image data within threshold of time {img_t}.")
            return None, None, None, None, None
        
        try:
            img_depth = self.depth_data.img(img_t)
        except NoDataNearTimeException as e:
            logger.info(f"[red]NoDataNearTimeException[/red]: No depth data within threshold of time {img_t}.")
            return None, None, None, None, None
        
        try:
            pose_odom_camera = self.camera_pose_data.T_WB(img_t)
        except NoDataNearTimeException as e:
            logger.info(f"[red]NoDataNearTimeException[/red]: No pose data within threshold of time {img_t}.")
            return None, None, None, None, None
           
        observations = self.fastsam.run(float(img_t), pose_odom_camera, img, img_depth=img_depth)
        return img_t, observations, pose_odom_camera, img, img_depth

    def update_segment_track(self, t, observations, pose_odom_camera, img, depth_img) -> None: 

        # Create the segmentation image
        height = self.data_params.img_data_params.height
        width = self.data_params.img_data_params.width
        seg_img = np.zeros((len(observations), height, width), dtype=np.uint16)
        for i, obs in enumerate(observations):
            mask = obs.mask
            seg_img[i] = np.multiply(mask.astype(np.uint16), np.full((height, width), i+1, dtype=np.uint16))

        if self.system_params.use_scene_graph:
            self.mapper.update(t, pose_odom_camera, observations, img, depth_img, self.data_params.img_data_params, seg_img)
        else:
            if len(observations) > 0:
                self.mapper.update(t, pose_odom_camera, observations)
            else:
                self.mapper.poses_flu_history.append(pose_odom_camera @ self.mapper._T_camera_flu)
                self.mapper.times_history.append(t)
