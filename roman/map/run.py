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
from roman.params.data_params import DataParams
from roman.params.mapper_params import MapperParams
from roman.params.fastsam_params import FastSAMParams

from ..scene_graph.logger import logger

@dataclass
class ProcessingTimes:
    fastsam_times: list
    map_times: list
    total_times: list

class ROMANMapRunner:
    def __init__(self, data_params: DataParams, fastsam_params: FastSAMParams, 
                 mapper_params: MapperParams, verbose=False, viz_map=False, 
                 viz_observations=False, viz_3d=False, save_viz=False):
        self.data_params = data_params
        self.fastsam_params = fastsam_params
        self.mapper_params = mapper_params

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
        self.fastsam = FastSAMWrapper.from_params(self.fastsam_params, self.depth_data.camera_params)
        self.mapper = SceneGraph3D(self.data_params.pose_data_params.T_camera_flu)
        
        self.verbose = verbose
        self.viz_map = viz_map
        self.viz_observations = viz_observations
        self.viz_3d = viz_3d
        self.save_viz = save_viz
        self.viz_imgs = []
        self.processing_times = ProcessingTimes([], [], [])

    def times(self):
        return np.arange(self.t0, self.tf, self.data_params.dt)

    def update(self, t: float): 
        t0 = self.img_data.t0
        tf = self.img_data.tf

        #if self.verbose: print(f"t: {t - t0:.2f} = {t}")
        img_output = None
        update_t0 = time.time()

        img_time, observations, pose_odom_camera, img, depth_img = self.update_fastsam(t)
        update_t1 = time.time()
        if observations is not None and pose_odom_camera is not None and img is not None:
            img_output = self.update_segment_track(img_time, observations, pose_odom_camera, img, depth_img)
        
        update_t2 = time.time()
        self.processing_times.map_times.append(update_t2 - update_t1)
        self.processing_times.fastsam_times.append(update_t1 - update_t0)
        self.processing_times.total_times.append(update_t2 - update_t0)

        return img_output

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
           
        observations = self.fastsam.run(img_t, pose_odom_camera, img, img_depth=img_depth)
        return img_t, observations, pose_odom_camera, img, img_depth

    def update_segment_track(self, t, observations, pose_odom_camera, img, depth_img): 

        # collect reprojected masks
        reprojected_bboxs = []
        img_ret = None
        if self.viz_observations:
            for i, node in enumerate(self.mapper.root_node):
                if not node.is_RootGraphNode():
                    # Get the bounding box for each object
                    bbox = node.reprojected_bbox(pose_odom_camera, self.img_data.camera_params.K, 
                            self.img_data.camera_params.width, self.img_data.camera_params.height)
                    if bbox is not None:
                        reprojected_bboxs.append((i, bbox))

        # Create the segmentation image
        height = self.data_params.img_data_params.height
        width = self.data_params.img_data_params.width
        seg_img = np.zeros((len(observations), height, width), dtype=np.uint16)
        for i, obs in enumerate(observations):
            mask = obs.mask
            seg_img[i] = np.multiply(mask.astype(np.uint16), np.full((height, width), i+1, dtype=np.uint16))

        self.mapper.update(t, pose_odom_camera, observations, img, depth_img, self.data_params.img_data_params, seg_img)

        # if not self.mapper_params.use_3D_Scene_graph:
        #     if self.viz_map or self.viz_observations or self.viz_3d:
        #         img_ret = self.draw(t, img, pose_odom_camera, observations, reprojected_bboxs)

        #     if self.save_viz:
        #         self.viz_imgs.append(img_ret)
        if self.viz_map or self.viz_observations or self.viz_3d:
            print("Visualization is not currently implemented for SceneGraph3D!")

        return img_ret
        

    def draw(self, t, img, pose_odom_camera, observations, reprojected_bboxs):
        img_orig = deepcopy(img)
        
        if len(img_orig.shape) == 2:
            img = np.concatenate([img_orig[...,None]]*3, axis=2)
            
        if self.viz_map:
            img = visualize_map_on_img(t, pose_odom_camera, img, self.mapper)

        if self.viz_observations:
            img_fastsam = visualize_observations_on_img(t, img_orig, self.mapper, observations, reprojected_bboxs)
        
        if self.viz_3d:
            img_3d = visualize_3d_on_img(t, pose_odom_camera @ self.mapper.T_camera_flu, self.mapper)
                
        # rotate images
        img = self.fastsam.apply_rotation(img)
        if self.viz_observations:
            img_fastsam = self.fastsam.apply_rotation(img_fastsam)
                
        # concatenate images
        img_ret = np.zeros((img.shape[0], 0, img.shape[2]), dtype=np.uint8)
        if self.viz_map:
            img_ret = np.hstack([img_ret, img])
        if self.viz_3d:
            img_ret = np.hstack([img_ret, img_3d])
        if self.viz_observations:
            img_ret = np.hstack([img_ret, img_fastsam])

        return img_ret