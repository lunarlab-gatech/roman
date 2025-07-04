###########################################################
#
# data_params.py
#
# Parameter class for data loading
#
# Authors: Mason Peterson
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
from dataclasses import dataclass
import yaml
from typing import List, Tuple, Optional
from functools import cached_property

from robotdatapy.data import ImgData, PoseData
from robotdatapy.transform import T_FLURDF, T_RDFFLU

from roman.utils import expandvars_recursive

@dataclass
class ImgDataParams:
    
    type: str
    path: str
    path_times: str
    topic: Optional[str] = None
    camera_info_topic: Optional[str] = None
    compressed: bool = True
    compressed_rvl: bool = False
    compressed_encoding: str = 'passthrough'
    K: Optional[list] = None
    D: Optional[list] = None
    width: int = None
    height: int = None
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        return cls(**params_dict)
    
@dataclass
class PoseDataParams:
    
    params_dict: dict
    T_camera_flu_dict: dict
    T_odombase_camera_dict: dict = None
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        params_dict_subset = {k: v for k, v in params_dict.items() 
                       if k != 'T_camera_flu' and k != 'T_odombase_camera'}
        T_camera_flu_dict = params_dict['T_camera_flu']
        T_odombase_camera_dict = params_dict['T_odombase_camera'] \
            if 'T_odombase_camera' in params_dict else None
        return cls(params_dict=params_dict_subset, T_camera_flu_dict=T_camera_flu_dict, 
                   T_odombase_camera_dict=T_odombase_camera_dict)
        
    @property
    def T_camera_flu(self) -> np.array:
        return self._find_transformation(self.T_camera_flu_dict)
    
    @property
    def T_odombase_camera(self) -> np.array:
        if self.T_odombase_camera_dict is not None:
            return self._find_transformation(self.T_odombase_camera_dict)
        else:
            return np.eye(4)
        
    def load_pose_data(self, extra_key_vals: dict) -> PoseData:
        """
        Loads pose data.

        Returns:
            PoseData: Pose data object.
        """
        params_dict = {k: v for k, v in self.params_dict.items()}
        for k, v in extra_key_vals.items():
            params_dict[k] = v
            
        # expand variables
        for k, v in params_dict.items():
            if type(params_dict[k]) == str:
                params_dict[k] = expandvars_recursive(params_dict[k])
        print("Called from data_params.py: ", params_dict)
        pose_data = PoseData.from_dict(params_dict)
        return pose_data
    
    def _find_transformation(self, param_dict) -> np.array:
        """
        Finds the transformation from the body frame to the camera frame.

        Returns:
            np.array: Transformation matrix.
        """
        # T_postmultiply = np.eye(4)
        # # if 'T_body_odom' in param_dict:
        # #     T_postmultiply = np.linalg.inv(np.array(param_dict['T_body_odom']).reshape((4, 4)))
        # if 'T_body_cam' in param_dict:
        #     T_postmultiply = T_postmultiply @ np.array(param_dict['T_body_cam']).reshape((4, 4))
        if param_dict['input_type'] == 'string':
            if param_dict['string'] == 'T_FLURDF':
                return T_FLURDF
            elif param_dict['string'] == 'T_RDFFLU':
                return T_RDFFLU
            else:
                raise ValueError("Invalid string.")
        elif param_dict['input_type'] == 'tf':
            img_file_path = expandvars_recursive(self.params_dict["path"])
            T = PoseData.static_tf_from_bag(
                expandvars_recursive(img_file_path), 
                expandvars_recursive(param_dict['parent']), 
                expandvars_recursive(param_dict['child'])
            )
            if param_dict['inv']:
                return np.linalg.inv(T)
            else:
                return T
        elif param_dict['input_type'][0:6] == 'matrix':
            T = np.array(param_dict[expandvars_recursive(param_dict['input_type'])], dtype=np.float64).reshape((4,4))
            return T
        else:
            raise ValueError("Invalid input type.")
    
@dataclass
class DataParams:
    
    img_data_params: ImgDataParams
    depth_data_params: ImgDataParams
    pose_data_params: PoseDataParams
    dt: float = 1/6
    runs: list = None
    run_env: str = None
    time_params: dict = None
    kitti: bool = False
    
    def __post_init__(self):
        if self.time_params is not None:
            assert 'relative' in self.time_params['time'], "relative must be specified in params"
            assert 't0' in self.time_params['time'], "t0 must be specified in params"
            assert 'tf' in self.time_params['time'], "tf must be specified in params"
        
    @classmethod
    def from_yaml(cls, yaml_path: str, run: str = None):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if run is None:
            return cls(
                None, None, None,
                dt=data['dt'] if 'dt' in data else 1/6,
                runs=data['runs'] if 'runs' in data else None,
                run_env=data['run_env'] if 'run_env' in data else None
            )
        elif run in data:
            run_data = data[run]
        else:
            run_data = data
        return cls(
            ImgDataParams.from_dict(run_data['img_data']),
            ImgDataParams.from_dict(run_data['depth_data']),
            PoseDataParams.from_dict(run_data['pose_data']),
            dt=run_data['dt'] if 'dt' in run_data else 1/6,
            runs=data['runs'] if 'runs' in data else None,
            run_env=data['run_env'] if 'run_env' in data else None,
            time_params=run_data['time_params'] if 'time_params' in run_data else None,
            kitti=run_data['kitti'] if 'kitti' in run_data else False
        )
        
    @cached_property
    def time_range(self) -> Tuple[float, float]:
        return self._extract_time_range()
    
    def load_pose_data(self) -> PoseData:
        """
        Loads pose data.

        Returns:
            PoseData: Pose data object.
        """        
        if self.pose_data_params.T_odombase_camera is not None:
            T_postmultiply = self.pose_data_params.T_odombase_camera
        else:
            T_postmultiply = None
            
        extra_key_vals={'T_postmultiply': T_postmultiply, 'interp': True}
            
        return self.pose_data_params.load_pose_data(extra_key_vals)
        
    def load_img_data(self) -> ImgData:
        """
        Loads image data.
        
        Args:
            time_range (List[float, float]): Time range to load image data.

        Returns:
            ImgData: Image data object.
        """
        return self._load_img_data(color=True)
    
    def load_depth_data(self) -> ImgData:
        """
        Loads depth data.
        
        Args:
            time_range (List[float, float]): Time range to load depth data.

        Returns:
            ImgData: Depth data object.
        """
        return self._load_img_data(color=False)
        
    def _load_img_data(self, color=True) -> ImgData:
        """
        Loads color or depth image data.

        Args:
            color (bool, optional): True if color, False if depth. Defaults to True.

        Returns:
            ImgData: Image data object.
        """

        # Extract relevant perams depending on if RGB or Depth
        if color:
            img_data_params = self.img_data_params
        else:
            img_data_params = self.depth_data_params

        # Depending on data type
        if self.kitti:
            img_data = ImgData.from_kitti(self.img_data_params.path, 'rgb' if color else 'depth')
            img_data.extract_params()
        elif img_data_params.type == "npy":
            img_file_path = expandvars_recursive(img_data_params.path)
            print(img_data_params.path_times)
            times_file_path = expandvars_recursive(img_data_params.path_times)
            img_data = ImgData.from_npy(
                path=img_file_path,
                path_times=times_file_path,
                K=img_data_params.K,
                D=img_data_params.D,
                width=img_data_params.width,
                height=img_data_params.height, 
                time_tol=self.dt / 2.0
            )
        else:
            img_file_path = expandvars_recursive(img_data_params.path)
            img_data = ImgData.from_bag(
                path=img_file_path,
                topic=expandvars_recursive(img_data_params.topic),
                time_tol=self.dt / 2.0,
                time_range=self.time_range,
                compressed=img_data_params.compressed,
                compressed_rvl=img_data_params.compressed_rvl,
                compressed_encoding=img_data_params.compressed_encoding
            )
            img_data.extract_params(expandvars_recursive(img_data_params.camera_info_topic))
        return img_data
    
    def _extract_time_range(self) -> Tuple[float, float]:
        """
        Uses the params dictionary and image data to set an absolute time range for the data.

        Args:
            params (dict): Params dict.

        Returns:
            Tuple[float, float]: Beginning and ending time (or none).
        """
        if self.kitti:
            time_range = [self.time_params['t0'], self.time_params['tf']]
        else:
            img_file_path = expandvars_recursive(self.img_data_params.path)
            if self.time_params is not None:
                if self.time_params['relative']:
                    topic_t0 = ImgData.topic_t0(img_file_path, 
                        expandvars_recursive(self.img_data_params.topic))
                    time_range = [topic_t0 + self.time_params['t0'], 
                                  topic_t0 + self.time_params['tf']]
                else:
                    time_range = [self.time_params['t0'], 
                                  self.time_params['tf']]
            else:
                time_range = None
        return time_range
        