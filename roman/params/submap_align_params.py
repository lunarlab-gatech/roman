###########################################################
#
# submap_align_params.py
#
# Params for ROMAN object registration.
#
# Authors: Mason Peterson
#
# Jan. 15, 2025
#
###########################################################

import numpy as np

from dataclasses import dataclass, field
from typing import List
import os
import yaml

import clipperpy
from roman.align.roman_registration import ROMANRegistration, ROMANParams
from roman.align.ransac_reg import RansacReg
from roman.align.dist_reg_with_pruning import DistRegWithPruning, GravityConstraintError
from robotdatapy.data.pose_data import PoseData

@dataclass
class SubmapAlignParams:

    dim: int = 3                            # 2 or 3. 2D or 3D object map registration
    method: str = 'roman'                   # by default, use semantic + pca + volume + gravity
                                            # same as in ROMAN paper.
                                            # See get_object_registration for other methods
    fusion_method: str = 'geometric_mean'   # How to fuse similarity scores. (geometric_mean, 
                                            # arithmetic_mean, product)
    submap_radius: float = 15.0             # Radius of submap in meters
    submap_center_dist: float = 10.0        # Distance between submap centers in meters
    submap_center_time: float = 50.0        # time threshold between segments and submap center times
    submap_max_size: int = 40               # Maximum number of segments in a submap (to save computation)
    single_robot_lc: bool = False           # If true, do not try and perform loop closures with submaps
                                            # nearby in time
    single_robot_lc_time_thresh: float = 50.0   # Time threshold for single robot loop closure
    force_rm_lc_roll_pitch: bool = True     # If true, remove parts of rotation about x or y axes
    force_rm_upside_down: bool = True       # If true, assumes upside down submap rotations are incorrect
    use_object_bottom_middle: bool = False  # If true, uses the bottom middle of the object as a reference
                                            # point for registration rather than the center of the object
    
    # registration params
    sigma: float = 0.4
    epsilon: float = 0.6
    mindist: float = 0.4
    epsilon_shape: float = 0.0
    ransac_iter: int = int(1e6)
    cosine_min: float = 0.85
    cosine_max: float = 1.0
    semantics_dim: int = 768

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            params = yaml.full_load(f)
        return cls(**params)
    
    def get_object_registration(self):
        if self.fusion_method == 'geometric_mean':
            sim_fusion_method = clipperpy.invariants.ROMAN.GEOMETRIC_MEAN
        elif self.fusion_method == 'arithmetic_mean':
            sim_fusion_method = clipperpy.invariants.ROMAN.ARITHMETIC_MEAN
        elif self.fusion_method == 'product':
            sim_fusion_method = clipperpy.invariants.ROMAN.PRODUCT
        if self.method == 'spvg':
            self.method = 'roman'           

        if self.method in ['clipper', 'gravity', 'pcavolgrav', 'extentvolgrav', 'roman', 'sevg', 'spv', 'semanticgrav']:
            roman_params = ROMANParams()
            roman_params.point_dim = self.dim
            roman_params.sigma = self.sigma
            roman_params.epsilon = self.epsilon
            roman_params.min_dist = self.mindist
            roman_params.fusion = sim_fusion_method

            roman_params.gravity = self.method in ['gravity', 'pcavolgrav', 'extentvolgrav', 'roman', 'sevg', 'semanticgrav']
            roman_params.volume = self.method in ['pcavolgrav', 'extentvolgrav', 'roman', 'sevg', 'spv']
            roman_params.extent = self.method in ['extentvolgrav', 'sevg']
            roman_params.pca = self.method in ['pcavolgrav', 'roman', 'spv']
            roman_params.cos_min = self.cosine_min
            roman_params.cos_max = self.cosine_max
            roman_params.epsilon_shape = self.epsilon_shape
            
            if self.method in ['roman', 'sevg', 'semanticgrav']:
                roman_params.semantics_dim = self.semantics_dim
            
            # if self.method == 'clipper':
            #     method_name = f'{self.dim}D Point CLIPPER'
            # elif self.method == 'gravity':
            #     method_name = 'Gravity Guided CLIPPER'
            #     roman_params.gravity = True
            # elif self.method == 'pcavolgrav':
            #     method_name = f'Gravity Guided PCA feature-based Volume Registration'
            # elif self.method == 'extentvolgrav':
            #     method_name = f'Gravity Guided Extent-based Volume Registration'
            # elif self.method == 'roman':
            #     method_name = 'CLIP Semantic + PCA + Volume + Gravity'
            # elif self.method == 'sevg':
            #     method_name = 'Semantic + Extent + Volume + Gravity'

            registration = ROMANRegistration(roman_params)

        elif self.method == 'clipper+prune':
            method_name = f'Gravity Filtered Pruning'
            registration = DistRegWithPruning(
                sigma=self.sigma, 
                epsilon=self.epsilon, 
                mindist=self.mindist, 
                shape_epsilon=self.epsilon_shape,
                cos_min=self.cosine_min,
                dim=self.dim, 
                use_gravity=True
            )
        elif self.method == 'ransac':
            method_name = 'RANSAC'
            registration = RansacReg(dim=self.dim, max_iteration=self.ransac_iter)
        else:
            assert False, "Invalid method"
        return registration
        
    
@dataclass
class SubmapAlignInputOutput:
    inputs: List[any]
    gt_pose_data: list[PoseData]
    output_dir: str
    run_name: str
    input_type_pkl: bool = True
    input_type_json: bool = False
    robot_names: List[str] = field(default_factory=lambda: ["0", "1"])
    robot_env: str = None
    lc_association_thresh: int = 4
    g2o_t_std: float = 0.5
    g2o_r_std: float = np.deg2rad(0.5)
    debug_show_maps: bool = False
            
    @property
    def output_img(self):
        return os.path.join(self.output_dir, f'{self.run_name}.png')
    
    @property
    def output_matrix(self):
        return os.path.join(self.output_dir, f'{self.run_name}.matrix.pkl')
    
    @property
    def output_pkl(self):
        return os.path.join(self.output_dir, f'{self.run_name}.pkl')
    
    @property
    def output_timing(self):
        return os.path.join(self.output_dir, f'{self.run_name}.timing.txt')
    
    @property
    def output_params(self):
        return os.path.join(self.output_dir, f'{self.run_name}.params.txt')
    
    @property
    def output_g2o(self):
        return os.path.join(self.output_dir, f'{self.run_name}.g2o')
    
    @property
    def output_lc_json(self):
        return os.path.join(self.output_dir, f'{self.run_name}.json')
    
    @property
    def output_submaps(self):
        return [os.path.join(self.output_dir, f'{rn}.sm.json') for rn in self.robot_names]
    