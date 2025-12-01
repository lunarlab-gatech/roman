import numpy as np
import os
from pathlib import Path
from robotdatapy.data.pose_data import PoseData
from roman.params.data_params import PoseDataGTParams, PathParams, DataParams, PoseDataParams
import unittest
from typeguard import typechecked
import yaml


@typechecked
class TestPoseDataGTParams(unittest.TestCase):

    def test_get_pose_data(self):
        """ Ensure we get data for two different robots instead of the same"""

        # Load params
        files_path = Path(__file__).parent / "files" / "test_data_params" / "test_get_pose_data" 

        with open(files_path / "system_params.yaml") as f:
            data = yaml.safe_load(f)
        path_params = PathParams.from_dict(data['path_params'])
        data_params = DataParams.from_yaml(files_path / "data.yaml", path_params)
        pose_data_gt_params = PoseDataGTParams.from_yaml(files_path / "gt_pose.yaml", path_params)

        # Extract the pose data
        os.environ['ROMAN_DEMO_DATA'] = str(files_path)
        gt_pose_data: list[PoseData] = pose_data_gt_params.get_pose_data(data_params)

        # Make sure that they match what we expect
        loaded = gt_pose_data[0].T_WB(1665777910.030874112)[:3,3]
        expected = np.array([411.50177706246,-238.032463323776,4.010165160349026])
        np.testing.assert_array_almost_equal(loaded, expected, 7)

        loaded = gt_pose_data[1].T_WB(1666030800.829998080)[:3,3]
        expected = np.array([502.4748159278734,-244.67149292615736,4.890290188060116])
        np.testing.assert_array_almost_equal(loaded, expected, 7)

@typechecked
class TestPoseDataParams(unittest.TestCase):

    def test_find_transformation(self):
        # Create the PoseDataParams object
        files_path = Path(__file__).parent / "files" / "test_data_params" / "test_find_transformation"
        with open(files_path / 'pose_data.yaml') as f:
            pose_data = yaml.safe_load(f)
        with open(files_path / 'system_params.yaml') as f:
            path_data = yaml.safe_load(f)
        path_params = PathParams.from_dict(path_data['path_params'])
        path_params.path_to_dataset_folder = files_path
        poseDataParams = PoseDataParams.from_dict(pose_data, path_params)

        # Make sure _find_transformation gets the correct matrix
        H_Drone1 =np.array([[ 0,  0.17364818,  0.98480775,  0.35],      
                            [-1,           0,           0,     0],       
                            [0, -0.98480775,  0.17364818,  -0.1],    
                            [0,           0,           0,     1]])
        H_Husky1 = np.array([[ 0, 0, 1, 0],
                             [-1, 0, 0, 0],
                             [0,-1, 0, 0.35],
                             [ 0, 0, 0, 1]])

        os.environ["ROBOT"] = "Drone1"
        H: np.ndarray = poseDataParams._find_transformation(pose_data['T_odombase_camera'])
        np.testing.assert_array_equal(H, H_Drone1)
        os.environ["ROBOT"] = "Husky1"
        H: np.ndarray = poseDataParams._find_transformation(pose_data['T_odombase_camera'])
        np.testing.assert_array_equal(H, H_Husky1)
        os.environ["ROBOT"] = "Drone2"
        H: np.ndarray = poseDataParams._find_transformation(pose_data['T_odombase_camera'])
        np.testing.assert_array_equal(H, H_Drone1)


if __name__ == "__main__":
    unittest.main()