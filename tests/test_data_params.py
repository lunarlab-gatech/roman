import numpy as np
import os
from pathlib import Path
from robotdatapy.data.pose_data import PoseData
from roman.params.data_params import PoseDataGTParams, DataParams
import unittest
from typeguard import typechecked


@typechecked
class TestPoseDataGTParams(unittest.TestCase):

    def test_get_pose_data(self):
        """ Ensure we get data for two different robots instead of the same"""

        # Load params
        files_path = Path(__file__).parent / "files" / "test_data_params" / "test_get_pose_data" 
        data_params = DataParams.from_yaml(files_path / "data.yaml")
        pose_data_gt_params = PoseDataGTParams.from_yaml(files_path / "gt_pose.yaml")

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


if __name__ == "__main__":
    unittest.main()