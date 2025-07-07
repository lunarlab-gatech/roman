import numpy as np
import unittest
from roman.object.segment import Segment
from roman.map.observation import Observation

class TestObject(unittest.TestCase):

    dummy_pose: np.ndarray

    def setUp(self):
        # Make a dummy pose
        self.dummy_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    
    def test_segment_add_semantic_descriptor(self):
        # Build test segment
        seg1 = Segment(Observation(0.0, self.dummy_pose), None, 0, 0.05)

        # Make some fake embeddings
        test_embeddings_1 = [(np.array([ 1, 0, 0], dtype=np.float64), 10)]
        test_embeddings_2 = [(np.array([ 0, 0,-1], dtype=np.float64),  1),
                             (np.array([ 0, 1, 0], dtype=np.float64),  5)]
        seg1._add_semantic_descriptor(test_embeddings_1)
        seg1._add_semantic_descriptor(test_embeddings_2)

        # See if the weighted average is calculated correctly
        np.testing.assert_array_almost_equal(seg1.semantic_descriptor, np.array([0.890871, 0.445435, -0.0890871]), decimal=6)


if __name__ == "__main__":
    unittest.main()