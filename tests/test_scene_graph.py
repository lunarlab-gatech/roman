import numpy as np
import unittest
from roman.object.segment import Segment
from roman.map.observation import Observation
from roman.scene_graph.graph_node import LeafGraphNode
from roman.scene_graph.scene_graph_3D import SceneGraph3D

class TestSceneGraph(unittest.TestCase):

    dummy_pose: np.ndarray

    def setUp(self):
        # Make a dummy pose
        self.dummy_pose = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float64)

    @staticmethod
    def helper_generate_cube_surface(n_points=2048, radius=1.0) -> np.ndarray:
        """ Helper method that generates a point cloud on the surface of a cube """
        rng = np.random.default_rng(42)

        points = np.empty((n_points, 3))
        # Each point randomly assigned to one of the 6 cube faces
        faces = rng.integers(0, 6, size=n_points)

        # Sample coordinates for the two axes on the face in [-radius, radius]
        coords = rng.uniform(-radius, radius, size=(n_points, 2))

        for i, face in enumerate(faces):
            if face == 0:    # +X face
                points[i] = [radius, coords[i, 0], coords[i, 1]]
            elif face == 1:  # -X face
                points[i] = [-radius, coords[i, 0], coords[i, 1]]
            elif face == 2:  # +Y face
                points[i] = [coords[i, 0], radius, coords[i, 1]]
            elif face == 3:  # -Y face
                points[i] = [coords[i, 0], -radius, coords[i, 1]]
            elif face == 4:  # +Z face
                points[i] = [coords[i, 0], coords[i, 1], radius]
            else:            # -Z face
                points[i] = [coords[i, 0], coords[i, 1], -radius]

        return points

    def test_convex_hull_geometric_overlap(self):
        # Setup two overlapping point clouds
        sphere1 = TestSceneGraph.helper_generate_cube_surface(radius=0.5)
        sphere2 = TestSceneGraph.helper_generate_cube_surface(radius=1)
        sphere2[:,0] += 1

        # Create a test segment and a test observation
        node1 = LeafGraphNode(0, None, False, Segment(Observation(0.0, self.dummy_pose, point_cloud=sphere1), None, 0, 0.05))
        obs2 = Observation(0.0, self.dummy_pose, point_cloud=sphere2)

        # Calculate the IOU and assert its what we expect
        iou, enc_seg_ratio, enc_obs_ratio = SceneGraph3D.convex_hull_geometric_overlap(node1, obs2)
        np.testing.assert_almost_equal(iou, 0.055555556, decimal=2)
        np.testing.assert_almost_equal(enc_seg_ratio, 0.5, decimal=3)
        np.testing.assert_almost_equal(enc_obs_ratio, 1/16, decimal=3)

    def test_semantic_consistency(self):
        # Define test embeddings
        emb1 = np.array([3, 4, -5])
        emb2 = np.array([-1, 0, -1])

        # Create a test segment and a test observation
        node1 = LeafGraphNode(0, None, False, Segment(Observation(0.0, self.dummy_pose), None, 0, 0.05))
        node1.get_segment()._add_semantic_descriptor([(emb1, 10)])
        obs2 = Observation(0.0, self.dummy_pose, clip_embedding=emb2)

        # Calculate the semantic consistency
        np.testing.assert_equal(SceneGraph3D.semantic_consistency(node1, obs2, rescaling=[0.7, 1.0]), 0)
        np.testing.assert_equal(SceneGraph3D.semantic_consistency(node1, obs2, rescaling=[0.2, 1.0]), 0)
        np.testing.assert_equal(SceneGraph3D.semantic_consistency(node1, obs2, rescaling=[-0.5, 1.0]), 0.7/1.5)
        np.testing.assert_equal(SceneGraph3D.semantic_consistency(node1, obs2, rescaling=[-0.5, 0.2]), 1)


if __name__ == "__main__":
    unittest.main()