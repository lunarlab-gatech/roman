import math
import numpy as np
import os
from pathlib import Path
from roman.scene_graph.hull_methods import *
import trimesh
import unittest

class TestHullMethods(unittest.TestCase):

    def setUp(self):
        """ Build shapes that are useful in multiple tests."""

        self.cube_1 = self.helper_generate_cube_surface_points()
        self.cube_2 = self.helper_generate_cube_surface_points(shift=0.5)
        self.cube_3 = self.helper_generate_cube_surface_points(shift=2)
        self.cube_4 = self.helper_generate_cube_surface_points(cube_max=3)
        self.cube_5 = self.helper_generate_cube_surface_points(cube_min=0.25, cube_max=0.75)

        self.hull_1 = get_convex_hull_from_point_cloud(self.cube_1)
        self.hull_2 = get_convex_hull_from_point_cloud(self.cube_2)
        self.hull_3 = get_convex_hull_from_point_cloud(self.cube_3)
        self.hull_4 = get_convex_hull_from_point_cloud(self.cube_4)

        np.random.seed(4215)

    def helper_generate_cube_surface_points(self, N=1000, cube_min=0.0, cube_max=1.0, shift=0):
        """ Generate point clouds sampled on cube surfaces. """

        # Initialize array to hold points
        points = np.zeros((N, 3))

        # Randomly choose one of 6 faces for each point
        face_ids = np.random.randint(0, 6, size=N) 

        # Generate points
        for i in range(N):
            face = face_ids[i]
            point = np.random.uniform(cube_min, cube_max, size=3)
            if face == 0: point[0] = cube_min
            elif face == 1: point[0] = cube_max
            elif face == 2: point[1] = cube_min
            elif face == 3: point[1] = cube_max
            elif face == 4: point[2] = cube_min
            elif face == 5: point[2] = cube_max
            points[i] = point

        # Shift points 
        points += np.array([shift, shift, shift])
        return points
    
    def test_get_convex_hull_from_point_cloud(self):
        """ Make sure Convex Hull properly returns None when point cloud is invalid. """

        test_pc1 = np.random.randint(0, 10, (1, 3))
        np.testing.assert_equal(get_convex_hull_from_point_cloud(test_pc1), None)

        test_pc2 = np.random.randint(0, 10, (3, 3))
        np.testing.assert_equal(get_convex_hull_from_point_cloud(test_pc2), None)

        test_pc3 = np.random.randint(0, 10, (4, 3))
        self.assertIsInstance(get_convex_hull_from_point_cloud(test_pc3), trimesh.Trimesh)

        test_pc4 = np.random.randint(0, 10, (400, 3))
        self.assertIsInstance(get_convex_hull_from_point_cloud(test_pc4), trimesh.Trimesh)

    def test_find_point_overlap_with_hulls(self):
        """ Make sure we get all points within a bound, and that we fail when expected. """

        # Test with no overlap
        contain_masks = find_point_overlap_with_hulls(self.cube_3, [self.hull_1])
        np.testing.assert_equal(0, contain_masks.sum())

        # Test with partial overlap
        test_pc = np.array([[-1,  0, 0],
                            [ 0, -1, 0],
                            [0.5, 0.5, 0.4],
                            [0.6, 0.6, 0.75]], dtype=np.float64)
        contain_masks = find_point_overlap_with_hulls(test_pc, [self.hull_1, self.hull_2])
        expected_masks = np.array([[0, 0, 1, 1],
                                   [0, 0, 0, 1]], dtype=np.float64)
        np.testing.assert_array_equal(contain_masks, expected_masks)

        # Test when fail on multi-assign is True
        with np.testing.assert_raises(RuntimeError):
            contain_masks = find_point_overlap_with_hulls(self.cube_5, [self.hull_1, self.hull_4], True)

    def test_convex_hull_geometric_overlap(self):
        """ Make sure IOU and overlaps are calculated correctly. """    

        np.testing.assert_almost_equal(convex_hull_geometric_overlap(self.hull_1, self.hull_2)[0], 0.066666667, 3)
        np.testing.assert_almost_equal(convex_hull_geometric_overlap(self.hull_1, self.hull_2)[1], 1/8, 3)
        np.testing.assert_almost_equal(convex_hull_geometric_overlap(self.hull_1, self.hull_2)[2], 1/8, 3)

        np.testing.assert_almost_equal(convex_hull_geometric_overlap(self.hull_1, None)[0], 0, 14)
        np.testing.assert_almost_equal(convex_hull_geometric_overlap(self.hull_1, None)[1], 1, 14)
        np.testing.assert_almost_equal(convex_hull_geometric_overlap(self.hull_1, None)[2], 1, 14)

        with np.testing.assert_raises(ValueError):
            _, _, _ = convex_hull_geometric_overlap(None, None)

    def test_shortest_dist_between_convex_hulls(self):
        """ Make sure the short distance calculation is reasonable. """

        # TODO: This turned out to be broken, even with these tests, so add a couple more to be more thorough.

        np.testing.assert_almost_equal(shortest_dist_between_convex_hulls(self.hull_1, self.hull_2), 0, 1)
        np.testing.assert_almost_equal(shortest_dist_between_convex_hulls(self.hull_1, self.hull_3), math.sqrt(3), 1)

    def test_longest_line_of_convex_hull(self):
        """ Make sure we can get the longest inner dimension. """

        np.testing.assert_almost_equal(longest_line_of_convex_hull(self.hull_1), math.sqrt(3), 1)
        np.testing.assert_almost_equal(longest_line_of_convex_hull(self.hull_4), 3 * math.sqrt(3) - 0.1, 1)

if __name__ == "__main__":
    unittest.main()