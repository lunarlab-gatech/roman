import math
import numpy as np
from roman.scene_graph.scene_graph_utils import *
import trimesh
import unittest

class TestSceneGraphUtils(unittest.TestCase):

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

        test_pc1 = np.random.randint(0, 10, (1, 3)).astype(float)
        np.testing.assert_equal(get_convex_hull_from_point_cloud(test_pc1), None)

        test_pc2 = np.random.randint(0, 10, (3, 3)).astype(float)
        np.testing.assert_equal(get_convex_hull_from_point_cloud(test_pc2), None)

        test_pc3 = np.random.randint(0, 10, (4, 3)).astype(float)
        self.assertIsInstance(get_convex_hull_from_point_cloud(test_pc3), trimesh.Trimesh)

        test_pc4 = np.random.randint(0, 10, (400, 3)).astype(float)
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

        # TODO: This turned out to be broken previously, even with these tests, so add a couple more to be more thorough.

        np.testing.assert_almost_equal(shortest_dist_between_convex_hulls(self.hull_1, self.hull_2), 0, 1)
        np.testing.assert_almost_equal(shortest_dist_between_convex_hulls(self.hull_1, self.hull_3), math.sqrt(3), 1)

    def test_longest_line_of_point_cloud(self):
        """ Make sure we can get the longest inner dimension. """

        np.testing.assert_almost_equal(longest_line_of_point_cloud(self.cube_1), math.sqrt(3), 1)
        np.testing.assert_almost_equal(longest_line_of_point_cloud(self.cube_4), 3 * math.sqrt(3) - 0.1, 1)

    def test_fix_winding_and_calculate_normals(self):
        """ Ensure windings are properly flipped and normals calculated correctly. """

        # Define simple point cloud for a regular tetrahedron
        pc = np.array([[  0,                0,            0],
                       [  1,                0,            0],
                       [0.5, 1/(2*np.sqrt(3)), np.sqrt(2/3)],
                       [0.5,     3/np.sqrt(2),            0]])
        faces = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])

        # This shouldn't be a volume
        mesh = trimesh.Trimesh(vertices=pc, faces=faces)
        np.testing.assert_equal(mesh.is_volume, False)

        # Call our method to fix the winding and normals
        faces, face_normals = fix_winding_and_calculate_normals(pc, faces)

        # Try again to create a trimesh, this should be a volume
        mesh = trimesh.Trimesh(vertices=pc, faces=faces, face_normals=face_normals)
        np.testing.assert_equal(mesh.is_volume, True)

        # Check that (some of) the calculated values match those calculated by hand
        des_faces = np.array([[0, 1, 2], [1, 3, 2]])
        des_normals_unnormalized = np.array([[0, -np.sqrt(2/3),  1/(2*np.sqrt(3))],
                                             [1.7321, 0.40825, 0.91632]])
        des_normals = des_normals_unnormalized / np.linalg.norm(des_normals_unnormalized, axis=1, keepdims=True)
        np.testing.assert_array_equal(faces[0:2], des_faces)
        np.testing.assert_array_almost_equal(face_normals[0:2], des_normals, decimal=5)

    def test_merge_overlapping_sets(self):
        # Generate example list of sets
        A = [{0, 1, 5}, {2, 3}, {3, 4}, {5, 2}]
        A_out_exp = [{0, 1, 5, 2, 3, 4}]

        B = [{1, 3}, {0, 4}, {5, 6}, {3, 12}, {24, 5}]
        B_out_exp = [{1, 3, 12}, {0, 4}, {5, 6, 24}]

        # Run the method 
        A_out = merge_overlapping_sets(A)
        B_out = merge_overlapping_sets(B)
        np.testing.assert_array_equal(A_out_exp, A_out)
        np.testing.assert_array_equal(B_out_exp, B_out)

    def test_merge_overlapping_holonyms(self):
        """ Ensure that we can merge overlapping holonyms correctly. """

        input_detected_holonyms: list[tuple[set[int], set[str]]] = \
            [({7, 26}, {'automotive vehicle', 'motor vehicle'}), 
             ({9, 26}, {'automotive vehicle', 'motor vehicle'}), 
             ({13, 26}, {'automotive vehicle', 'motor vehicle'})]
        
        output_detected_holonyms_exp: list[tuple[set[int], set[str]]] = \
            [({7, 9, 13, 26}, {'automotive vehicle', 'motor vehicle'})]
        
        output_detected_holonyms: list[tuple[set[int], set[str]]] = merge_overlapping_holonyms(input_detected_holonyms)

        for tup in output_detected_holonyms_exp:
            self.assertIn(tup, output_detected_holonyms)

    def test_merge_objs_via_function(self):
        # Generate example list of sets
        A = [{0, 1, 5}, {2, 3}, {3, 4}, {5, 2}]
        A_out_exp = sorted([{0, 1, 5, 2, 3, 4}], key=lambda s: min(s))

        B = [{1, 3}, {0, 4}, {5, 6}, {3, 12}, {24, 5}]
        B_out_exp = sorted([{1, 3, 12}, {0, 4}, {5, 6, 24}], key=lambda s: min(s))

        # Generate merge function
        def merge_func(x: set[int], y: set[int]) -> set[int] | None:
            if x & y:
                x |= y
                return x
            return None
        
        # Run the test
        A_out = sorted(merge_objs_via_function(A, merge_func), key=lambda s: min(s))
        B_out = sorted(merge_objs_via_function(B, merge_func), key=lambda s: min(s))
        np.testing.assert_array_equal(A_out_exp, A_out)
        np.testing.assert_array_equal(B_out_exp, B_out)

if __name__ == "__main__":
    unittest.main()