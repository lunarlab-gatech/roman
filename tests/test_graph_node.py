import copy
import numpy as np
from pathlib import Path
import unittest
from roman.map.voxel_grid import VoxelGrid
from roman.scene_graph.graph_node import GraphNode
from roman.params.scene_graph_3D_params import GraphNodeParams
from typeguard import typechecked

# Set random seeds
np.random.seed(42)

@typechecked
class TestGraphNode(unittest.TestCase):

    I = np.zeros((4, 4))

    def helper_create_dummy_node(self, id: int = 0, parent_node = None, is_RootGraphNode: bool = False) -> GraphNode:
        """ Helper method that creates a dummy node with no values"""
        node = GraphNode.create_node_if_possible(id, parent_node, np.zeros(0), np.zeros((0, 3)), [], is_RootGraphNode=is_RootGraphNode)
        assert node is not None, "Node failed creation!"
        return node
    
    def test_create_node_if_possible(self):
        """ Make sure that node creation fails if convex hull is required but invalid """

        GraphNode.params = GraphNodeParams.get_default_for_test_cases()

        # Create a valid node
        valid_points = np.random.uniform(low=0, high=1, size=(100, 3))
        valid_node = GraphNode.create_node_if_possible(0, None, np.zeros(0), valid_points, [])
        self.assertIsNotNone(valid_node)

        # Create a node with no points, but should still succeed as we don't require valid hull
        no_points = np.zeros((0, 3))
        no_points_node = GraphNode.create_node_if_possible(1, None, np.zeros(0), no_points, [])
        self.assertIsNotNone(no_points_node)

        # Now, require valid convex hull and try to create invalid node
        GraphNode.params.require_valid_convex_hull = True

        invalid_points = np.array([[0,0,0], [0,0,0], [0,0,0]])
        invalid_node = GraphNode.create_node_if_possible(2, None, np.zeros(0), invalid_points, [])
        self.assertIsNone(invalid_node)

        # Make sure we can still create valid node
        another_valid_node = GraphNode.create_node_if_possible(3, None, np.zeros(0), valid_points, [])
        self.assertIsNotNone(another_valid_node)

    def test_get_id_and_set_id(self):
        """ Ensure we get the correct node ID """

        GraphNode.params = GraphNodeParams.get_default_for_test_cases()

        node = self.helper_create_dummy_node(5)
        self.assertEqual(node.get_id(), 5)
        node2 = self.helper_create_dummy_node(10)
        self.assertEqual(node2.get_id(), 10)

        node2.set_id(15)
        self.assertEqual(node2.get_id(), 15)

    def test_parent_methods(self):
        """ Test get_parent, is_parent, and set_parent methods """

        GraphNode.params = GraphNodeParams.get_default_for_test_cases()

        parent_node = self.helper_create_dummy_node(0, is_RootGraphNode=True)
        child_node = self.helper_create_dummy_node(1, parent_node=parent_node)
        parent_node.add_child(child_node)

        with self.assertRaises(RuntimeError):
            _ = parent_node.get_parent() # Is RootGraphNode, should error
        self.assertEqual(child_node.get_parent(), parent_node)

        self.assertTrue(parent_node.is_parent(child_node))
        self.assertFalse(child_node.is_parent(parent_node))

        self.assertTrue(child_node.is_child(parent_node))
        self.assertFalse(parent_node.is_child(child_node))

        self.assertTrue(parent_node.is_parent_or_child(child_node))
        self.assertTrue(child_node.is_parent_or_child(parent_node))

        # Disconnect parent/child and re-test
        new_parent_node = self.helper_create_dummy_node(2, is_RootGraphNode=True)
        child_node.set_parent(new_parent_node)
        parent_node.remove_child(child_node)
        new_parent_node.add_child(child_node)

        self.assertEqual(child_node.get_parent(), new_parent_node)
        self.assertTrue(new_parent_node.is_parent_or_child(child_node))
        self.assertTrue(new_parent_node.is_parent(child_node))

        self.assertFalse(parent_node.is_parent(child_node))
        self.assertFalse(child_node.is_parent_or_child(parent_node))

    def test_is_sibling(self):
        """ Test is_sibling method """

        GraphNode.params = GraphNodeParams.get_default_for_test_cases()

        parent_node = self.helper_create_dummy_node(0, is_RootGraphNode=True)
        child_node_1 = self.helper_create_dummy_node(1, parent_node=parent_node)
        child_node_2 = self.helper_create_dummy_node(2, parent_node=parent_node)
        parent_node.add_children([child_node_1, child_node_2])

        self.assertTrue(child_node_1.is_sibling(child_node_2))
        self.assertTrue(child_node_2.is_sibling(child_node_1))

        parent_of_non_sibling_node = self.helper_create_dummy_node(4, is_RootGraphNode=False)
        non_sibling_node = self.helper_create_dummy_node(3, parent_node=parent_of_non_sibling_node)
        parent_of_non_sibling_node.add_child(non_sibling_node)

        self.assertFalse(child_node_1.is_sibling(non_sibling_node))
        self.assertFalse(child_node_2.is_sibling(non_sibling_node))

        # Create RootGraphNode and ensure no siblings
        root_node = self.helper_create_dummy_node(4, is_RootGraphNode=True)
        self.assertFalse(root_node.is_sibling(child_node_1))
        self.assertFalse(root_node.is_sibling(child_node_2))

    def test_is_descendent_or_ascendents(self):
        GraphNode.params = GraphNodeParams.get_default_for_test_cases()

        root_node = self.helper_create_dummy_node(0, is_RootGraphNode=True)
        child_node = self.helper_create_dummy_node(1, parent_node=root_node)
        root_node.add_child(child_node)
        grandchild_node = self.helper_create_dummy_node(2, parent_node=child_node)
        child_node.add_child(grandchild_node)
        other_child_node = self.helper_create_dummy_node(3, parent_node=root_node)
        root_node.add_child(other_child_node)

        self.assertTrue(root_node.is_ascendent(grandchild_node))
        self.assertTrue(root_node.is_ascendent(child_node))
        self.assertTrue(child_node.is_ascendent(grandchild_node))

        self.assertFalse(grandchild_node.is_ascendent(child_node))
        self.assertFalse(grandchild_node.is_ascendent(root_node))
        self.assertFalse(child_node.is_ascendent(root_node))

        self.assertFalse(child_node.is_ascendent(other_child_node))
        self.assertFalse(grandchild_node.is_ascendent(other_child_node))
        self.assertFalse(other_child_node.is_ascendent(child_node))
        self.assertFalse(other_child_node.is_ascendent(grandchild_node))
        self.assertTrue(root_node.is_ascendent(other_child_node))

        self.assertTrue(grandchild_node.is_descendent_or_ascendent(root_node))
        self.assertTrue(child_node.is_descendent_or_ascendent(root_node))
        self.assertTrue(grandchild_node.is_descendent_or_ascendent(child_node))
        self.assertTrue(root_node.is_descendent_or_ascendent(other_child_node))
        self.assertFalse(other_child_node.is_descendent_or_ascendent(child_node))
        self.assertFalse(other_child_node.is_descendent_or_ascendent(grandchild_node))
        
    def test_voxel_grid_methods(self):
        """ Test get_voxel_grid and get_num_points """

        GraphNode.params = GraphNodeParams.get_default_for_test_cases()

        points = np.random.uniform(low=0, high=1, size=(100, 3))
        node: GraphNode | None = GraphNode.create_node_if_possible(0, None, np.zeros(0), points, [])
        assert node is not None

        np.testing.assert_array_equal(node.get_point_cloud(), points)

        voxel_grid = node.get_voxel_grid(0.05)
        self.assertTrue(isinstance(voxel_grid, VoxelGrid))

        self.assertEqual(node.get_num_points(), points.shape[0])

        root_graph_node = self.helper_create_dummy_node(1, is_RootGraphNode=True)
        self.assertEqual(root_graph_node.get_num_points(), 0)
        self.assertEqual(root_graph_node.get_voxel_grid(0.05), None)

    def test_get_all_descendents(self):
        """ Ensure we successfully get all descendents """
        yaml_file = Path(__file__).parent / "files" / "test_graph_node" / "get_all_descendents.yaml"
        GraphNode.params = GraphNodeParams.from_yaml(yaml_file)

        nodes: list[GraphNode] = []
        for i in range(0, 10):
            nodes.append(self.helper_create_dummy_node(i))

        nodes[0].add_children(nodes[1:3])
        nodes[1].add_child(nodes[3])
        nodes[2].add_children(nodes[4:6])

        np.testing.assert_array_equal(nodes[0].get_all_descendents(), nodes[1:6])
        np.testing.assert_array_equal(nodes[1].get_all_descendents(), nodes[3])

    def test_update_point_cloud(self):
        """ Make sure the point cloud is actually changed by this method. """

        # TODO: Add tests to ensure the functionality actually works fully.

        yaml_file = Path(__file__).parent / "files" / "test_graph_node" / "get_semantic_descriptor.yaml"
        GraphNode.params = GraphNodeParams.from_yaml(yaml_file)

        # Create a random point cloud w/outliers
        points = np.random.uniform(low=0, high=1, size=(1000, 3))
        outliers = np.random.uniform(low=8, high=12, size=(3, 3))
        point_cloud = np.vstack([points, outliers])

        # Generate a graph node with point cloud (runs update_point_cloud)
        node = GraphNode.create_node_if_possible(0, None, np.zeros(0), point_cloud.copy(), [], False)
        curr_cloud = node.get_point_cloud().copy()

        # Generate another cloud and update
        new_points = np.random.normal(loc=5, scale=1, size=(1000, 3))
        node.update_point_cloud(new_points)
        newest_cloud = node.get_point_cloud().copy()

        # Make sure the cloud has changed
        self.assertNotEqual(curr_cloud.shape[0], newest_cloud.shape[0])

    def test_iterator(self):
        """ Ensure iterator returns all nodes and in the same order each time (assuming no changes). """

        GraphNode.params = GraphNodeParams.get_default_for_test_cases()

        # Create a simple tree
        root_node = self.helper_create_dummy_node(0, is_RootGraphNode=True)
        child_node_1 = self.helper_create_dummy_node(1, parent_node=root_node)
        child_node_2 = self.helper_create_dummy_node(2, parent_node=root_node)
        root_node.add_children([child_node_1, child_node_2])

        grandchild_node_1 = self.helper_create_dummy_node(3, parent_node=child_node_1)
        grandchild_node_2 = self.helper_create_dummy_node(4, parent_node=child_node_1)
        child_node_1.add_children([grandchild_node_1, grandchild_node_2])

        # Get nodes via iterator
        iterated_nodes: list[GraphNode] = []
        for node in root_node:
            iterated_nodes.append(node)
        
        # Make sure it gets all nodes
        iterated_nodes_copy = copy.deepcopy(iterated_nodes)
        expected_nodes = [root_node, child_node_1, grandchild_node_1, grandchild_node_2, child_node_2]
        np.testing.assert_array_equal(iterated_nodes_copy.sort(key=lambda x: x.get_id()), expected_nodes.sort(key=lambda x: x.get_id()))
        np.testing.assert_equal(len(iterated_nodes_copy), 5)

        # Do it again to make sure order is the same
        iterated_nodes_2 = []
        for node in root_node:
            iterated_nodes_2.append(node)

        # Print nodes in each of the lists below for debugging
        np.testing.assert_array_equal(iterated_nodes, iterated_nodes_2)

if __name__ == "__main__":
    unittest.main()