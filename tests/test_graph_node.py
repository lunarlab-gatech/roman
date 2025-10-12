import copy
import numpy as np
from pathlib import Path
import unittest
from roman.scene_graph.graph_node import GraphNode
from roman.params.scene_graph_3D_params import GraphNodeParams
from typeguard import typechecked

# Set random seeds
np.random.seed(42)

@typechecked
class TestGraphNode(unittest.TestCase):

    I = np.zeros((4, 4))

    def helper_create_dummy_node(self, id: int = 0) -> GraphNode:
        """ Helper method that creates a dummy node with no values"""
        node = GraphNode.create_node_if_possible(id, None, [], None, 0, np.zeros((0, 3)), [], 0, 0, 0, self.I, self.I, self.I)
        assert node is not None, "Node failed creation!"
        return node

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

    def test_get_semantic_descriptor(self):
        """ 
        Ensure our calculations of semantic descriptors among parents/children is sound
        
        NOTE: This only tests for descriptor calculated using children, and not calculated
        incrementally, observation descriptor, or without children.
        """

        yaml_file = Path(__file__).parent / "files" / "test_graph_node" / "get_semantic_descriptor.yaml"
        GraphNode.params = GraphNodeParams.from_yaml(yaml_file)
        
        # ================ Test Weighted Semantic Descriptor w/children ================
        # Define some example descriptors w/weights
        des1 = [(np.array([1, 4, 6]), 10)]
        des2 = [(np.array([-1, 40, 3]), 5),
                (np.array([23, -4, 20]), 2)]
        des3 = [(np.array([5, 10, -20]), 20)]

        # Build an example graph with above descriptors
        root_node = GraphNode.create_node_if_possible(0, None, [], None, 0, np.zeros((0, 3), dtype=np.float64), [], 0, 0, 0, np.empty(0), np.empty(0), np.empty(0), None, True)
        parent_node = GraphNode.create_node_if_possible(1, root_node, des1, None, 0, np.random.randint(0, 10, (30, 3)), [], 0, 0, 0, self.I, self.I, self.I, False)
        child1_node = GraphNode.create_node_if_possible(2, parent_node, des2, None, 0, np.random.randint(0, 5, (30, 3)), [], 0, 0, 0, self.I, self.I, self.I, False)
        child2_node = GraphNode.create_node_if_possible(3, parent_node, des3, None, 0, np.random.randint(5, 10, (30, 3)), [], 0, 0, 0, self.I, self.I, self.I, False)
        parent_node.add_children([child1_node, child2_node])

        assert root_node is not None
        assert parent_node is not None
        assert child1_node is not None
        assert child2_node is not None

        # Assert the weighted descriptors calculated are what we expect
        with np.testing.assert_raises(RuntimeError):
            _ = root_node.get_semantic_descriptor()
        np.testing.assert_array_almost_equal(child2_node.get_semantic_descriptor(), [0.218217890235992, 0.436435780471985, -0.872871560943969], 15)
        np.testing.assert_array_almost_equal(child1_node.get_semantic_descriptor(), [0.263969542944024, 0.909079489470806, 0.322326794141320], 15)
        np.testing.assert_array_almost_equal(parent_node.get_semantic_descriptor(), [0.329206024579985, 0.877329854506504, -0.349164316292257], 15)

        # ================ Test Unweighted Semantic Descriptor w/children ================
        GraphNode.params.use_weighted_average_for_descriptor = False
        child2_node.reset_saved_descriptor_vars()
        child1_node.reset_saved_descriptor_vars()
        parent_node.reset_saved_descriptor_vars()

        np.testing.assert_array_almost_equal(child2_node.get_semantic_descriptor(), [0.2182178902359924,0.4364357804719848,-0.8728715609439696], 15)
        np.testing.assert_array_almost_equal(child1_node.get_semantic_descriptor(), [0.5390077496416228,0.6459509199615656,0.5405719700715906], 15)
        np.testing.assert_array_almost_equal(parent_node.get_semantic_descriptor(), [0.479886509157352,0.8240845921195751,0.3009875800757971], 15)

        # ================ Test Unweighted Semantic Descriptor without children ================
        GraphNode.params.parent_node_inherits_descriptors_from_children = False
        child2_node.reset_saved_descriptor_vars()
        child1_node.reset_saved_descriptor_vars()
        parent_node.reset_saved_descriptor_vars()
        np.testing.assert_array_almost_equal(child2_node.get_semantic_descriptor(), [0.2182178902359924,0.4364357804719848,-0.8728715609439696], 15)
        np.testing.assert_array_almost_equal(child1_node.get_semantic_descriptor(), [0.5390077496416228,0.6459509199615656,0.5405719700715906], 15)
        np.testing.assert_array_almost_equal(parent_node.get_semantic_descriptor(), [0.13736056394868904,0.5494422557947561,0.8241633836921342], 15)

    def test_get_total_weight_of_semantic_descriptors(self):
        yaml_file = Path(__file__).parent / "files" / "test_graph_node" / "get_semantic_descriptor.yaml"
        GraphNode.params = GraphNodeParams.from_yaml(yaml_file)

        des1 = [(np.array([1, 4, 6]), 10)]
        des2 = [(np.array([-1, 40, 3]), 5),
                (np.array([23, -4, 20]), 2)]
        des3 = [(np.array([5, 10, -20]), 20)]

        root_node = GraphNode.create_node_if_possible(0, None, [], None, 0, np.zeros((0, 3), dtype=np.float64), [], 0, 0, 0, np.empty(0), np.empty(0), np.empty(0), None, True)
        parent_node = GraphNode.create_node_if_possible(1, root_node, des1, None, 0, np.random.randint(0, 10, (30, 3)), [], 0, 0, 0, self.I, self.I, self.I, False)
        child1_node = GraphNode.create_node_if_possible(2, parent_node, des2, None, 0, np.random.randint(0, 5, (30, 3)), [], 0, 0, 0, self.I, self.I, self.I, False)
        child2_node = GraphNode.create_node_if_possible(3, parent_node, des3, None, 0, np.random.randint(5, 10, (30, 3)), [], 0, 0, 0, self.I, self.I, self.I, False)
        parent_node.add_children([child1_node, child2_node])

        np.testing.assert_equal(parent_node.get_total_weight_of_semantic_descriptors(), 37)

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
        node = GraphNode.create_node_if_possible(0, None, [], None, 0, point_cloud.copy(), [], 0, 0, 0, np.empty(0), np.empty(0), np.empty(0),False)
        curr_cloud = node.get_point_cloud().copy()

        # Generate another cloud and update
        new_points = np.random.normal(loc=5, scale=1, size=(1000, 3))
        node.update_point_cloud(new_points, False, False, False)
        newest_cloud = node.get_point_cloud().copy()

        # Make sure the cloud has changed
        self.assertNotEqual(curr_cloud.shape[0], newest_cloud.shape[0])

if __name__ == "__main__":
    unittest.main()