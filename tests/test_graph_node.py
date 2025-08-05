import numpy as np
from pathlib import Path
import unittest
from roman.scene_graph.graph_node import GraphNode
from typeguard import typechecked

# Set random seeds
np.random.seed(42)

class TestGraphNode(unittest.TestCase):

    I = np.zeros((4, 4))

    @typechecked
    def test_get_weighted_semantic_descriptor(self):
        """ Ensure our calculations of semantic descriptors among parents/children is sound"""

        # Define some example descriptors w/weights
        des1 = [(np.array([1, 4, 6]), 10)]
        des2 = [(np.array([-1, 40, 3]), 5),
                (np.array([23, -4, 20]), 2)]
        des3 = [(np.array([5, 10, -20]), 20)]

        # Build an example graph with above descriptors
        root_node = GraphNode.create_node_if_possible(0, None, [], np.zeros((0, 3), dtype=np.float64), [], 0, 0, 0, np.empty(0), np.empty(0), True)
        parent_node = GraphNode.create_node_if_possible(1, root_node, des1, np.random.randint(0, 10, (30, 3)), [], 0, 0, 0, self.I, self.I, False)
        child1_node = GraphNode.create_node_if_possible(2, parent_node, des2, np.random.randint(0, 5, (30, 3)), [], 0, 0, 0, self.I, self.I, False)
        child2_node = GraphNode.create_node_if_possible(3, parent_node, des3, np.random.randint(5, 10, (30, 3)), [], 0, 0, 0, self.I, self.I, False)
        parent_node.add_children([child1_node, child2_node])

        assert parent_node is not None
        assert child1_node is not None
        assert child2_node is not None

        # Assert the weighted descriptors calculated are what we expect
        with np.testing.assert_raises(RuntimeError):
            _ = root_node.get_weighted_semantic_descriptor()
        np.testing.assert_array_almost_equal(child2_node.get_weighted_semantic_descriptor(), [0.218217890235992, 0.436435780471985, -0.872871560943969], 15)
        np.testing.assert_array_almost_equal(child1_node.get_weighted_semantic_descriptor(), [0.263969542944024, 0.909079489470806, 0.322326794141320], 15)
        np.testing.assert_array_almost_equal(parent_node.get_weighted_semantic_descriptor(), [0.329206024579985, 0.877329854506504, -0.349164316292257], 15)


if __name__ == "__main__":
    unittest.main()