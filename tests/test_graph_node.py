import numpy as np
from pathlib import Path
import pickle
import unittest
from roman.scene_graph.graph_node import RootGraphNode
from typeguard import typechecked

class TestGraphNode(unittest.TestCase):

    @typechecked
    def test_prune_dummy_nodes(self):
        """ Ensure that any child loops aren't broken by child alterations during prune step. """

        # Load the two root nodes from pickle files
        with open((Path('.') / "tests" / "files" / "tpdnBefore.pkl").absolute(), 'rb') as file:
            before_root_node: RootGraphNode = pickle.load(file)
        with open((Path('.') / "tests" / "files" / "tpdnAfter.pkl").absolute(), 'rb') as file:
            after_root_node: RootGraphNode = pickle.load(file)

        # Run the prune nodes process on the before node
        before_root_node.prune_dummy_nodes()

        # Naively check that it is working by making sure the number of nodes in the graph are the same
        # TODO: Make this more robust by ensuring the graphs match exactly
        np.testing.assert_equal(before_root_node.get_number_of_nodes(), after_root_node.get_number_of_nodes())

if __name__ == "__main__":
    unittest.main()