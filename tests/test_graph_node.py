import unittest

class TestGraphNode(unittest.TestCase):

    def test_prune_dummy_nodes(self):
        """ Ensure that any child loops aren't broken by child alterations during prune step. """

        pass

if __name__ == "__main__":
    unittest.main()