import numpy as np
from pathlib import Path
from roman.params.fastsam_params import FastSAMParams
from roman.params.scene_graph_3D_params import SceneGraph3DParams, GraphNodeParams
from roman.scene_graph.scene_graph_3D import SceneGraph3D
import unittest
from typeguard import typechecked

class TestSceneGraph3D(unittest.TestCase):

    @typechecked
    def test_pass_minimum_requirements_for_association(self):
        """ Make sure that our logic is sound for association requirements. """

        # Load parameters & create the graph
        files_path = Path(__file__).parent / "files" / "test_scene_graph_3D" 
        scene_graph_3D_params = SceneGraph3DParams.from_yaml(files_path / "scene_graph_3D.yaml")
        node_params = GraphNodeParams.from_yaml(files_path / "graph_node.yaml")
        fastsam_params = FastSAMParams.from_yaml(files_path / "fastsam.yaml")
        graph = SceneGraph3D(scene_graph_3D_params, node_params, np.zeros((4, 4)), fastsam_params)

        iou_pass = 0.5 * (1 + graph.params.min_iou_for_association)
        iou_fail = 0.5 * graph.params.min_iou_for_association

        np.testing.assert_equal(graph.pass_minimum_requirements_for_association(iou_pass), True)
        np.testing.assert_equal(graph.pass_minimum_requirements_for_association(iou_fail), False)

if __name__ == "__main__":
    unittest.main()