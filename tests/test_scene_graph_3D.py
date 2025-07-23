import numpy as np
from pathlib import Path
from roman.scene_graph.scene_graph_3D import SceneGraph3D
import unittest
from typeguard import typechecked

class TestSceneGraph3D(unittest.TestCase):

    @typechecked
    def test_pass_minimum_requirements_for_association(self):
        """ Make sure that our logic is sound for association requirements. """

        graph = SceneGraph3D(np.zeros((4, 4)))

        iou_pass = 0.5 * (1 + graph.min_iou_for_association)
        sem_pass = 0.5 * (1 + graph.min_sem_con_for_association)
        iou_fail = 0.5 * graph.min_iou_for_association
        sem_fail = 0.5 * graph.min_sem_con_for_association

        np.testing.assert_equal(graph.pass_minimum_requirements_for_association(iou_pass, sem_pass), True)
        with np.testing.assert_raises(RuntimeError):
            _ = graph.pass_minimum_requirements_for_association(iou_pass, sem_fail)
        np.testing.assert_equal(graph.pass_minimum_requirements_for_association(iou_fail, sem_pass), False)
        np.testing.assert_equal(graph.pass_minimum_requirements_for_association(iou_fail, sem_fail), False)

if __name__ == "__main__":
    unittest.main()