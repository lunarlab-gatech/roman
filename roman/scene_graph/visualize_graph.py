import pyvista as pv
import numpy as np
import colorsys
from .graph_node import GraphNode
import random

class SceneGraphViewer:
    def __init__(self):
        self.plotter = pv.Plotter(window_size=[1280, 720])
        self.plotter.set_background("white")
        self._window_open = False

    def update(self, nodes: list[GraphNode]):
        self.plotter.clear()  # Remove all actors before redraw

        def hsv_shift(hsv, h_shift=0.1, s_shift=0.0, v_shift=-0.1):
            h = (hsv[0] + h_shift) % 1.0
            s = min(max(hsv[1] + s_shift, 0.0), 1.0)
            v = min(max(hsv[2] + v_shift, 0.0), 1.0)
            return (h, s, v)

        def traverse(node: GraphNode, parent_centroid=None, parent_hsv=None):
            node_id = node.get_id()

            points = node.get_point_cloud()
            if points.shape[0] < 4 or np.isnan(points).any():
                print(f"Skipping node {node_id} with invalid point cloud.")
                return

            hull = node.get_convex_hull()
            if hull is None or len(hull.faces) == 0:
                print(f"Skipping node {node_id} with invalid hull.")
                return

            hsv = hsv_shift(parent_hsv) if parent_hsv else (random.random(), 0.6, 0.9)
            rgb = colorsys.hsv_to_rgb(*hsv)
            centroid = np.mean(points, axis=0)

            try:
                faces = np.hstack(
                    [np.full((len(hull.faces), 1), 3), hull.faces]
                ).astype(np.int64).flatten()

                mesh = pv.PolyData(hull.vertices, faces)
                mesh.compute_normals(inplace=True)
                self.plotter.add_mesh(mesh, color=rgb, opacity=0.25, show_edges=True)
            except Exception as e:
                print(f"Failed to create mesh for node {node_id}: {e}")

            if parent_centroid is not None:
                try:
                    line = pv.Line(parent_centroid, centroid)
                    self.plotter.add_mesh(line, color=rgb, line_width=2)
                except Exception as e:
                    print(f"Failed to create edge for node {node_id}: {e}")

            try:
                label_text = str(node_id)
                self.plotter.add_point_labels(
                    [centroid],
                    [label_text],
                    text_color=rgb,
                    point_color=None,
                    font_size=24,
                    shape_opacity=0.0,
                    always_visible=True,
                )
            except Exception as e:
                print(f"Failed to add label for node {node_id}: {e}")

            for child in node.get_children():
                traverse(child, centroid, hsv)

        for node in nodes:
            traverse(node)

        self.plotter.reset_camera()
        if not self._window_open:
            self.plotter.show(auto_close=False, interactive_update=True)
            self._window_open = True
        else:
            self.plotter.update()
