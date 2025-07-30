from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from .graph_node import RootGraphNode, GraphNode
import numpy as np
import rerun as rr
from typeguard import typechecked

@dataclass
@typechecked
class HSVSpace:
    h_range: tuple[float, float]
    s_range: tuple[float, float]
    v_range: tuple[float, float]

    def center(self) -> tuple[float, float, float]:
        h = (self.h_range[0] + self.h_range[1]) / 2 % 1.0
        s = (self.s_range[0] + self.s_range[1]) / 2
        v = (self.v_range[0] + self.v_range[1]) / 2
        return h, s, v

    def split(self, n: int, axis: int) -> list[HSVSpace]:
        """Split this HSV space into `n` parts along the given axis (0=H, 1=S, 2=V)."""
        if axis == 0:
            r0, r1 = self.h_range
        elif axis == 1:
            r0, r1 = self.s_range
        else:
            r0, r1 = self.v_range

        step = (r1 - r0) / n
        subspaces = []
        for i in range(n):
            split_range = (r0 + i * step, r0 + (i + 1) * step)
            if axis == 0:
                subspaces.append(HSVSpace(split_range, self.s_range, self.v_range))
            elif axis == 1:
                subspaces.append(HSVSpace(self.h_range, split_range, self.v_range))
            else:
                subspaces.append(HSVSpace(self.h_range, self.s_range, split_range))
        return subspaces

@typechecked
class RerunWrapper():
    """ Wrapper for spawning and visualizing using Rerun. """

    def __init__(self):
        rr.init("Meronomy_Visualization", spawn=True)
        self.update_frame = 0

    def _hsv_to_rgb255(self, h: float, s: float = 1.0, v: float = 1.0) -> np.ndarray[np.uint8]:
        """ Output will be integers 0-255 to be compatible with Rerun """
        
        def clamp(val: float, min_val=0.0, max_val=1.0) -> float:
            return max(min_val, min(max_val, val))

        r, g, b = colorsys.hsv_to_rgb(clamp(h), clamp(s), clamp(v))
        return np.array([[int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]], dtype=np.uint8)

    def _assign_colors_recursive(self, node: GraphNode, hsv_space: HSVSpace,
        node_colors: dict[int, tuple[float, float, float]], depth: int = 0) -> None:

        node_id = node.get_id()
        node_colors[node_id] = hsv_space.center()

        children = sorted(node.get_children(), key=lambda c: c.get_id())
        if not children:
            return

        axis = depth % 3  # 0: H, 1: S, 2: V
        subspaces = hsv_space.split(len(children), axis)
        for child, child_space in zip(children, subspaces):
            self._assign_colors_recursive(child, child_space, node_colors, depth + 1)

    def update(self, root_node: RootGraphNode, curr_time: float):
        """
        Args:
            nodes (RootGraphNode]): Takes as input the children of the RootGraphNode.
            curr_time (float): The current time of the camera frame.
        """

        # Update the timelines
        rr.set_time("camera_frame_time", timestamp=curr_time)
        rr.set_time("update_frame_tick", sequence=self.update_frame)
        self.update_frame += 1

        # Assign/refresh color intervals starting at the root.
        full_hsv = HSVSpace((0.0, 1.0), (0.2, 1.0), (0.2, 1.0))
        node_colors: dict[int, tuple[float, float, float]] = {}
        self._assign_colors_recursive(root_node, full_hsv, node_colors)

        # Create structures to store graph information
        node_ids: list[int] = []
        edges: list[tuple[int, int]] = []
        colors_rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)

        # Iterate through all nodes reachable from the root.
        for node in root_node:
            # Extract id
            id = node.get_id()
            node_ids.append(id)

            # Add edges
            if not node.is_RootGraphNode():
                parent = node.get_parent()
                edges.append((id, parent.get_id()))
            for child in node.get_children():
                edges.append((id, child.get_id()))

            # Colors
            h, s, v = node_colors[id]
            colors_rgb = np.concatenate((colors_rgb, self._hsv_to_rgb255(h, s, v)), dtype=np.uint8)

        # Remove all edges related to root node, so it doesn't appear in visualization
        node_ids.remove(root_node.get_id())
        to_remove = []
        for i, edge in enumerate(edges):
            if edge[0] == root_node.get_id() or edge[1] == root_node.get_id():
                to_remove.append(i)
        to_remove.sort(reverse=True)
        for i in to_remove:
            edges.pop(i)

        # Send the data to Rerun
        rr.log("graph_connectivity",
            rr.GraphNodes(node_ids=node_ids, labels=node_ids, colors=colors_rgb),
            rr.GraphEdges(edges=edges, graph_type="directed"),
        )