from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from .graph_node import RootGraphNode, GraphNode
import numpy as np
import rerun as rr
from typeguard import typechecked

@typechecked
class Interval():
    _interval: tuple[float, float]
    
    def __init__(self, a: float, b: float):
        """ Assumes b > a """
        assert b > a
        self._interval = (a,b)

    def __iter__(self):
        yield self._interval[0]
        yield self._interval[1]

    def __eq__(self, other: Interval):
        return self._interval[0] == other._interval[0] and self._interval[1] == self._interval[1]

    def get(self) -> tuple[float, float]:
        return self._interval[0], self._interval[1]

    def len(self) -> float:
        return self._interval[1] - self._interval[0]
    
    def midpoint(self) -> float:
        return (self._interval[0] + self._interval[1]) * 0.5
    
    def f(self) -> float:
        return self._interval[0]
    
    def e(self) -> float:
        return self._interval[1]

@typechecked
def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping/adjacent intervals. Assumes arbitrary order."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x.get()[0])
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb:  # overlap or touch
            merged[-1] = Interval(la, max(lb, b))
        else:
            merged.append(Interval(a, b))
    return merged

@dataclass
@typechecked
class ColorAllocator:
    parent_span: dict[int, Interval] = field(default_factory=dict)
    child_spans: dict[int, dict[int, Interval]] = field(default_factory=dict)
    free_spans: dict[int, list[Interval]] = field(default_factory=dict)

    def update_parent(self, parent_id: int, span: Interval) -> None:
        """ If it doesn't exist or it has changed, create a new span. """
        if parent_id not in self.parent_span or self.parent_span[parent_id] != span:
            self.parent_span[parent_id] = span
            self.child_spans[parent_id] = {}
            self.free_spans[parent_id] = [span]

    def _allocate_even(self, parent_id: int, children_ids: list[int]) -> None:
        """Initial even partition among current children (first time only)."""

        # Get number of children
        num_children = len(children_ids)
        assert len(children_ids) != 0

        # Extract interval values
        span = self.parent_span[parent_id]
        a, b = span.get()
        
        # Calculate amount of span per child
        width = span.len() / num_children

        # Reset the children spans and free spans under this child
        self.child_spans[parent_id].clear()
        self.free_spans[parent_id].clear()

        # Assign the sub area to each child
        for i, cid in enumerate(children_ids):
            self.child_spans[parent_id][cid] = Interval(a + i * width, a + (i + 1) * width)

    def _take_largest_free(self, parent_id: int) -> Interval | None:
        """ Gets the largest free interval currently available, or otherwise returns None """

        free = self.free_spans[parent_id]
        if not free:
            return None
        idx = max(range(len(free)), key=lambda i: free[i].len())
        return free.pop(idx)

    def _free_child(self, parent_id: int, child_id: int) -> None:
        iv = self.child_spans[parent_id].pop(child_id, None)
        if iv is not None:
            self.free_spans[parent_id].append(iv)
            self.free_spans[parent_id] = _merge_intervals(self.free_spans[parent_id])

    def reconcile_children(self, parent_id: int, current_children: list[int]) -> None:
        """
        Make sure current children under parent_id have intervals:
        - Keep existing assignments.
        - Free removed children.
        - Allocate intervals for new children without moving existing ones.
        """

        # Deterministic order helps stability
        kids_now = list(sorted(current_children))
        assigned = self.child_spans[parent_id]

        # First-time: even split
        if not assigned and kids_now:
            self._allocate_even(parent_id, kids_now)
            return

        # Free removed
        existing_ids: set[int] = set(assigned.keys())
        current_ids: set[int] = set(kids_now)
        for removed in sorted(existing_ids - current_ids):
            self._free_child(parent_id, removed)

        # Allocate for new children
        new_ids = [cid for cid in kids_now if cid not in assigned]
        for cid in new_ids:
            largest = self._take_largest_free(parent_id)
            if largest is None:
                # No free space: fall back to splitting the parent span proportionally (rare)
                # Here we avoid moving existing by carving a tiny slice from parent end.
                p0, p1 = self.parent_span[parent_id]
                epsilon = (p1 - p0) * 0.01  # 1% slice
                self.parent_span[parent_id] = Interval(p0, p1 - epsilon)
                self.child_spans[parent_id][cid] = Interval(p1 - epsilon, p1)
            else:
                # Split largest free interval in half, give the left half to the new child.
                a, b = largest
                mid = 0.5 * (a + b)
                self.child_spans[parent_id][cid] = Interval(a, mid)

                # Keep remaining half free
                self.free_spans[parent_id].append(Interval(mid, b))
                self.free_spans[parent_id] = _merge_intervals(self.free_spans[parent_id])

    def hue_for_node(self, node_id: int) -> float | None:
        """ Returns the midpoint of its hue. """

        if node_id in self.parent_span:
            return self.parent_span[node_id].midpoint()
        else:
            raise RuntimeError("This node has no color!")    
        
@typechecked
class RerunWrapper():
    """ Wrapper for spawning and visualizing using Rerun. """

    def __init__(self):
        rr.init("Meronomy_Visualization", spawn=True)
        self.update_frame = 0
        self._colors = ColorAllocator()

    def _hsv_to_rgb255(self, h: float, s: float = 1.0, v: float = 1.0) -> np.ndarray[np.uint8]:
        """ Output will be integers 0-255 to be compatible with Rerun """
        
        def clamp(val: float, min_val=0.0, max_val=1.0) -> float:
            return max(min_val, min(max_val, val))

        r, g, b = colorsys.hsv_to_rgb(clamp(h), clamp(s), clamp(v))
        return np.array([[int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]], dtype=np.uint8)

    def _assign_colors_recursive(self, parent_node: GraphNode, parent_span: Interval) -> None:
        """
        Ensure a node’s children get stable intervals within parent_span,
        then recurse into each child with its assigned sub-interval.
        """

        # Make sure the parent's span is tracked and wiped if it has changed.
        curr_id = parent_node.get_id()
        self._colors.update_parent(curr_id, parent_span)

        # Determine children deterministically
        children = sorted(parent_node.get_children(), key=lambda c: c.get_id())
        child_ids = [c.get_id() for c in children]

        # Reconcile current children vs stored state
        self._colors.reconcile_children(curr_id, child_ids)

        # Recurse: each child becomes a parent of its own subspace
        for child in children:
            self._assign_colors_recursive(child, self._colors.child_spans[curr_id][child.get_id()])

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
        initial_span: Interval = Interval(0.0, 1.0)
        self._assign_colors_recursive(root_node, initial_span)

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
            hue = self._colors.hue_for_node(id)
            colors_rgb = np.concatenate((colors_rgb, self._hsv_to_rgb255(hue)), dtype=np.uint8)
            print(colors_rgb)

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