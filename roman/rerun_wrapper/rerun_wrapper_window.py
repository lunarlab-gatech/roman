from __future__ import annotations

from .color_utils import hsvF_to_rgb255, rgb255_to_hsvF, HSVSpace
from ..scene_graph.graph_node import GraphNode
import hashlib
from ..logger import logger
import numpy as np
from open3d.geometry import OrientedBoundingBox
import random
import rerun as rr
import rerun.blueprint as rrb
from roman.scene_graph.word_net_wrapper import WordListWrapper
from scipy.spatial.transform import Rotation as R
import trimesh

class RerunWrapperWindow():
    def __init__(self, enable: bool):
        self.enable = enable
        self.update_frame = 0

    # ===================== Methods to Override =====================
    def _get_blueprint_part(self) -> rrb.BlueprintPart:
        raise NotImplementedError("RerunWrapperWindow should not be used directly!")

    def _get_curr_robot_name(self) -> str:
        raise NotImplementedError("RerunWrapperWindow should not be used directly!")

    # ===================== Timeline Handlers =====================
    def _update_frame_tick(self) -> None:
        rr.set_time(f"robot_{self._get_curr_robot_name()}_update_frame_tick", sequence=self.update_frame)
        self.update_frame += 1

    def update_curr_time(self, curr_time: float) -> None:
        rr.set_time(f"robot_{self._get_curr_robot_name()}_camera_frame_time", timestamp=curr_time)

    # ===================== Data Logger Helpers =====================
    @staticmethod
    def extract_data_from_obb(obb: OrientedBoundingBox, box_centers: list, box_half_sizes: list, box_quats: list) -> None:
        box_centers.append(obb.center.tolist())
        box_half_sizes.append((obb.extent / 2).tolist())

        R_matrix = np.array(obb.R, copy=True)
        quat = R.from_matrix(R_matrix).as_quat()
        box_quats.append(quat.tolist())

    # ===================== Data Loggers =====================
    def _update_graph_general(self, root_node: GraphNode, id_to_color_mapping: dict[int, np.ndarray], 
                                  node_statuses_to_show: list[GraphNode.SegmentStatus]):
        if not self.enable: return
        self._update_frame_tick()

        # Create structures to store graph information
        node_ids: list[int] = []
        edges: list[tuple[int, int]] = []
        colors_rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        points: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        point_to_node_ids: np.ndarray = np.zeros((0), dtype=np.uint32)
        point_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)

        # Iterate through all nodes reachable from the root that we want to display
        for node in root_node:
            if not node.get_status() in node_statuses_to_show: continue

            node_ids.append(node.get_id())

            # Add edges
            if not node.is_RootGraphNode():
                parent = node.get_parent()
                if parent.get_status() != GraphNode.SegmentStatus.GRAVEYARD:
                    edges.append((node.get_id(), parent.get_id()))
            for child in node.get_children():
                if child.get_status() != GraphNode.SegmentStatus.GRAVEYARD:
                    edges.append((node.get_id(), child.get_id()))

            # Colors
            colors_rgb = np.concatenate((colors_rgb, [id_to_color_mapping[node.get_id()]]), dtype=np.uint8)

            # Extract points for this node
            points = np.concatenate((points, node.point_cloud), dtype=float)
            point_to_node_ids = np.concatenate((point_to_node_ids, 
                                                np.full((node.point_cloud.shape[0]), node.get_id(), dtype=np.uint32)))
            point_colors = np.concatenate((point_colors, np.full((node.point_cloud.shape[0], 3), [id_to_color_mapping[node.get_id()]])))

        # Calculate bounding boxes & extract convex hulls as lines for nodes
        box_centers, box_half_sizes = [], []
        box_quats = []
        box_colors = []
        box_ids = []
        box_words = []
        line_ends = []
        line_colors = []

        for node in root_node:
            if not node.get_status() in node_statuses_to_show or node.is_RootGraphNode(): continue

            # Axis-aligned bounding box
            obb = node.get_oriented_bbox()
            if obb is None: continue
            self.extract_data_from_obb(node.get_oriented_bbox(), box_centers, box_half_sizes, box_quats)

            # Box colors
            box_colors.append([id_to_color_mapping[node.get_id()]])
            box_ids.append(node.get_id())
            words: WordListWrapper | None = node.get_words()
            if words is not None: box_words.append(words.to_list())
            else: box_words.append([])

            # Line segments
            mesh: trimesh.Trimesh | None = node.get_convex_hull()
            line_edges: set[tuple] = set()
            if mesh is not None:
                for face in mesh.faces:
                    a, b, c = face
                    line_edges.add(tuple(sorted((a, b))))
                    line_edges.add(tuple(sorted((b, c))))
                    line_edges.add(tuple(sorted((c, a))))

                for i1, i2 in line_edges:
                    v1 = mesh.vertices[i1].tolist()
                    v2 = mesh.vertices[i2].tolist()
                    line_ends.append([v1, v2])  # a strip of 2 points = one line
                    line_colors.append([id_to_color_mapping[node.get_id()]])

        # Send the data to Rerun
        rr.log(f"/graph/{self._get_curr_robot_name()}", rr.GraphNodes(node_ids=node_ids, labels=node_ids, colors=colors_rgb), 
                        rr.GraphEdges(edges=edges, graph_type="directed"))
        rr.log(f"/world/{self._get_curr_robot_name()}/points", rr.Points3D(positions=points, colors=point_colors))
        rr.log(f"/world/{self._get_curr_robot_name()}/boxes", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0.01, fill_mode="line",
                            labels=None))
        rr.log(f"/world/{self._get_curr_robot_name()}/labels", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_ids)) #TODO: Can label size change?
        rr.log(f"/world/{self._get_curr_robot_name()}/words", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_words))
        rr.log(f"/world/{self._get_curr_robot_name()}/meshes", rr.LineStrips3D(strips=line_ends, colors=line_colors, radii=0.01))
    
    # ===================== Color Assignment =====================
    def _assign_colors_graph(self, node_colors: dict, node: GraphNode, hsv_space: HSVSpace, depth: int = 0) -> dict:

        NUM_BUCKETS = 20
        node_id = node.get_id()

        if node_id not in node_colors:
            # Deterministic RNG seeded from node_id
            seed_bytes = hashlib.sha256(str(node_id).encode()).digest()
            seed_int = int.from_bytes(seed_bytes[:8], "big")
            rng = random.Random(seed_int)

            # Precompute hue buckets
            hue_buckets = [i / NUM_BUCKETS for i in range(NUM_BUCKETS)]

            if depth > 0:
                parent_h = rgb255_to_hsvF(node_colors[node.get_parent().get_id()])[0]
                parent_bucket = min(
                    range(NUM_BUCKETS),
                    key=lambda i: abs(hue_buckets[i] - parent_h)
                )
                min_sep = NUM_BUCKETS // 4  # 90° separation
                available_buckets = [
                    b for b in range(NUM_BUCKETS)
                    if abs(b - parent_bucket) >= min_sep and abs(b - parent_bucket) <= NUM_BUCKETS - min_sep
                ]
                bucket = available_buckets[seed_int % len(available_buckets)]
            else:
                bucket = seed_int % NUM_BUCKETS

            h = hue_buckets[bucket]
            s = rng.uniform(0.7, 1.0)
            v = rng.uniform(0.8, 1.0)
        
            node_colors[node_id] = hsvF_to_rgb255(np.array([h % 1.0, s, v], dtype=float))

        # Assign colors to children (each will get their own deterministic hue)
        children = sorted(node.get_children(), key=lambda c: c.get_id())
        if not children: return node_colors
        for child in children:
            node_colors = self._assign_colors_graph(node_colors, child, hsv_space, depth + 1)
        return node_colors