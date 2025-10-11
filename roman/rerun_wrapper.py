from __future__ import annotations

import colorsys
import copy
import cv2
from dataclasses import dataclass
from enum import Enum
from .scene_graph.graph_node import GraphNode
import hashlib
from .logger import logger
import matplotlib.pyplot as plt
import numpy as np
from .params.data_params import ImgDataParams
import random
import rerun as rr
import rerun.blueprint as rrb
from roman.object.segment import Segment
from roman.params.fastsam_params import FastSAMParams
from roman.scene_graph.word_net_wrapper import WordListWrapper
from scipy.spatial.transform import Rotation as R
import trimesh
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

    class RerunWrapperWindow(Enum):
        MapLive = 0
        MapFinal = 1

    """ Wrapper for spawning and visualizing using Rerun. """
    # TODO: Switch from Axis-Aligned Bounding Box to Oriented Bounding Box

    def __init__(self, name: str, windows: list[RerunWrapper.RerunWrapperWindow], 
                 fastsam_params: FastSAMParams | None = None, enable: bool = True):
        
        # Save & check parameters
        self.name = name
        self.windows = windows
        self.fastsam_params = fastsam_params
        self.enable = enable
        if RerunWrapper.RerunWrapperWindow.MapLive in windows and fastsam_params is None:
            raise ValueError("MapLive window requires fastsam_params to be provided for depth imagery")
        if RerunWrapper.RerunWrapperWindow.MapLive in windows and RerunWrapper.RerunWrapperWindow.MapFinal in windows:
            raise ValueError("Windows can only comprise of MapLive or MapFinal, not both at the same time")
        if RerunWrapper.RerunWrapperWindow.MapFinal in windows and len(windows) > 1:
            raise ValueError("Only single MapFinal window supported at a time")
        
        # Tracking variables
        self.curr_robot_index = 0
        self.update_frame = 0
        self.node_colors: dict[int, tuple[float, float, float]] = {} # H, S, V
        self.robot_colors: dict[int, np.ndarray] = {} # RGB arrays

        # Do initialization if we are enabled
        if self.enable:
            
            # Create a tab for each robot
            robot_tabs = []
            for window in windows:
                robot_tab = self.get_blueprint_part_for_window(window)
                robot_tabs.append(robot_tab)

            # Create the blueprint
            blueprint = rrb.Blueprint(rrb.Tabs(*robot_tabs))

            # Spawn Rerun
            rr.init(self.name, spawn=True, default_blueprint=blueprint)

            # Calculate colors for each robot (used for MapFinal)
            cmap = plt.get_cmap("tab10")
            for i in range(self.curr_robot_index):
                rgb_f = cmap(i % 10)[:3]
                self.robot_colors[i] = np.array([[int(round(rgb_f[0] * 255)), int(round(rgb_f[1] * 255)), 
                                                  int(round(rgb_f[2] * 255))]], dtype=np.uint8) 

            # Set current robot index to None, calling code should set it now
            self.curr_robot_index = None

    def get_blueprint_part_for_window(self, window: RerunWrapper.RerunWrapperWindow) -> rrb.BlueprintPart:
        if window == RerunWrapper.RerunWrapperWindow.MapLive:  

            # Create the views
            graph_view = rrb.GraphView(name="Graph", origin=f'/graph/robot_{self.curr_robot_index}')
            world_view = rrb.Spatial3DView(name="World", origin=f'/world/robot_{self.curr_robot_index}')
            log_view = rrb.TextLogView(name="Text Logs", origin=f"/logs")
            image_view = rrb.Spatial2DView(name="Image", origin=f'/world/robot_{self.curr_robot_index}/camera/image')
            depth_view = rrb.Spatial2DView(name="Depth", origin=f'/world/robot_{self.curr_robot_index}/camera/depth')
            seg_view = rrb.Spatial2DView(name="Segmentation Mask", origin=f'/world/robot_{self.curr_robot_index}/camera/segmentation')

            # Create the tab
            world_graph_log_horiz = rrb.Horizontal(graph_view, world_view, log_view)
            img_depth_seg_horiz = rrb.Horizontal(image_view, depth_view, seg_view)
            tab_name: str = f"Robot {self.curr_robot_index}"
            self.curr_robot_index += 1
            return rrb.Vertical(world_graph_log_horiz, img_depth_seg_horiz, name=tab_name)

        elif window == RerunWrapper.RerunWrapperWindow.MapFinal:

            # Create the views
            graph_0_view = rrb.GraphView(name="Graph", origin=f'/graph/robot_{self.curr_robot_index}')
            self.curr_robot_index += 1
            graph_1_view = rrb.GraphView(name="Graph", origin=f'/graph/robot_{self.curr_robot_index}')
            self.curr_robot_index += 1
            world_view = rrb.Spatial3DView(name="World", origin=f'/world')

            # Create the tab
            graphs_vert = rrb.Vertical(graph_0_view, graph_1_view)
            return rrb.Horizontal(world_view, graphs_vert)


    def _hsv_to_rgb255(self, h: float, s: float = 1.0, v: float = 1.0) -> np.ndarray[np.uint8]:
        """ Output will be integers 0-255 to be compatible with Rerun """
        
        def clamp(val: float, min_val=0.0, max_val=1.0) -> float:
            return max(min_val, min(max_val, val))

        r, g, b = colorsys.hsv_to_rgb(clamp(h), clamp(s), clamp(v))
        return np.array([[int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]], dtype=np.uint8)
    
    def _rgb255_to_bgr255(self, rgb: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        return rgb[:, ::-1]

    def _assign_colors_recursive(self, node: GraphNode, hsv_space: HSVSpace, depth: int = 0) -> None:

        NUM_BUCKETS = 20
        node_id = node.get_id()

        if node_id not in self.node_colors:
            # Deterministic RNG seeded from node_id
            seed_bytes = hashlib.sha256(str(node_id).encode()).digest()
            seed_int = int.from_bytes(seed_bytes[:8], "big")
            rng = random.Random(seed_int)

            # Precompute hue buckets
            hue_buckets = [i / NUM_BUCKETS for i in range(NUM_BUCKETS)]

            if depth > 0:
                parent_h, _, _ = self.node_colors[node.get_parent().get_id()]
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

            self.node_colors[node_id] = (h % 1.0, s, v)

        # Assign colors to children (each will get their own deterministic hue)
        children = sorted(node.get_children(), key=lambda c: c.get_id())
        if not children:
            return

        for child in children:
            self._assign_colors_recursive(child, hsv_space, depth + 1)

    def _update_frame_tick(self) -> None:
        rr.set_time(f"robot_{self.curr_robot_index}_update_frame_tick", sequence=self.update_frame)
        self.update_frame += 1

    def set_curr_robot_index(self, index: int) -> None:
        self.curr_robot_index = index

    def update_curr_time(self, curr_time: float) -> None:
        rr.set_time(f"robot_{self.curr_robot_index}_camera_frame_time", timestamp=curr_time)

    def update_img(self, img: np.ndarray) -> None:
        if not self.enable: return
        self._update_frame_tick()
        rr.log(f"/world/robot_{self.curr_robot_index}/camera/image", rr.Image(img, color_model="BGR"))

    def update_depth_img(self, depth_img: np.ndarray) -> None:
        if not self.enable: return
        self._update_frame_tick()

        # Scale image and threshold values past max_detph
        depth_img_vis = copy.deepcopy(depth_img).astype(np.float32)
        depth_img_vis /= self.fastsam_params.depth_scale
        depth_img_vis[depth_img_vis > self.fastsam_params.max_depth] = self.fastsam_params.max_depth
        rr.log(f"/world/robot_{self.curr_robot_index}/camera/depth", rr.DepthImage(depth_img_vis))

    def update_camera_pose(self, camera_pose: np.ndarray) -> None:
        if not self.enable: return
        self._update_frame_tick()

        rot = R.from_matrix(camera_pose[:3,:3])
        rr.log(f"/world/robot_{self.curr_robot_index}/camera", rr.Transform3D(translation=camera_pose[:3,3],
            quaternion=rot.as_quat(), relation=rr.TransformRelation.ParentFromChild, clear=False), strict=True)
        
    def update_camera_intrinsics(self, img_data_params: ImgDataParams) -> None:
        if not self.enable: return
        self._update_frame_tick()

        rr.log(f"/world/robot_{self.curr_robot_index}/camera/image", 
                rr.Pinhole(resolution=[img_data_params.width, img_data_params.height],
                            focal_length=[img_data_params.K[0], img_data_params.K[4]],
                            principal_point=[img_data_params.K[2], img_data_params.K[5]],
                            image_plane_distance=1.0))
        
    def update_seg_img(self, seg_img: np.ndarray, img: np.ndarray, associations: list[tuple[int, int]], node_to_obs_mapping: dict) -> None:
        if not self.enable: return
        self._update_frame_tick()

        # Create mapping from observation to color
        num_obs: int = seg_img.shape[0]
        colormap: np.ndarray = np.full((num_obs+1, 3), 128, dtype=np.uint8)
        colormap[0] = [0, 0, 0]
        for pair in associations:
            h, s, v = self.node_colors[pair[1]]
            rgb_color = self._hsv_to_rgb255(h, s, v)
            colormap[node_to_obs_mapping[pair[0]]+1] = self._rgb255_to_bgr255(rgb_color)
        
        # Calculate image of masks with bools instead of per number
        bool_img = np.zeros_like(seg_img)
        bool_img[seg_img > 0] = 1

        # Get array representing number of observations in each pixel
        obs_per_pixel = np.sum(bool_img, axis=0) + np.full(bool_img.shape[1:], 0.00001)
        obs_per_pixel = obs_per_pixel[:,:,np.newaxis]
        obs_per_pixel = np.repeat(obs_per_pixel, 3, axis=2)

        # Calculate the color as the average of the colors divided by the number of observations
        color_mask = np.divide(np.sum(colormap[seg_img], axis=0), obs_per_pixel).astype(np.uint8)
        
        # Overlay color onto the normal image
        overlay = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
        rr.log(f"/world/robot_{self.curr_robot_index}/camera/segmentation", rr.Image(overlay, color_model="BGR"))

    def update_graph(self, root_node: GraphNode):
        if not self.enable: return
        self._update_frame_tick()

        # Assign/refresh color intervals for nodes starting at the root.
        full_hsv = HSVSpace((0.0, 1.0), (0.2, 1.0), (0.2, 1.0))
        self._assign_colors_recursive(root_node, full_hsv)

        # Create structures to store graph information
        node_ids: list[int] = []
        edges: list[tuple[int, int]] = []
        colors_rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        points: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        point_to_node_ids: np.ndarray = np.zeros((0), dtype=np.uint32)
        point_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)

        # Iterate through all nodes reachable from the root.
        for j, node in enumerate(root_node):

            # If we are mapping live, also skip Graveyard nodes:
            if (RerunWrapper.RerunWrapperWindow.MapLive in self.windows) and (node.get_status() == GraphNode.SegmentStatus.GRAVEYARD):
                continue

            # Extract id
            id = node.get_id()
            node_ids.append(id)

            # Add edges
            if not node.is_RootGraphNode():
                parent = node.get_parent()
                if parent.get_status() != GraphNode.SegmentStatus.GRAVEYARD:
                    edges.append((id, parent.get_id()))
            for child in node.get_children():
                if child.get_status() != GraphNode.SegmentStatus.GRAVEYARD:
                    edges.append((id, child.get_id()))

            # Colors
            if RerunWrapper.RerunWrapperWindow.MapLive in self.windows:
                h, s, v = self.node_colors[id]
                color = self._hsv_to_rgb255(h, s, v)
            else:
                color = self.robot_colors[self.curr_robot_index]
                
            colors_rgb = np.concatenate((colors_rgb, color), dtype=np.uint8)

            # Extract points for this node
            points = np.concatenate((points, node.point_cloud), dtype=np.float128)
            point_to_node_ids = np.concatenate((point_to_node_ids, 
                                                np.full((node.point_cloud.shape[0]), id, dtype=np.uint32)))
            point_colors = np.concatenate((point_colors, np.full((node.point_cloud.shape[0], 3), color)))

        # Calculate bounding boxes & extract convex hulls as lines for nodes
        box_centers, box_half_sizes = [], []
        box_quats = []
        box_colors = []
        box_ids = []
        box_words = []
        line_ends = []
        line_colors = []

        for j, node in enumerate(root_node):
            # Skip the root graph node
            if node.is_RootGraphNode():
                continue

            # If we are mapping live, also skip Graveyard nodes:
            if (RerunWrapper.RerunWrapperWindow.MapLive in self.windows) and (node.get_status() == GraphNode.SegmentStatus.GRAVEYARD):
                continue

            # Axis-aligned bounding box
            pc = node.get_point_cloud()
            min_corner = pc.min(axis=0)
            max_corner = pc.max(axis=0)
            center = (min_corner + max_corner) / 2.0
            size = (max_corner - min_corner) / 2.0

            box_centers.append(center.tolist())
            box_half_sizes.append(size.tolist())
            box_quats.append(rr.Quaternion.identity()) 

            # Box colors
            if RerunWrapper.RerunWrapperWindow.MapLive in self.windows:
                h, s, v = self.node_colors[node.get_id()]
                color = self._hsv_to_rgb255(h, s, v)
            else:
                color = self.robot_colors[self.curr_robot_index]
            box_colors.append(color)
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
                    line_colors.append(color)


        # Send the data to Rerun
        rr.log(f"/graph/robot_{self.curr_robot_index}", rr.GraphNodes(node_ids=node_ids, labels=node_ids, colors=colors_rgb), 
                        rr.GraphEdges(edges=edges, graph_type="directed"))
        rr.log(f"/world/robot_{self.curr_robot_index}/points", rr.Points3D(positions=points, colors=point_colors))
        rr.log(f"/world/robot_{self.curr_robot_index}/boxes", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0.01, fill_mode="line",
                            labels=None))
        rr.log(f"/world/robot_{self.curr_robot_index}/labels", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_ids)) #TODO: Can label size change?
        rr.log(f"/world/robot_{self.curr_robot_index}/words", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_words))
        rr.log(f"/world/robot_{self.curr_robot_index}/meshes", rr.LineStrips3D(strips=line_ends, colors=line_colors, radii=0.01))

    def update_segments(self, segments: list[Segment]):
        if not self.enable: return
        self._update_frame_tick()
        assert RerunWrapper.RerunWrapperWindow.MapFinal in self.windows, "MapLive not supported for update_segments()"

        # Create structures to store graph information
        seg_ids: list[int] = []
        colors_rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        points: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        point_to_node_ids: np.ndarray = np.zeros((0), dtype=np.uint32)
        point_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)

        # Iterate through all segments
        for j, seg in enumerate(segments):

            # Extract id
            id = seg.id
            seg_ids.append(id)

            # Colors
            color = self.robot_colors[self.curr_robot_index]
            colors_rgb = np.concatenate((colors_rgb, color), dtype=np.uint8)

            # Extract points for this node
            points = np.concatenate((points, seg.points), dtype=np.float128)
            point_to_node_ids = np.concatenate((point_to_node_ids, 
                                                np.full((seg.points.shape[0]), id, dtype=np.uint32)))
            point_colors = np.concatenate((point_colors, np.full((seg.points.shape[0], 3), color)))

        # Calculate bounding boxes
        box_centers, box_half_sizes = [], []
        box_quats = []
        box_colors = []
        box_ids = []

        for j, seg in enumerate(segments):

            # Axis-aligned bounding box
            pc = seg.points
            min_corner = pc.min(axis=0)
            max_corner = pc.max(axis=0)
            center = (min_corner + max_corner) / 2.0
            size = (max_corner - min_corner) / 2.0

            box_centers.append(center.tolist())
            box_half_sizes.append(size.tolist())
            box_quats.append(rr.Quaternion.identity()) 

            # Box colors
            color = self.robot_colors[self.curr_robot_index]
            box_colors.append(color)
            box_ids.append(seg.id)

        # Send the data to Rerun
        rr.log(f"/world/robot_{self.curr_robot_index}/points", rr.Points3D(positions=points, colors=point_colors))
        rr.log(f"/world/robot_{self.curr_robot_index}/boxes", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0.01, fill_mode="line",
                            labels=None))
        rr.log(f"/world/robot_{self.curr_robot_index}/labels", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_ids))
    

