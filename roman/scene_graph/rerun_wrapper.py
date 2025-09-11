from __future__ import annotations

import colorsys
import copy
import cv2
from dataclasses import dataclass
from .graph_node import GraphNode
import hashlib
from .logger import logger
import numpy as np
from ..params.data_params import ImgDataParams
import random
import rerun as rr
import rerun.blueprint as rrb
from roman.params.fastsam_params import FastSAMParams
from scipy.spatial.transform import Rotation as R
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

    def __init__(self,  fastsam_params: FastSAMParams, enable=True):
        
        # TODO: Make line sizes of labels, meshes, and boxes scale based on object size.

        # Save parameters
        self.fastsam_params = fastsam_params

        # If not enabled, don't actually send any data
        self.enable = enable

        if self.enable:
            # Create the blueprint
            blueprint = rrb.Blueprint(
                rrb.Tabs(
                rrb.Vertical(
                rrb.Horizontal(
                    rrb.GraphView(name="Graph", origin='/graph'),
                    rrb.Spatial3DView(name="World", origin='/world'),
                    rrb.TextLogView(name="Text Logs", origin="/logs"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Image", origin='/world/robot/camera/image'),
                    rrb.Spatial2DView(name="Depth", origin='/world/robot/camera/depth'),
                    rrb.Spatial2DView(name="Segmentation Mask", origin='/world/robot/camera/segmentation')
                )),
                rrb.Horizontal(
                    rrb.Spatial3DView(name="FastSAM Projections", origin='/fastsam'),
                    rrb.Vertical(
                        rrb.Spatial2DView(name="Depth", origin='/fastsam/camera/depth'),
                        rrb.Spatial2DView(name='Flow Magnitude', origin='/fastsam/camera/flow_mag'),
                        rrb.Spatial2DView(name="Threshold for Dynamic Pixels", origin='/fastsam/camera/thresh'),
                        rrb.Spatial2DView(name="Detected Dynamic Pixels", origin='/fastsam/camera/mask')
                ))))

            # Spawn Rerun
            rr.init("Meronomy_Visualization", spawn=True, default_blueprint=blueprint)
            self.update_frame = 0

            # Keep track of node colors:
            self.node_colors: dict[int, tuple[float, float, float]] = {}

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

    def update(self, root_node: GraphNode, curr_time: float, img: np.ndarray | None = None, depth_img: np.ndarray | None = None, camera_pose: np.ndarray | None = None, img_data_params: ImgDataParams | None = None, seg_img: np.ndarray | None = None, associations: list[tuple[int, int]] = [], node_to_obs_mapping: dict | None = None):
        """
        Args:
            nodes (GraphNode]): Takes as input the RootGraphNode.
            curr_time (float): The current time of the camera frame.
            img (np.ndarray): An optional image to be visualized
            depth_img (np.ndarray): An optional depth image to be visualized.
            camera_pose
            img_data_params
            seg_img
            associations
        """

        if not self.enable:
            return

        # Update the timelines
        rr.set_time("camera_frame_time", timestamp=curr_time)
        rr.set_time("update_frame_tick", sequence=self.update_frame)
        self.update_frame += 1

        # Assign/refresh color intervals starting at the root.
        full_hsv = HSVSpace((0.0, 1.0), (0.2, 1.0), (0.2, 1.0))
        self._assign_colors_recursive(root_node, full_hsv)

        # Create structures to store graph information
        node_ids: list[int] = []
        edges: list[tuple[int, int]] = []
        colors_rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        points: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        point_to_node_ids: np.ndarray = np.zeros((0), dtype=np.uint32)
        point_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)

        if seg_img is not None:
            num_obs = seg_img.shape[0]
            colormap: np.ndarray = np.full((num_obs+1, 3), 128, dtype=np.uint8)
            colormap[0] = [0, 0, 0]

        # Iterate through all nodes reachable from the root.
        for j, node in enumerate(root_node):
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
            h, s, v = self.node_colors[id]
            color = self._hsv_to_rgb255(h, s, v)
            colors_rgb = np.concatenate((colors_rgb, color), dtype=np.uint8)

            # Extract points for this node
            points = np.concatenate((points, node.point_cloud), dtype=np.float128)
            point_to_node_ids = np.concatenate((point_to_node_ids, 
                                                np.full((node.point_cloud.shape[0]), id, dtype=np.uint32)))
            point_colors = np.concatenate((point_colors, np.full((node.point_cloud.shape[0], 3), color)))

            # Remember this association color for the segmentation mask
            if seg_img is not None:
                for pair in associations:
                    if pair[1] == node.get_id():
                        colormap[node_to_obs_mapping[pair[0]]+1] = self._rgb255_to_bgr255(color)

        # Calculate bounding boxes & extract convex hulls as lines for nodes
        box_centers = []
        box_half_sizes = []
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
            h, s, v = self.node_colors[node.get_id()]
            color = self._hsv_to_rgb255(h, s, v)
            box_colors.append(color)
            box_ids.append(node.get_id())
            box_words.append(node.get_word())

            # Line segments
            mesh = node.get_convex_hull()
            line_edges: set[tuple] = set()
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
        rr.log("/graph", rr.GraphNodes(node_ids=node_ids, labels=node_ids, colors=colors_rgb), 
                        rr.GraphEdges(edges=edges, graph_type="directed"))
        rr.log("/world/points", rr.Points3D(positions=points, colors=point_colors))
        rr.log("/world/boxes", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0.01, fill_mode="line",
                            labels=None))
        rr.log("/world/labels", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_ids)) #TODO: Can label size change?
        rr.log("/world/words", rr.Boxes3D(centers=box_centers, half_sizes=box_half_sizes,
                            quaternions=box_quats, colors=box_colors, radii=0, fill_mode="line",
                            labels=box_words))
        rr.log("/world/meshes", rr.LineStrips3D(strips=line_ends, colors=line_colors, radii=0.01))

        if img is not None:
            rr.log("/world/robot/camera/image", rr.Image(img, color_model="BGR"))
        if depth_img is not None:
            depth_img_vis = copy.deepcopy(depth_img).astype(np.float32)
            depth_img_vis /= self.fastsam_params.depth_scale
            depth_img_vis[depth_img_vis > self.fastsam_params.max_depth] = self.fastsam_params.max_depth
            rr.log("/world/robot/camera/depth", rr.DepthImage(depth_img_vis))
        if camera_pose is not None:
            rot = R.from_matrix(camera_pose[:3,:3])
            rr.log("/world/robot/camera", rr.Transform3D(translation=camera_pose[:3,3],
                quaternion=rot.as_quat(), relation=rr.TransformRelation.ParentFromChild, clear=False), strict=True)
        if img_data_params is not None:
            rr.log("/world/robot/camera/image", 
                   rr.Pinhole(resolution=[img_data_params.width, img_data_params.height],
                              focal_length=[img_data_params.K[0], img_data_params.K[4]],
                              principal_point=[img_data_params.K[2], img_data_params.K[5]],
                              image_plane_distance=1.0))
        if seg_img is not None:
            # Calculate image of masks with bools instead of per number
            bool_img = np.zeros_like(seg_img)
            bool_img[seg_img > 0] = 1

            # Get array representing nubmer of observations in each pixel
            obs_per_pixel = np.sum(bool_img, axis=0) + np.full(bool_img.shape[1:], 0.00001)
            obs_per_pixel = obs_per_pixel[:,:,np.newaxis]
            obs_per_pixel = np.repeat(obs_per_pixel, 3, axis=2)

            # Calculate the color as the average of the colors divided by the number of observations
            color_mask = np.divide(np.sum(colormap[seg_img], axis=0), obs_per_pixel).astype(np.uint8)
            
            # Overlay color onto the normal image
            overlay = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
            rr.log("/world/robot/camera/segmentation", rr.Image(overlay, color_model="BGR"))