import colorsys
import hashlib
import numpy as np
import random
import trimesh
import pyvista as pv

def hsv_shift(hsv, h_shift=0.1, s_shift=0.0, v_shift=-0.1):
    h = (hsv[0] + h_shift) % 1.0
    s = min(max(hsv[1] + s_shift, 0.0), 1.0)
    v = min(max(hsv[2] + v_shift, 0.0), 1.0)
    return (h, s, v)

def random_base_color_hsv():
    return (random.random(), 0.6, 0.9)

def hsv_to_rgb_tuple(hsv):
    return list(colorsys.hsv_to_rgb(*hsv))

def get_convex_hull_from_point_cloud(points):
    if len(points) < 4:
        return None
    return trimesh.convex.convex_hull(trimesh.PointCloud(points))

class SceneGraphViewer:
    def __init__(self):
        self.plotter = pv.Plotter(window_size=[1280, 720])
        self.plotter.set_background("white")
        self.geometry_actors = {}
        self._window_open = False

        self._last_node_hashes = {}
        self._last_positions = {}

    def _hash_mesh(self, vertices, faces):
        """Create a hash string for mesh vertices and faces for change detection."""
        m = hashlib.sha256()
        m.update(vertices.tobytes())
        m.update(faces.tobytes())
        return m.hexdigest()

    def clear_actor(self, key):
        """Remove a single actor if exists."""
        actor = self.geometry_actors.pop(key, None)
        if actor is not None:
            self.plotter.remove_actor(actor)

    def update(self, scene_graph):
        current_node_hashes = {}
        current_positions = {}

        def traverse(node, parent_centroid=None, parent_hsv=None, depth=0):
            if parent_hsv is None:
                hsv = random_base_color_hsv()
            else:
                h_shift = 0.2 * (random.random() - 0.5)
                hsv = hsv_shift(parent_hsv, h_shift=h_shift, s_shift=0.05, v_shift=-0.05)

            rgb = hsv_to_rgb_tuple(hsv)
            points = node.get_point_cloud()
            if points is None or len(points) == 0:
                return
            centroid = np.mean(points, axis=0)
            current_positions[node] = centroid

            # Convex hull mesh and hash
            hull = None
            mesh_hash = None
            try:
                hull = get_convex_hull_from_point_cloud(points)
                if hull:
                    faces = np.hstack(
                        [np.full((len(hull.faces), 1), 3), hull.faces]
                    ).astype(np.int64)
                    mesh_hash = self._hash_mesh(hull.vertices, faces)
            except Exception as e:
                print(f"Failed to compute hull: {e}")

            current_node_hashes[node] = mesh_hash

            # Compare with last stored hash and position to decide if update needed
            last_hash = self._last_node_hashes.get(node)
            last_pos = self._last_positions.get(node)

            # Update convex hull if changed or new
            mesh_key = f"mesh_{id(node)}"
            if mesh_hash != last_hash:
                # Remove old actor if exists
                self.clear_actor(mesh_key)

                if hull:
                    mesh = pv.PolyData(hull.vertices, faces)
                    mesh.compute_normals(inplace=True)
                    actor = self.plotter.add_mesh(
                        mesh, color=rgb, opacity=0.25, show_edges=True
                    )
                    self.geometry_actors[mesh_key] = actor

            # Update edge to parent if position changed or new
            edge_key = f"edge_{id(node)}"
            if parent_centroid is not None:
                if last_pos is None or not np.allclose(centroid, last_pos) or not np.allclose(parent_centroid, self._last_positions.get(node, parent_centroid)):
                    self.clear_actor(edge_key)
                    line_points = np.array([parent_centroid, centroid])
                    line = pv.Line(line_points[0], line_points[1])
                    actor = self.plotter.add_mesh(line, color=rgb, line_width=3)
                    self.geometry_actors[edge_key] = actor
            else:
                self.clear_actor(edge_key)  # No parent line for root

            for child in node.get_children():
                traverse(child, centroid, hsv, depth + 1)

        traverse(scene_graph.root_node)

        # Remove actors for nodes that no longer exist
        removed_nodes = set(self._last_node_hashes.keys()) - set(current_node_hashes.keys())
        for node in removed_nodes:
            self.clear_actor(f"mesh_{id(node)}")
            self.clear_actor(f"edge_{id(node)}")
            self.clear_actor(f"graph_node_{id(node)}")
            self.clear_actor(f"link_{id(node)}")
            for child in self._last_positions.keys():
                self.clear_actor(f"conn_{id(node)}_{id(child)}")
                self.clear_actor(f"conn_{id(child)}_{id(node)}")

        # Abstract graph above with scaled node spheres and links
        scale_factor = 0.5
        node_sizes = {}

        # Compute node sizes (like before)
        for node, pos in current_positions.items():
            try:
                points = node.get_point_cloud()
                hull = get_convex_hull_from_point_cloud(points)
                if hull:
                    faces = np.hstack(
                        [np.full((len(hull.faces), 1), 3), hull.faces]
                    ).astype(np.int64)
                    mesh = pv.PolyData(hull.vertices, faces)
                    bounds = mesh.bounds
                    bbox_size = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
                    node_sizes[node] = bbox_size
                else:
                    node_sizes[node] = 1.0
            except:
                node_sizes[node] = 1.0

        for node, pos in current_positions.items():
            hull_size = node_sizes.get(node, 1.0)
            offset = scale_factor * 5 * hull_size
            above = pos + np.array([0, 0, offset])
            sphere_radius = 0.05 * hull_size * scale_factor

            # Check if graph node sphere or link changed
            graph_node_key = f"graph_node_{id(node)}"
            graph_node_changed = False
            if graph_node_key not in self.geometry_actors:
                graph_node_changed = True
            else:
                # Could add more advanced checks (position changes) if desired
                pass

            if graph_node_changed:
                self.clear_actor(graph_node_key)
                sphere = pv.Sphere(radius=sphere_radius, center=above)
                actor = self.plotter.add_mesh(sphere, color=[0.2, 0.2, 1.0])
                self.geometry_actors[graph_node_key] = actor

            # Line from hull centroid to graph node
            link_key = f"link_{id(node)}"
            link_changed = False
            if link_key not in self.geometry_actors:
                link_changed = True

            if link_changed:
                self.clear_actor(link_key)
                link_line = pv.Line(pos, above)
                actor = self.plotter.add_mesh(link_line, color=[0, 0, 0], line_width=2)
                self.geometry_actors[link_key] = actor

            # Connections between graph nodes
            for child in node.get_children():
                if child in current_positions:
                    child_hull_size = node_sizes.get(child, 1.0)
                    child_offset = scale_factor * child_hull_size
                    child_above = current_positions[child] + np.array([0, 0, child_offset])

                    conn_key = f"conn_{id(node)}_{id(child)}"
                    if conn_key not in self.geometry_actors:
                        conn_line = pv.Line(above, child_above)
                        actor = self.plotter.add_mesh(conn_line, color=[0, 0, 0], line_width=1)
                        self.geometry_actors[conn_key] = actor

        self._last_node_hashes = current_node_hashes
        self._last_positions = current_positions

        self.plotter.reset_camera()
        if not self._window_open:
            self.plotter.show(auto_close=False, interactive_update=True)
            self._window_open = True
        else:
            self.plotter.update()
