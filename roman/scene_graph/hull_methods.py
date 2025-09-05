from scipy.spatial import KDTree
import numpy as np
import scipy
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist
import trimesh
from typeguard import typechecked

""" Methods for interacting with Convex Hulls and Point Clouds. """

@typechecked
def fix_winding_and_calculate_normals(point_cloud: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ 
    Fixes winding of faces and calculates outward facing normals so results
    can be used to calculate a valid volume with trimesh.Trimesh. Replaces
    trimesh.Trimesh.fix_volume(), mainly for speed reasons. 

    Arguments:
        point_cloud - Point cloud passed to scipy.spatial.ConvexHull
        faces - Simplicies of the scipy.spatial.ConvexHull

    Returns:
        faces - Same as input argument, but with order rearranged to fix winding.
        face_normals - Outward facing normals.
    """

    # Extract the vertices of each face
    v0 = point_cloud[faces[:, 0]]
    v1 = point_cloud[faces[:, 1]]
    v2 = point_cloud[faces[:, 2]]

    # Calculate the normals
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    # Flip normals that point inward & correct winding
    centroid = point_cloud.mean(axis=0)
    face_centers = (v0 + v1 + v2) / 3
    direction = face_centers - centroid
    flip = np.einsum('ij,ij->i', face_normals, direction) < 0
    face_normals[flip] *= -1
    faces[flip] = faces[flip][:, [0, 2, 1]]  # flip winding

    return faces, face_normals

@typechecked
def get_convex_hull_from_point_cloud(point_cloud: np.ndarray) -> trimesh.Trimesh | None:
    # If there are no points passed, return none
    if point_cloud.shape[0] == 0:
        return None

    # Calculate the convex hull
    try:  
        hull = ConvexHull(point_cloud, qhull_options='Qx')
        faces, face_normals = fix_winding_and_calculate_normals(point_cloud, hull.simplices)
        mesh = trimesh.Trimesh(vertices=point_cloud, faces=faces, face_normals=face_normals, process=False)
        assert mesh.is_volume
        return mesh
    
    except scipy.spatial._qhull.QhullError as e: 
        # If not possible to form a hull, return None. 
        return None
    
@typechecked
def find_point_in_hull(points: np.ndarray, hull: trimesh.Trimesh) -> np.ndarray:
    """ Faster Delaunay check if points are inside a convex hull """
    delaunay = Delaunay(hull.vertices)
    return delaunay.find_simplex(points) >= 0
        
@typechecked
def find_point_overlap_with_hulls(pc: np.ndarray, hulls: list[trimesh.Trimesh | None], fail_on_multi_assign: bool = False) -> np.ndarray:  
    # Remove None hulls as they can't overlap with any points
    valid_hulls_indices = []
    for i in range(len(hulls)):
        if hulls[i] is not None:
            valid_hulls_indices.append(i)

    # Find which points fall into which Convex hulls
    contain_masks = np.zeros((len(hulls), len(pc)), dtype=int)
    for i in valid_hulls_indices:
        contain_masks[i] = find_point_in_hull(pc, hulls[i]).astype(int)

    # If fail_on_multi_assign, throw an error if some points fall into multiple hulls
    if fail_on_multi_assign:
        num_mask_assignments = np.sum(contain_masks, axis=0)
        if np.any(num_mask_assignments > 1):
            overlaps = np.where(num_mask_assignments > 1)[0]
            raise RuntimeError(f"Points in observation overlap with multiple child Convex Hulls: {overlaps.tolist()}")
        
    return contain_masks

@typechecked
def convex_hull_geometric_overlap(a: trimesh.Trimesh | None, b: trimesh.Trimesh | None) -> tuple[float, float, float]:
    """
    Returns:
        iou (float): The Intersection over Union between the two hulls.
        enc_a_ratio (float): The percentage of hull a that is enclosed by b.
        enc_b_ratio (float): The percentage of hull b that is enclosed by a.
    """
    # If both hulls are None, throw an error
    if a is None and b is None:
        raise ValueError("Both hulls are None, this shouldn't happen during normal operation!")

    # If either of the hulls are None, then IOU is zero and encompassment is 1.0.
    if a is None or b is None:
        return 0.0, 1.0, 1.0
    
    # Make sure both meshes are watertight
    assert a.is_watertight
    assert b.is_watertight

    # Calculate the intersection trimesh
    intersection = a.intersection(b, engine='manifold')

    # Calculate the IOU value
    inter_vol = intersection.volume
    iou = min(inter_vol / (a.volume + b.volume - inter_vol), 1.0)

    # Calculate the relative enclosure ratios
    enc_a_ratio = min(inter_vol / a.volume, 1.0)
    enc_b_ratio = min(inter_vol / b.volume, 1.0)

    return iou, enc_a_ratio, enc_b_ratio

@typechecked
def shortest_dist_between_convex_hulls(a: trimesh.Trimesh | None, b: trimesh.Trimesh | None) -> float:
    """ Since we sample surfance points, this is an approximation. """
    
    # If both hulls are None, throw an error
    if a is None and b is None:
        raise ValueError("Both hulls are None, this shouldn't happen during normal operation!")

    # If either of the hulls are None, then assume that they are really far away
    if a is None or b is None:
        return np.inf
    
    # Sample surface points on each hull
    points_a = a.sample(1000)
    points_b = b.sample(1000)
    return shortest_dist_between_point_clouds(points_a, points_b)

@typechecked
def shortest_dist_between_point_clouds(a: np.ndarray, b: np.ndarray):
    """ Shortest distance between any pair of points in the point clouds. """

    # Get minimum distance efficently with KDTree
    tree_a = KDTree(a)
    distances = tree_a.query(b, 1)[0]
    return np.array(distances).min()

@typechecked
def longest_line_of_point_cloud(a: np.ndarray[np.float64]) -> float:
    if a.shape[0] == 0: return 0.0
    return pdist(a).max()