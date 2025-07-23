from scipy.spatial import KDTree
import numpy as np
import scipy
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import trimesh
from typeguard import typechecked

""" Methods for interacting with Convex Hulls. """

@typechecked
def get_convex_hull_from_point_cloud(point_cloud: np.ndarray) -> trimesh.Trimesh | None:
    # If there are no points passed, return none
    if point_cloud.shape[0] == 0:
        return None

    # Calculate the convex hull
    try:  
        hull = ConvexHull(point_cloud)
        mesh = trimesh.Trimesh(vertices=point_cloud, faces=hull.simplices, process=True)
        if not mesh.is_volume:
            mesh.fix_normals()
        return mesh
    
    except scipy.spatial._qhull.QhullError as e: 
        # If not possible to form a hull, return None. 
        return None
        
@typechecked
def find_point_overlap_with_hulls(pc: np.ndarray, hulls: list[trimesh.Trimesh | None], fail_on_multi_assign: bool = False) -> np.ndarray:  
    # Remove None hulls as they can't overlap with any points
    valid_hulls_indices = []
    for i in range(len(hulls)):
        if hulls[i] is not None:
            valid_hulls_indices.append(i)

    # Find which points fall into which Convex hulls
    contain_masks = np.zeros((len(hulls), len(pc)), dtype=int)
    for i, hull in enumerate(hulls):
        if i in valid_hulls_indices:
            contain_masks[i] = np.array(hull.contains(pc), dtype=int)

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

    # Calculate the intersection trimesh
    intersection = a.intersection(b, engine='manifold')

    # Calculate the IOU value
    inter_vol = intersection.volume
    iou = inter_vol / (a.volume + b.volume - inter_vol)

    # Calculate the relative enclosure ratios
    enc_a_ratio = inter_vol / a.volume
    enc_b_ratio = inter_vol / b.volume

    return iou, enc_a_ratio, enc_b_ratio

@typechecked
def shortest_dist_between_convex_hulls(a: trimesh.Trimesh, b: trimesh.Trimesh):
    """ Since we sample surfance points, this is an approximation. """

    # Sample surface points on each hull
    points_a = a.sample(1000)
    points_b = b.sample(1000)

    # Get minimum distance efficently with KDTree
    tree_a = KDTree(points_a)
    distances = tree_a.query(points_b, 1)
    return np.array(distances).min()

@typechecked
def shortest_dist_between_point_clouds(a: np.ndarray, b: np.ndarray):
    """ Shortest distance between any pair of points in the point clouds. """

    # Get minimum distance efficently with KDTree
    tree_a = KDTree(a)
    distances = tree_a.query(b, 1)
    return np.array(distances).min()

@typechecked
def longest_line_of_point_cloud(a: np.ndarray):
    return pdist(a).max()