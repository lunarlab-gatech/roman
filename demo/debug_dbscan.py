import numpy as np
import open3d as o3d

def final_cleanup(points, epsilon=0.25, min_points=10):
        if points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Perform DBSCAN clustering
            labels = np.array(pcd.cluster_dbscan(eps=epsilon, min_points=min_points))
            print(f"Labels: {labels}")

            # Number of clusters, ignoring noise if present
            max_label = labels.max()
            
            # get largest cluster
            cluster_sizes = np.zeros(max_label + 1)
            for i in range(max_label + 1):
                cluster_sizes[i] = np.sum(labels == i)
            max_cluster = np.argmax(cluster_sizes)

            # Filter out any points not belonging to max cluster
            filtered_indices = np.where(labels == max_cluster)[0]
            points = points[filtered_indices]
        return points

# Load the .npy file with the points to debug
points = np.load('points.npy')

# Print the average distance between points
print(f"Number of points: {len(points)}")

# Find the closest point to each point and calculate the average distance
closest_distances = []
for i, point in enumerate(points):
    # Be sure not to include the point itself in the distance calculation
    points_without_self = np.delete(points, i, axis=0)
    closest_distances.append(np.min(np.linalg.norm(points_without_self - point, axis=1)))

# Print average distance to closest point
print(f"Average distance to closest point: {np.mean(closest_distances)}")

# Print the bounding box of the points
print(f"Bounding box: {np.min(points, axis=0)} to {np.max(points, axis=0)}")

# Visualize the points using Open3D
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# vis.add_geometry(pcd)
# vis.run()

# Run the dbscan cleanup
points = final_cleanup(points, epsilon=2, min_points=10)

# Visualize the cleaned geometry
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
vis.add_geometry(pcd)
vis.run()