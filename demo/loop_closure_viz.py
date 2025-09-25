import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse

def load_gt_csv(gt_csv):
    df = pd.read_csv(gt_csv)
    df["timestamp"] = df["#timestamp_kf"] / 1e9
    df.set_index("timestamp", inplace=True)
    return df

def get_gt_pose(df, timestamp, tolerance=1e-1):
    if timestamp in df.index:
        row = df.loc[timestamp]
    else:
        closest_idx = (np.abs(df.index - timestamp)).argmin()
        closest_time = df.index[closest_idx]
        if abs(closest_time - timestamp) > tolerance:
            raise ValueError(f"No timestamp within {tolerance}s of {timestamp}")
        row = df.iloc[closest_idx]
    t = np.array([row["x"], row["y"], row["z"]])
    r = R.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]])  # GT qw,qx,qz,qy -> xyzw
    return t, r

def relative_transform(t1, r1, t2, r2):
    r_rel = r1.inv() * r2
    t_rel = r1.inv().apply(t2 - t1)
    return t_rel, r_rel

def compute_error(pred_t, pred_r, gt_t, gt_r):
    trans_error = np.linalg.norm(pred_t - gt_t)
    r_diff = pred_r.inv() * gt_r
    angle_error = r_diff.magnitude() * 180/np.pi
    return trans_error, angle_error

def plot_pose(ax, t, r: R, scale=1.0):
    """
    Plot a pose at origin t with rotation r using thick, visible axes.
    t: 3-element array
    r: scipy Rotation
    scale: axis length
    """
    origin = np.array(t)
    R_mat = r.as_matrix()  # get 3x3 matrix

    # Re-orthonormalize to remove tiny numerical errors
    U, _, Vt = np.linalg.svd(R_mat)
    R_mat = U @ Vt

    axes = R_mat  # now guaranteed orthonormal

    # Colors for XYZ
    colors = ['r', 'g', 'b']

    for i in range(3):
        vec = axes[:, i] * scale
        ax.plot([origin[0], origin[0]+vec[0]],
                [origin[1], origin[1]+vec[1]],
                [origin[2], origin[2]+vec[2]],
                color=colors[i], linewidth=3)

    # Optional: draw a small sphere at the origin
    ax.scatter(*origin, color='k', s=20)

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    centers = np.mean(limits, axis=1)
    max_range = np.max(limits[:,1] - limits[:,0]) / 2

    ax.set_xlim3d([centers[0]-max_range, centers[0]+max_range])
    ax.set_ylim3d([centers[1]-max_range, centers[1]+max_range])
    ax.set_zlim3d([centers[2]-max_range, centers[2]+max_range])


def calculate_loop_closure_error(json_file, gt_csv1, gt_csv2, plot=False) -> tuple[float, float]:
    with open(json_file, 'r') as f:
        loops = json.load(f)

    gt_df1 = load_gt_csv(gt_csv1)
    gt_df2 = load_gt_csv(gt_csv2)

    trans_errors = []
    rot_errors = []

    for idx, loop in enumerate(loops):
        t0_sec, t1_sec = loop["seconds"]
        t0_ns, t1_ns = loop["nanoseconds"]
        t0 = t0_sec + t0_ns * 1e-9
        t1 = t1_sec + t1_ns * 1e-9

        # Get GT poses
        t_gt0, r_gt0 = get_gt_pose(gt_df1, t0)
        t_gt1, r_gt1 = get_gt_pose(gt_df2, t1)

        # Compute GT relative transform
        gt_t_rel, gt_r_rel = relative_transform(t_gt0, r_gt0, t_gt1, r_gt1)

        # Predicted transform
        pred_t = np.array(loop["translation"])
        pred_r = R.from_quat(loop["rotation"])

        # Compute errors
        trans_err, rot_err = compute_error(pred_t, pred_r, gt_t_rel, gt_r_rel)
        trans_errors.append(trans_err)
        rot_errors.append(rot_err)

        # print("Translation error:", trans_err)
        # print("Rotation error (deg):", rot_err)

        # --- Create a new figure for this loop closure ---
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot GT pose (solid)
            plot_pose(ax, gt_t_rel, gt_r_rel, scale=0.5)
            # Plot predicted pose (dashed)
            plot_pose(ax, pred_t, pred_r, scale=0.5)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Loop Closure {idx} (GT solid, Predicted dashed)')
            set_axes_equal(ax)
            plt.show()  # display each loop closure plot separately

    mean_trans_error = np.mean(trans_errors)
    mean_rot_error_deg = np.mean(rot_errors)
    print("Mean translation error:",  mean_trans_error)
    print("Mean rotation error (deg):", mean_rot_error_deg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare loop closures to GT and visualize poses.")
    parser.add_argument("json_file", help="Loop closures JSON file")
    parser.add_argument("gt_csv1", help="GT CSV for robot 1")
    parser.add_argument("gt_csv2", help="GT CSV for robot 2")
    args = parser.parse_args()

    calculate_loop_closure_error(args.json_file, args.gt_csv1, args.gt_csv2, plot=True)

