from helpers import *
from metrics import *
from MeshProcessor import MeshProcessor
import trimesh as tri
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


lg = log.getLogger(__name__)
tm_fmt = r"%d.%m.%y_%H.%M.%S"

def side_by_side_visualization(mesh1: tri.Geometry, mesh2: tri.Geometry, translation_amount: float = 1.2, titles: tuple = ("Original Mesh", "Processed Mesh"), show:bool = True):
    "Visualize the original and processed meshes side by side."
    mesh2_t = mesh2.copy()
    mesh2_t.apply_translation([translation_amount, 0, 0])
    
    scene = tri.Scene()
    scene.add_geometry(mesh1, node_name=titles[0])
    scene.add_geometry(mesh2_t, node_name=titles[1])
    if show: scene.show()
    return scene

def plot_error_per_axis(vert1: tri.caching.TrackedArray, vert2: tri.caching.TrackedArray, pfx: str = "", show:bool = True):
    out = []
    if vert1.shape != vert2.shape:
        lg.error("Vertex array shapes do not match. Cannot plot error.")
        raise ValueError("Vertex arrays must have the same shape.")
    
    metrics = {
        "mae": mae(vert1, vert2, axis_wise=True),
        "mse": mse(vert1, vert2, axis_wise=True),
        "rmse": rmse(vert1, vert2, axis_wise=True)
    }
    
    axes_labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
    
    # Create a figure with 1 row and 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Set a main title
    if pfx:
        fig.suptitle(f"'{pfx}' - Summary Error Metrics per Axis", fontsize=16, y=1.02)
    else:
        fig.suptitle("Summary Error Metrics per Axis", fontsize=16, y=1.02)

    # MAE Bar Chart 
    ax1.bar(axes_labels, metrics["mae"], color=['red', 'green', 'blue'], alpha=0.7)
    ax1.set_title('Mean Absolute Error (MAE)')
    ax1.set_ylabel('Error')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # MSE Bar Chart
    ax2.bar(axes_labels, metrics["mse"], color=['red', 'green', 'blue'], alpha=0.7)
    ax2.set_title('Mean Squared Error (MSE)')
    ax2.set_ylabel('Error')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    # RMSE Bar Chart 
    ax3.bar(axes_labels, metrics["rmse"], color=['red', 'green', 'blue'], alpha=0.7)
    ax3.set_title('Root Mean Squared Error (RMSE)')
    ax3.set_ylabel('Error')
    ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    plt.tight_layout()
    if show: plt.show()
    
    out.append(fig)
    
    error_vectors = vert1 - vert2
    error_x = error_vectors[:, 0]
    error_y = error_vectors[:, 1]
    error_z = error_vectors[:, 2]
    
    fig, (ax_x, ax_y, ax_z) = plt.subplots(1, 3, figsize=(20, 6))
    
    if pfx:
        fig.suptitle(f"'{pfx}' - Reconstruction Error Distribution per Axis", fontsize=16, y=1.02)
    else:
        fig.suptitle("Reconstruction Error Distribution per Axis", fontsize=16, y=1.02)
    
    # Plot X-Axis Error Histogram
    ax_x.hist(error_x, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax_x.set_title(f'X-Axis Error\n(MAE: {metrics["mae"][0]:.6f} | RMSE: {metrics["rmse"][0]:.6f})', fontsize=12)
    ax_x.set_xlabel('Error Magnitude')
    ax_x.set_ylabel('Frequency (Vertex Count)')
    ax_x.grid(True, linestyle='--', alpha=0.5)
    ax_x.axvline(0, color='black', linestyle='--', linewidth=1)

    # Plot Y-Axis Error Histogram
    ax_y.hist(error_y, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax_y.set_title(f'Y-Axis Error\n(MAE: {metrics["mae"][1]:.6f} | RMSE: {metrics["rmse"][1]:.6f})', fontsize=12)
    ax_y.set_xlabel('Error Magnitude')
    ax_y.grid(True, linestyle='--', alpha=0.5)
    ax_y.axvline(0, color='black', linestyle='--', linewidth=1)

    # Plot Z-Axis Error Histogram
    ax_z.hist(error_z, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax_z.set_title(f'Z-Axis Error\n(MAE: {metrics["mae"][2]:.6f} | RMSE: {metrics["rmse"][2]:.6f})', fontsize=12)
    ax_z.set_xlabel('Error Magnitude')
    ax_z.grid(True, linestyle='--', alpha=0.5)
    ax_z.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # Adjust layout and show plot
    plt.tight_layout()
    if show: plt.show()
    out.append(fig)
    
    return out