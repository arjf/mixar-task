# %%
import os
import sys
import logging as log
import datetime

import numpy as np
import trimesh as tri
import scipy as sp
from octree import get_octree

# %%
lg = log.getLogger("main")
lg.setLevel(log.DEBUG)

fmt = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
tm_fmt = r"%d.%m.%y_%H.%M.%S"
timestamp = datetime.datetime.now().strftime(tm_fmt)

fh = log.FileHandler(f'logs/Main.py - {timestamp}.log')
fh.setLevel(log.DEBUG)
fh.setFormatter(fmt)

ch = log.StreamHandler(sys.stdout)
ch.setLevel(log.INFO)
ch.setFormatter(fmt)

lg.addHandler(fh)
lg.addHandler(ch)

# %%
def load_mesh(file_path: str = "") -> tri.Geometry:
    if file_path:
        mesh = tri.load(file_path)
        lg.info(f"Loaded mesh at {file_path}")
        return mesh
    else:
        raise ValueError("No defined arguments")

def samplepath(sample_name: str) -> str:
    file_path = "./data/8samples/" + sample_name + ".obj"
    return file_path

def get_stats(vertices: np.ndarray) -> dict:
    stats = {
        "Number of Vertices": vertices.shape[0],
        "Min (x, y, z)": vertices.min(axis=0),
        "Max (x, y, z)": vertices.max(axis=0),
        "Mean (x, y, z)": vertices.mean(axis=0),
        "Std. Dev. (x, y, z)": vertices.std(axis=0),
    }
    return stats

def process_mesh(mesh: tri.Geometry, base_bins: int = 1024) -> tri.Geometry:
    lg.info("Processing mesh")
    vertices = mesh.vertices
    lg.debug(get_stats(vertices))
    
    # Unit sphere norm - [-1, 1] shape - 1 radius sphere
    centroid = vertices.mean(axis=0)
    centered_vertices = vertices - centroid
    max_distance = np.linalg.norm(centered_vertices, axis=1).max()
    normalized_vertices = centered_vertices / max_distance
    mesh.vertices = normalized_vertices
    normalized_mesh = mesh.copy()

    lg.info("Applied Unit sphere norm")
    lg.debug(get_stats(normalized_vertices))
    
    # Adaptive quant 
    lg.info("Quantizing mesh")
    tree = get_octree(mesh) # sparse octree by default
    all_leaf_nodes = tree.leaves()
    lg.info("Quantization: Got octree")
    
    # We are going to use variance as our binning metric, anisotropicly
    leaf_variance = []
    for leaf in all_leaf_nodes:
        if len(leaf.indices) < 2: # no point in variance with <2 points
            variance_per_axis = np.zeros(3)
        else:
            local_vertices = mesh.vertices[leaf.indices] 
            variance_per_axis = np.var(local_vertices, axis=0) # variance for x,y,z
        leaf_variance.append((variance_per_axis, leaf))
    
    # Map leaf variances to vertex variances
    vertex_variances = np.zeros((len(normalized_vertices), 3))
    vertices_assigned = np.zeros(len(normalized_vertices), dtype=bool)
    
    # assign leaf variance to all vertices
    for variance_per_axis, leaf in leaf_variance:
        for vertex_idx in leaf.indices:
            vertex_variances[vertex_idx] = variance_per_axis
            vertices_assigned[vertex_idx] = True # for unassigned check
    
    # check for unassigned verts
    unassigned_count = np.sum(~vertices_assigned)
    if unassigned_count > 0:
        lg.info(f"Warning: {unassigned_count} vertices not assigned to any leaf")
    
    # Convert variances to bin counts
    bin_counts = np.zeros((len(normalized_vertices), 3), dtype=int)
    
    for axis in range(3):
        axis_variances = vertex_variances[:, axis]
        # Normalize to [0, 1] 
        # explanation for doing norm here again
        # norm here makes bin allocation relative to each mesh's own complexity distribution
        # We ensure we are adapting to the internal structure of this specfici mesh instead of making
        # arbitary guesses
        min_var, max_var = axis_variances.min(), axis_variances.max()
        if max_var > min_var:
            normalized_var = (axis_variances - min_var) / (max_var - min_var)
        else:
            normalized_var = np.zeros_like(axis_variances)
        
        # Map to bin counts
        min_bins = base_bins // 4
        max_bins = base_bins * 2
        # linear interpol
        bin_counts[:, axis] = (min_bins + normalized_var * (max_bins - min_bins)).astype(int)
    
    lg.info("Quantization: Calculated bins")
    
    # convert to minmax for quantization
    norm_vertices_minmax = (mesh.vertices + 1.0 ) / 2.0
    quantized_vertices = np.zeros_like(bin_counts, dtype=np.int32)
    for axis in range(3):
        bins_for_axis = bin_counts[:, axis]
        quantized_vertices[:, axis] = np.floor(norm_vertices_minmax[:,axis] * (bins_for_axis -1))
    
    lg.info("Quantization: calculated quantized verts")
    
    # dequantize
    dequantized_vertices_minmax = np.zeros_like(quantized_vertices, dtype=np.float32)
    for axis in range(3):
        bins_for_axis = bin_counts[:, axis]
        denominator = np.maximum(bins_for_axis-1, 1)
        dequantized_vertices_minmax[:, axis] = (quantized_vertices[:, axis] / denominator)
    
    lg.info("Quantization: dequantized vertices")
    
    # Reconstruct geometry
    reconstructed_vertices = (dequantized_vertices_minmax * 2.0) - 1.0
    lg.info("Quantization: reconstructed geoemetry")
    lg.debug(get_stats(reconstructed_vertices))

    mesh.vertices = reconstructed_vertices 
    
    
    return mesh, normalized_mesh, bin_counts, centroid, max_distance


def density_map_based_quant(all_leaf_nodes):
    # This is the approach I was originally going to use, 
    # before discovering anisotropic variance based binning
    density_map = []
    
    for leaf in all_leaf_nodes:
        # Calculate a density score by multiplying the leaf nodes depth
        vertex_count = len(leaf.indices)
        volume_of_the_leaf_box = np.prod(leaf.extents)
        if volume_of_the_leaf_box == 0:
            density = 0 
        else:
            current_node = leaf
            depth = 0 
            while current_node.parent is not None:
                current_node = current_node.parent
                depth += 1
            density = vertex_count * depth
        density_map.append((density, leaf))
    all_scores = [score for score, _ in density_map]
    return all_scores

def rmse(mesh1:tri.caching.TrackedArray, mesh2:tri.caching.TrackedArray):
    vert1 = mesh1.vertices
    vert2 = mesh2.vertices
    if vert1.shape != vert2.shape:
        raise ValueError("Vertex arrays must have the same shape.")
    error_vectors = vert1 - vert2
    squared_distances = np.sum(error_vectors**2, axis=1)
    ms_distance = np.mean(squared_distances)
    rmse = np.sqrt(ms_distance)
    return rmse

def mse(mesh1:tri.caching.TrackedArray, mesh2:tri.caching.TrackedArray):
    vert1 = mesh1.vertices
    vert2 = mesh2.vertices
    if vert1.shape != vert2.shape:
        raise ValueError("Vertex arrays must have the same shape.")
    error_vectors = vert1 - vert2
    squared_distances = np.sum(error_vectors**2, axis=1)
    ms_distance = np.mean(squared_distances)
    return ms_distance

def mae(mesh1:tri.caching.TrackedArray, mesh2:tri.caching.TrackedArray):
    vert1 = mesh1.vertices
    vert2 = mesh2.vertices
    if vert1.shape != vert2.shape:
        raise ValueError("Vertex arrays must have the same shape.")
    error_vectors = vert1 - vert2
    squared_distances = np.sum(error_vectors**2, axis=1)
    distances = np.sqrt(squared_distances)
    m_distance = np.mean(distances)
    return m_distance

def export(mesh:tri.caching.TrackedArray, target, nick="processed", path="./exports"):
    if not os.path.exists(path): os.mkdir(path)
    ts = datetime.datetime.now().strftime(tm_fmt)
    fh = f"exports/{nick}_{target}_{ts}.obj"
    mesh.export(fh)
    lg.info(f"Successfully exported {nick} mesh to: {fh}")


# %%
target = "talwar"
original = load_mesh(samplepath(target))
export(original, target, nick="original_trimesh")
processed, normalized, *quant_params = process_mesh(original)

lg.info(f"The calculated RMSE: {rmse(normalized, processed)}")
lg.info(f"The calculated MAE: {mae(normalized, processed)}")
lg.info(f"The calculated MSE: {mse(normalized, processed)}")
export(normalized, target, "normalized")
export(processed, target)
processed.merge_vertices()
export(processed, target, "processed_merged_vertices")

translation_amount = 2.2
processed.apply_translation([translation_amount, 0, 0])
lg.info(f"Moving processed mesh by {translation_amount:.2f} units for viewing.")

scene = tri.Scene()
scene.add_geometry(normalized)
scene.add_geometry(processed)
scene.show()


#%%