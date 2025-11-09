# %%
import os

import numpy as np
import trimesh as tri


# %%
def load_mesh(file_path: str = "") -> tri.Geometry:
    if file_path:
        return tri.load(file_path)
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

def process_mesh(mesh: tri.Geometry) -> tri.Geometry:
    
    vertices = mesh.vertices
    print(get_stats(vertices))
    
    # Unit sphere norm - [-1, 1] shape - 1 radius sphere
    centroid = vertices.mean(axis=0)
    centered_vertices = vertices - centroid
    max_distance = np.linalg.norm(centered_vertices, axis=1).max()
    normalized_vertices = centered_vertices / max_distance
    
    return mesh


# %%
branch = load_mesh(samplepath("branch"))
branch.show()
# %%
print(branch.visual.uv.shape)

# %%
processed_branch = process_mesh(branch)
processed_branch.show()

#%%
print(branch.visual.uv.shape)