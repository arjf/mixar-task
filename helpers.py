import trimesh as tri
import numpy as np
import os, sys, datetime, logging as log

lg = log.getLogger(__name__)
tm_fmt = r"%d.%m.%y_%H.%M.%S"

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

    
def setup_logging():
    "Configure the root logger."
    lg = log.getLogger() 
    lg.setLevel(log.DEBUG)

    fmt = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tm_fmt = r"%d.%m.%y_%H.%M.%S"
    timestamp = datetime.datetime.now().strftime(tm_fmt)

    fh = log.FileHandler(f'logs/application - {timestamp}.log')
    fh.setLevel(log.DEBUG)
    fh.setFormatter(fmt)

    ch = log.StreamHandler(sys.stdout)
    ch.setLevel(log.INFO)
    ch.setFormatter(fmt)
    
    lg.addHandler(fh)
    lg.addHandler(ch)