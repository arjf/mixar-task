import trimesh as tri
import numpy as np
import os, sys, datetime, logging as log
from helpers import *
from octree import get_octree
from metrics import rmse, mse, mae

lg = log.getLogger(__name__)
tm_fmt = r"%d.%m.%y_%H.%M.%S"

class MeshProcessor:
    def __init__(self, mesh: tri.Geometry, base_bins = 1024):
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.base_bins = base_bins 
        
        self.norm_method = None
        self.normalized_mesh = None
        self.normalized_vertices = None
        
        # variables for quantization
        self.quant_method = None
        self.quantized_mesh = None
        self.reconstructed_vertices = None 

        # variables for denormalization 
        self.centroid = None
        self.max_distance = None
        self.v_min = None
        self.v_max = None
        
        lg.info(f"initialized MP with shape: {self.vertices.shape}")
        lg.debug(self.get_stats(self.vertices))
        
        
    
    def process(self, normalization:str, quantization:str): 
        if normalization == "unit_sphere":
            self.unit_sphere_norm()
            self.norm_method = normalization
        elif normalization == "minmax":
            self.minmax_norm()
            self.norm_method = normalization
        else:
            raise ValueError("Invalid normalization method")
        
        if quantization == "uniform":
            self.uniform_quant()
            self.quant_method = quantization
        elif quantization == "adaptive":
            self.adaptive_quant()
            self.quant_method = quantization
        else:
            raise ValueError("Invalid quantization method")
        
        if quantization:
            lg.info("Processing complete")
            return self.quantized_mesh
            
    
    def unit_sphere_norm(self):
        normalized_mesh = self.mesh.copy()
        vertices = self.mesh.vertices
        
        lg.info("Applying Unit Sphere normalization")
        # Unit sphere norm - [-1, 1] shape - 1 radius sphere
        centroid = vertices.mean(axis=0)
        self.centroid = centroid
        centered_vertices = vertices - centroid
        max_distance = np.linalg.norm(centered_vertices, axis=1).max()
        self.max_distance = max_distance
        if self.max_distance == 0:
            self.max_distance = 1e-9
        normalized_vertices = centered_vertices / max_distance
        
        normalized_mesh.vertices = normalized_vertices
        self.normalized_mesh = normalized_mesh
        self.normalized_vertices = normalized_vertices
        
        lg.info("Applied unit sphere normalization")
        lg.debug(self.get_stats(normalized_vertices))
        return normalized_mesh
    
    def minmax_norm(self):
        normalized_mesh = self.mesh.copy()
        vertices = self.mesh.vertices
        lg.info("Applying minmax normalization")

        v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
        self.v_min, self.v_max = v_min, v_max
        v_range = self.v_max - self.v_min
        v_range[v_range == 0] = 1e-9
        normalized_vertices = (vertices - v_min) / v_range
        
        normalized_mesh.vertices = normalized_vertices
        self.normalized_mesh = normalized_mesh
        self.normalized_vertices = normalized_mesh.vertices

        lg.info("Applied minmax normalization")
        lg.debug(self.get_stats(self.normalized_vertices))
        return normalized_mesh
    
    
    def uniform_quant(self):
        if self.normalized_mesh is None:
            lg.error("Run norm before quantization")
            raise AssertionError("Run norm first")
        
        quantized_mesh = self.normalized_mesh.copy()
        lg.info(f"Applying Uniform Quantization with {self.base_bins} bins")
        
        if self.norm_method == "unit_sphere":
            lg.warning("remapping to [0,1] for uniform quantization")
            norm_vertices_minmax = (self.normalized_vertices + 1.0) /2.0
        else:
            norm_vertices_minmax = self.normalized_vertices
        
        num_bins = self.base_bins
        if num_bins <=1:
            raise ValueError("base bins must be atleast 1")
        
        quantized_vertices = np.floor(norm_vertices_minmax * (num_bins - 1)).astype(np.int32)
        dequantized_vertices_minmax = quantized_vertices.astype(np.float32) / (num_bins - 1)
        
        if self.norm_method == 'unit_sphere':
            lg.info("Remapping to [-1,1] after dequantization.")
            self.reconstructed_vertices = (dequantized_vertices_minmax * 2.0) - 1.0
        else:
            self.reconstructed_vertices = dequantized_vertices_minmax
        
        quantized_mesh.vertices = self.reconstructed_vertices
        self.quantized_mesh = quantized_mesh
        self.quantized_vertices = self.quantized_mesh.vertices
        
        lg.info("Applied uniform quant")
        lg.debug(self.get_stats(self.quantized_mesh.vertices))
        return quantized_mesh
    
    def adaptive_quant(self):
        if self.normalized_mesh is None:
            lg.error("Run norm before quantization")
            raise AssertionError("Run norm first")
        
        normalized_mesh = self.normalized_mesh.copy()
        normalized_vertices = self.normalized_vertices
        
        # Adaptive quant 
        lg.info("Quantizing mesh")
        tree = get_octree(normalized_mesh) # sparse octree by default
        all_leaf_nodes = tree.leaves()
        lg.info("Quantization: Got octree")
        
        # We are going to use variance as our binning metric, anisotropicly
        leaf_variance = []
        for leaf in all_leaf_nodes:
            if len(leaf.indices) < 2: # no point in variance with <2 points
                variance_per_axis = np.zeros(3)
            else:
                local_vertices = normalized_vertices[leaf.indices] 
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
            min_bins = self.base_bins // 4
            max_bins = self.base_bins * 2
            # linear interpol
            bin_counts[:, axis] = (min_bins + normalized_var * (max_bins - min_bins)).astype(int)
        
        lg.info("Quantization: Calculated bins")
        
        # convert to minmax for quantization
        if self.norm_method == "unit_sphere":
            lg.debug("Quantization: remapping to [0,1]")
            norm_vertices_minmax = (normalized_vertices + 1.0 ) / 2.0
        else:
            lg.debug("Quantization: using [0,1] data directly")
            norm_vertices_minmax = normalized_vertices
        quantized_vertices = np.zeros_like(bin_counts, dtype=np.int32)
        for axis in range(3):
            bins_for_axis = bin_counts[:, axis]
            bins_for_axis[bins_for_axis < 2] = 2 # ensure atleast 2
            quantized_vertices[:, axis] = np.floor(norm_vertices_minmax[:,axis] * (bins_for_axis -1))
        
        lg.info("Quantization: calculated quantized verts")
        
        # dequantize
        dequantized_vertices_minmax = np.zeros_like(quantized_vertices, dtype=np.float32)
        for axis in range(3):
            bins_for_axis = bin_counts[:, axis]
            denominator = np.maximum(bins_for_axis-1, 1) # avoid div / 0
            dequantized_vertices_minmax[:, axis] = (quantized_vertices[:, axis] / denominator)
        
        lg.info("Quantization: dequantized vertices")
        
        # Reconstruct geometry
        if self.norm_method == 'unit_sphere':
            lg.debug("Quantization: remapping to [-1,1]")
            reconstructed_vertices = (dequantized_vertices_minmax * 2.0) - 1.0
        else:
            lg.debug("Quantization: already [0,1]")
            reconstructed_vertices = dequantized_vertices_minmax
        lg.info("Quantization: reconstructed geoemetry")
        lg.debug(self.get_stats(reconstructed_vertices))

        quantized_mesh = normalized_mesh.copy()
        quantized_mesh.vertices = reconstructed_vertices
        
        self.quantized_mesh = quantized_mesh
        self.reconstructed_vertices = reconstructed_vertices

        lg.info("Applied adaptive quant")
        lg.debug(self.get_stats(self.quantized_mesh.vertices))
        return quantized_mesh
    
    def denormalize(self):
        if self.reconstructed_vertices is None:
            raise ValueError("run quantization first.")
            
        lg.info(f"Denormalizing vertices using method: {self.norm_method}")
        
        if self.norm_method == 'unit_sphere':
            if self.centroid is None or self.max_distance is None:
                raise ValueError("normalization data missing.")
            denormalized_vertices = (self.reconstructed_vertices * self.max_distance) + self.centroid
            
        else:
            if self.v_min is None or self.v_max is None:
                raise ValueError("normalization data missing")
            v_range = self.v_max - self.v_min
            v_range[v_range == 0] = 1e-9
            denormalized_vertices = (self.reconstructed_vertices * v_range) + self.v_min
            
            
        final_mesh = self.quantized_mesh.copy()
        final_mesh.vertices = denormalized_vertices
        
        lg.info("Denormalization complete.")
        lg.debug(self.get_stats(denormalized_vertices))
        return final_mesh
    
    @staticmethod
    def get_metrics(mesh1:tri.caching.TrackedArray, mesh2:tri.caching.TrackedArray):
        metrics = {
            "RMSE": rmse(mesh1, mesh2),
            "MAE": mae(mesh1, mesh2),
            "MSE": mse(mesh1, mesh2)
        }
        lg.info(f"The calculated RMSE: {metrics['RMSE']}")
        lg.info(f"The calculated MAE: {metrics['MAE']}")
        lg.info(f"The calculated MSE: {metrics['MSE']}")
        return metrics
    
    @staticmethod
    def export(mesh:tri.caching.TrackedArray, target, nick="processed", path="./exports", appened_time:bool = True):
        if not os.path.exists(path): os.mkdir(path)
        ts = datetime.datetime.now().strftime(tm_fmt)
        if appened_time: fh = f"{path}/{nick}_{target}_{ts}.obj"
        else: fh = f"{path}/{nick}_{target}.obj"
        mesh.export(fh)
        lg.info(f"Successfully exported {nick} mesh to: {fh}")
    
    @staticmethod
    def get_stats(vertices: tri.caching.TrackedArray) -> dict:
        stats = {
            "Number of Vertices": vertices.shape[0],
            "Min (x, y, z)": vertices.min(axis=0),
            "Max (x, y, z)": vertices.max(axis=0),
            "Mean (x, y, z)": vertices.mean(axis=0),
            "Std. Dev. (x, y, z)": vertices.std(axis=0),
        }
        return stats
