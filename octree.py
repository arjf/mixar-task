import numpy as np
import trimesh
import open3d as o3d

class _LeafNode:
    "a tree leaf"
    def __init__(self, indices):
        self.indices = np.array(indices)


class Octree:
    def __init__(self, mesh: trimesh.Geometry, max_depth: int = 10):
        self._all_leaves = []
        self._vertices = mesh.vertices # store for spatial queries
        self._build_tree(mesh, max_depth)

    def _build_tree(self, mesh: trimesh.Geometry, max_depth: int):
        # convert to pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._vertices)
        
        # build octree
        octree = o3d.geometry.Octree(max_depth=max_depth)
        octree.convert_from_point_cloud(pcd, size_expand=0.01) # insert the points into the tree.
        
        # callback function for traversal
        def visit_leaf(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                # min_bound = np.array(node_info.origin)
                # max_bound = np.array(node_info.origin) + node_info.size
                
                # in_bounds = np.all(
                #     (self._vertices >= min_bound) & (self._vertices <= max_bound),
                #     axis=1
                # )
                # vertex_indices = np.where(in_bounds)[0]
                vertex_indices = node.indices
                if vertex_indices is not None and len(vertex_indices) > 0:
                    self._all_leaves.append(_LeafNode(vertex_indices))
            return False 
        
        octree.traverse(visit_leaf)

    def leaves(self) -> list:
        return self._all_leaves


def get_octree(mesh: trimesh.Geometry, max_depth: int = 10) -> Octree:
    return Octree(mesh, max_depth=max_depth)