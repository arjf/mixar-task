import trimesh as tri
import numpy as np
import os, sys, logging as log

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