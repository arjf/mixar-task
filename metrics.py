import trimesh as tri
import numpy as np
import os, sys, logging as log

def rmse(vert1:tri.caching.TrackedArray, vert2:tri.caching.TrackedArray, axis_wise: bool = False):
    if vert1.shape != vert2.shape:
        raise ValueError("Vertex arrays must have the same shape.")
    
    error_vectors = vert1 - vert2
    
    if axis_wise:
        error_x = error_vectors[:, 0]
        error_y = error_vectors[:, 1]
        error_z = error_vectors[:, 2]
        
        rmse_x = np.sqrt(np.mean(error_x ** 2))
        rmse_y = np.sqrt(np.mean(error_y ** 2))
        rmse_z = np.sqrt(np.mean(error_z ** 2))
        
        rmse = [rmse_x, rmse_y, rmse_z]
    else:
        squared_distances = np.sum(error_vectors**2, axis=1)
        ms_distance = np.mean(squared_distances)
        rmse = np.sqrt(ms_distance)
    
    return rmse

def mse(vert1:tri.caching.TrackedArray, vert2:tri.caching.TrackedArray, axis_wise: bool = False):
    if vert1.shape != vert2.shape:
        raise ValueError("Vertex arrays must have the same shape.")
    
    error_vectors = vert1 - vert2
    
    if axis_wise:
        error_x = error_vectors[:, 0]
        error_y = error_vectors[:, 1]
        error_z = error_vectors[:, 2]
        
        mse_x = np.mean(error_x ** 2)
        mse_y = np.mean(error_y ** 2)
        mse_z = np.mean(error_z ** 2)

        ms_distance = [mse_x, mse_y, mse_z]
    else:
        squared_distances = np.sum(error_vectors**2, axis=1)
        ms_distance = np.mean(squared_distances)
    
    return ms_distance

def mae(vert1:tri.caching.TrackedArray, vert2:tri.caching.TrackedArray, axis_wise: bool = False):
    if vert1.shape != vert2.shape:
        raise ValueError("Vertex arrays must have the same shape.")
    
    error_vectors = vert1 - vert2
    
    if axis_wise:
        error_x = error_vectors[:, 0]
        error_y = error_vectors[:, 1]
        error_z = error_vectors[:, 2]
        
        mae_x = np.mean(np.abs(error_x))
        mae_y = np.mean(np.abs(error_y))
        mae_z = np.mean(np.abs(error_z))
        
        m_distance = [mae_x, mae_y, mae_z]
    else:
        squared_distances = np.sum(error_vectors**2, axis=1)
        distances = np.sqrt(squared_distances)
        m_distance = np.mean(distances)
    
    return m_distance