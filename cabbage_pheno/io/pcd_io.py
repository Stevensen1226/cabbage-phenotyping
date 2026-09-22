import open3d as o3d
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def read_point_cloud(file_path):
    """
    Reads a point cloud file.
    Supported formats: pcd, ply, xyz, etc. (via Open3D) AND .npy
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        if file_path.lower().endswith('.npy'):
            data = np.load(file_path)
            # data shape: (N, 3) or (N, C)
            if data.ndim == 2 and data.shape[1] >= 3:
                xyz = data[:, :3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                # If colors exist
                if data.shape[1] >= 6:
                    colors = data[:, 3:6]
                    # check if colors are 0-1 or 0-255 using heuristic
                    if np.max(colors) > 1.1:
                         colors = colors / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                logger.error(f"Unknown .npy shape: {data.shape}")
                pcd = o3d.geometry.PointCloud()
        else:
            pcd = o3d.io.read_point_cloud(file_path)

        if pcd.is_empty():
            logger.warning(f"Point cloud loaded from {file_path} is empty.")
        else:
            logger.info(f"Loaded point cloud from {file_path} with {len(pcd.points)} points.")
        return pcd
    except Exception as e:
        logger.error(f"Failed to read point cloud: {e}")
        raise

def save_point_cloud(pcd, file_path, write_ascii=False):
    """
    Saves a point cloud to a file.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        success = o3d.io.write_point_cloud(file_path, pcd, write_ascii=write_ascii)
        if success:
            logger.info(f"Saved point cloud to {file_path}")
        else:
            logger.error(f"Failed to save point cloud to {file_path}")
        return success
    except Exception as e:
        logger.error(f"Error saving point cloud: {e}")
        raise
