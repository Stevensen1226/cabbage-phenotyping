import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, cfg):
        self.output_dir = cfg.get('pipeline', {}).get('output_dir', 'output')
        
    def save_colored_cloud(self, pcd, labels, filename, colormap='tab20'):
        """
        Save a point cloud with colors based on labels.
        """
        if len(pcd.points) == 0 or len(labels) == 0:
            logger.warning(f"No points to visualize for {filename}.")
            return

        if len(labels) != len(pcd.points):
            logger.error("Label count does not match point count.")
            return
            
        # Get colormap
        cmap = plt.get_cmap(colormap)
        max_label = labels.max()
        
        colors = np.zeros((len(labels), 3))
        
        # Handle noise (-1) separately -> Black or Grey
        noise_mask = (labels == -1)
        colors[noise_mask] = [0.1, 0.1, 0.1]
        
        # Color valid labels
        valid_mask = ~noise_mask
        if max_label >= 0:
            # Normalize labels to 0-1 for colormap
            # We use modulo to cycle colors if many instances
            norm_labels = (labels[valid_mask] % 20) / 20.0
            colors[valid_mask] = cmap(norm_labels)[:, :3] # RGB
            
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        full_path = os.path.join(self.output_dir, filename)
        o3d.io.write_point_cloud(full_path, pcd)
        logger.info(f"Saved visualization to {full_path}")
