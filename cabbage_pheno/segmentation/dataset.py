import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class CabbageDataset(Dataset):
    def __init__(self, root_dir, npoints=4096, split='train', classes=None):
        """
        Args:
            root_dir (str): Path to the folder containing 'train' and 'test' subfolders.
            npoints (int): Number of points to sample per block.
            split (str): 'train' or 'test'.
            classes (list): List of class names (optional).
        """
        self.root_dir = root_dir
        self.npoints = npoints
        self.split = split
        
        self.data_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist.")
            self.file_list = []
        else:
            self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
            logger.info(f"Found {len(self.file_list)} samples in {self.data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_dir, self.file_list[index])
        try:
            data = np.load(file_path) # Expected shape (N, 4) -> x,y,z,label
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            # Return a dummy sample or handle error appropriately
            return torch.zeros((self.npoints, 3)), torch.zeros((self.npoints), dtype=torch.long)
        
        points = data[:, :3]
        labels = data[:, 3]

        # Sampling strategy: Random choice with replacement if N < npoints
        if len(points) > 0:
            choice = np.random.choice(len(points), self.npoints, replace=True)
            points = points[choice, :]
            labels = labels[choice]
        else:
            points = np.zeros((self.npoints, 3))
            labels = np.zeros((self.npoints))

        # Normalization: Center the block to (0,0,0)
        # Note: We don't scale to unit sphere here usually for semantic segmentation of blocks,
        # but centering is important.
        points = points - np.mean(points, axis=0)
        
        # Convert to Tensor
        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()
