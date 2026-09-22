import open3d as o3d
import logging

logger = logging.getLogger(__name__)

class PointCloudCleaner:
    def __init__(self, cfg):
        """
        Initialize with configuration dictionary.
        Expects cfg['pipeline']['voxel_size'] and cfg['preprocess']
        """
        self.voxel_size = cfg.get('pipeline', {}).get('voxel_size', 0.005)
        self.sor_cfg = cfg.get('preprocess', {})
        
    def voxel_downsample(self, pcd):
        """
        Apply voxel grid downsampling.
        """
        # Check if point cloud is small enough to skip downsampling
        if len(pcd.points) < 200000:
            logger.info(f"Point cloud has {len(pcd.points)} points (< 100k). Skipping downsampling.")
            return pcd

        logger.info(f"Downsampling with voxel size: {self.voxel_size}")
        down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        logger.info(f"Points after downsampling: {len(down_pcd.points)}")
        return down_pcd

    def remove_outliers(self, pcd):
        """
        Apply Statistical Outlier Removal (SOR).
        """
        enable = self.sor_cfg.get('sor_enable', True)
        if not enable:
            logger.info("Skipping outlier removal (SOR disabled by config).")
            return pcd

        nb_neighbors = self.sor_cfg.get('sor_neighbors', 20)
        std_ratio = self.sor_cfg.get('sor_std_ratio', 2.0)

        if nb_neighbors is None or nb_neighbors <= 0 or std_ratio is None or std_ratio <= 0:
            logger.info("Skipping outlier removal (invalid SOR params).")
            return pcd
        
        logger.info(f"Removing outliers (SOR): nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        # Select the inliers
        inlier_cloud = pcd.select_by_index(ind)
        logger.info(f"Points after outlier removal: {len(inlier_cloud.points)}")
        return inlier_cloud
