import numpy as np
import torch

from .custom import CustomDataset


class CabbageDataset(CustomDataset):
    """甘蓝点云实例分割数据集。

    数据格式: 每个 .pth 文件为 (xyz, rgb, semantic_label, instance_label)
      - xyz: (N, 3) float32
      - rgb: (N, 3) float32 (0~1 或 -1~1)
      - semantic_label: (N,) int, 0=背景, 1=甘蓝
      - instance_label: (N,) int, -100=背景, 1~K=甘蓝植株编号 (注意: 从 1 开始)

    与官方 ScanNet 的差异:
      - 语义类 2 类 (背景 + 甘蓝), 实例类 1 类 (甘蓝)
      - 实例标签从 1 开始 (官方从 0 开始), 需 remap 到 0 开始
    """

    CLASSES = ('cabbage', )
    # 语义类别 id -> 实例类别 id (SoftGroup 内部实例类从 0 开始)
    # 语义 1 (甘蓝) 是唯一的 thing 类, 对应实例类 0
    SEMANTIC_TO_INSTANCE = {1: 0}

    def load(self, filename):
        if self.with_label:
            return torch.load(filename, weights_only=False)
        else:
            xyz, rgb = torch.load(filename, weights_only=False)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, dummy_sem_label, dummy_inst_label

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        # 甘蓝实例标签从 1 开始 (1~K), 背景为 -100
        # 官方 getInstanceInfo 假设实例标签从 0 连续, 需先 remap
        instance_label = instance_label.copy()
        inst_mask = instance_label > 0
        instance_label[inst_mask] = instance_label[inst_mask] - 1  # 1-based -> 0-based

        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # 官方 instance_cls 是语义类别 id, 需映射到实例类别 (0-based)
        # 甘蓝语义 1 (thing) -> 实例类 0
        instance_cls = [
            self.SEMANTIC_TO_INSTANCE.get(x, x) if x != -100 else x for x in instance_cls
        ]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=1.0):
        """甘蓝是刚性植物, 禁用 elastic 变形 (官方 elastic 破坏 offset 几何一致性,
        导致 BFS 聚类严重欠分割)。仅保留旋转/翻转/抖动/缩放增强。"""
        xyz_middle = self.dataAugment(xyz, True, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        # 注意: 不做 elastic 变形 (与官方 ScanNet 的关键差异)
        xyz = xyz - xyz.min(0)
        max_tries = 5
        while max_tries > 0:
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label
