import os
import glob
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

# Import pointgroup_ops (same path as scannet)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.pointgroup_ops.functions import pointgroup_ops


class CabbageDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.data_root = cfg.data_root
        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        # 查找所有的 .pth 数据文件
        all_files = sorted(glob.glob(os.path.join(self.data_root, '*.pth')))
        
        # 按 80/20 划分 train/val
        np.random.seed(cfg.manual_seed if hasattr(cfg, 'manual_seed') else 123)
        idx = np.random.permutation(len(all_files))
        split_n = max(1, int(len(all_files) * 0.8))

        if split == 'train':
            self.files = [all_files[i] for i in idx[:split_n]]
        else:
            self.files = [all_files[i] for i in idx[split_n:]]

        # Pre-load all data (small dataset)
        self.data_list = []
        for f in self.files:
            xyz, rgb, semantic_label, instance_label = torch.load(f, weights_only=False)
            self.data_list.append((xyz, rgb, semantic_label, instance_label))

        print(f"CabbageDataset [{split}]: {len(self.files)} samples")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def getInstanceInfo(xyz, instance_label):
    '''
    :param xyz: (n, 3)
    :param instance_label: (n), int, (0~nInst-1, -100)
    :return: instance_num, dict
    '''
    instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
    instance_pointnum = []
    instance_num = int(instance_label.max()) + 1
    for i_ in range(instance_num):
        inst_idx_i = np.where(instance_label == i_)
        xyz_i = xyz[inst_idx_i]
        min_xyz_i = xyz_i.min(0)
        max_xyz_i = xyz_i.max(0)
        mean_xyz_i = xyz_i.mean(0)
        instance_info_i = instance_info[inst_idx_i]
        instance_info_i[:, 0:3] = mean_xyz_i
        instance_info_i[:, 3:6] = min_xyz_i
        instance_info_i[:, 6:9] = max_xyz_i
        instance_info[inst_idx_i] = instance_info_i
        instance_pointnum.append(inst_idx_i[0].size)
    return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


def dataAugment(xyz, jitter=False, flip=False, rot=False):
    m = np.eye(3)
    if jitter:
        m += np.random.randn(3, 3) * 0.01
    if flip:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
    if rot:
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                           [-math.sin(theta), math.cos(theta), 0],
                           [0, 0, 1]])
    return np.matmul(xyz, m)


def crop(xyz, full_scale, max_npoint):
    xyz_offset = xyz.copy()
    valid_idxs = (xyz_offset.min(1) >= 0)
    full_scale_arr = np.array([full_scale[1]] * 3)
    room_range = xyz.max(0) - xyz.min(0)
    while valid_idxs.sum() > max_npoint:
        offset = np.clip(full_scale_arr - room_range + 0.001, None, 0) * np.random.rand(3)
        xyz_offset = xyz + offset
        valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale_arr).sum(1) == 3)
        full_scale_arr[:2] -= 32
    return xyz_offset, valid_idxs


def getCroppedInstLabel(instance_label, valid_idxs):
    instance_label = instance_label[valid_idxs]
    j = 0
    while j < instance_label.max():
        if len(np.where(instance_label == j)[0]) == 0:
            instance_label[instance_label == instance_label.max()] = j
        j += 1
    return instance_label


def cabbage_collate_fn(batch):
    '''
    Collate function that matches scannet format:
    returns dict with: locs, voxel_locs, p2v_map, v2p_map, locs_float, feats,
                       labels, instance_labels, instance_info, instance_pointnum,
                       id, offsets, spatial_shape
    '''
    from util.config import cfg

    locs = []
    locs_float = []
    feats = []
    labels = []
    instance_labels = []
    instance_infos = []
    instance_pointnum = []
    batch_offsets = [0]
    total_inst_num = 0
    ids = []

    for i, (xyz_origin, rgb, label, instance_label) in enumerate(batch):
        # Convert to numpy if torch
        if isinstance(xyz_origin, torch.Tensor):
            xyz_origin = xyz_origin.numpy()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if isinstance(instance_label, torch.Tensor):
            instance_label = instance_label.numpy()

        # Data augmentation (train only)
        xyz_middle = dataAugment(xyz_origin, jitter=True, flip=True, rot=True)

        # Scale to voxel space
        xyz = xyz_middle * cfg.scale
        xyz -= xyz.min(0)

        # Crop
        xyz, valid_idxs = crop(xyz, cfg.full_scale, cfg.max_npoint)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        label = label[valid_idxs]
        instance_label = getCroppedInstLabel(instance_label, valid_idxs)

        # Get instance info
        inst_num, inst_infos = getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]
        inst_pointnum = inst_infos["instance_pointnum"]

        instance_label[np.where(instance_label != -100)] += total_inst_num
        total_inst_num += inst_num

        batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

        locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i),
                                torch.from_numpy(xyz).long()], 1))
        locs_float.append(torch.from_numpy(xyz_middle))
        feats.append(torch.from_numpy(rgb))
        labels.append(torch.from_numpy(label))
        instance_labels.append(torch.from_numpy(instance_label))
        instance_infos.append(torch.from_numpy(inst_info))
        instance_pointnum.extend(inst_pointnum)
        ids.append(i)

    # Merge all scenes
    batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
    locs = torch.cat(locs, 0)
    locs_float = torch.cat(locs_float, 0).to(torch.float32)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0).long()
    instance_labels = torch.cat(instance_labels, 0).long()
    instance_infos = torch.cat(instance_infos, 0).to(torch.float32)
    instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)
    ids = torch.tensor(ids)

    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale[0], None)

    # Voxelize
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(
        locs, cfg.batch_size, cfg.mode)

    return {
        'locs': locs, 'voxel_locs': voxel_locs,
        'p2v_map': p2v_map, 'v2p_map': v2p_map,
        'locs_float': locs_float, 'feats': feats,
        'labels': labels, 'instance_labels': instance_labels,
        'instance_info': instance_infos,
        'instance_pointnum': instance_pointnum,
        'id': ids, 'offsets': batch_offsets,
        'spatial_shape': spatial_shape
    }

