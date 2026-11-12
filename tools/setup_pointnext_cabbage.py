from pathlib import Path

# ---- 1) 自定义数据集类 ----
ds_dir = Path('/root/autodl-tmp/PointNeXt/openpoints/dataset/cabbage')
ds_dir.mkdir(parents=True, exist_ok=True)
(ds_dir / '__init__.py').write_text('')
(ds_dir / 'cabbage.py').write_text('''
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS


@DATASETS.register_module()
class CabbageSeg(Dataset):
    """Generic loader over our .pth tuples (xyz, rgb, sem[, inst]).

    Works for both the cabbage sub-regions and Crops3D_IS plots.
    """
    num_classes = 2
    classes = ['background', 'plant']

    def __init__(self, data_root, split='train', voxel_size=0.02, voxel_max=65536,
                 transform=None, loop=1, presample=False, variable=False,
                 shuffle=True, **kwargs):
        super().__init__()
        self.split = split
        self.transform = transform
        self.loop = loop
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        sub = 'val' if split in ('val', 'validation') else split
        self.files = sorted(glob.glob(os.path.join(data_root, sub, '*.pth')))
        if not self.files:
            raise FileNotFoundError(f'no .pth in {data_root}/{sub}')
        print(f'CabbageSeg [{split}]: {len(self.files)} scenes', flush=True)

    def __len__(self):
        return len(self.files) * self.loop

    def __getitem__(self, idx):
        idx = idx % len(self.files)
        d = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        xyz = np.asarray(d[0], dtype=np.float32)
        rgb = np.asarray(d[1], dtype=np.float32)
        sem = np.asarray(d[2], dtype=np.float32).astype(np.int64)
        n = len(xyz)

        # voxel downsample (keeps statistics close to the cabbage pipeline)
        if self.voxel_size and self.voxel_size > 0:
            q = np.floor(xyz / self.voxel_size).astype(np.int64)
            _, keep = np.unique(q, axis=0, return_index=True)
            keep = np.sort(keep)
            xyz, rgb, sem = xyz[keep], rgb[keep], sem[keep]
            n = len(xyz)

        if self.voxel_max and n > self.voxel_max:
            sel = np.random.choice(n, self.voxel_max, replace=False) if self.split == 'train' \
                  else np.linspace(0, n - 1, self.voxel_max).astype(np.int64)
            xyz, rgb, sem = xyz[sel], rgb[sel], sem[sel]

        xyz = xyz - xyz.mean(0, keepdims=True)
        data = {'pos': torch.from_numpy(xyz).float(),
                'x': torch.from_numpy(rgb).float(),
                'y': torch.from_numpy(sem).long()}
        if self.transform is not None:
            data = self.transform(data)
        return data
''')

# ---- 2) 注册到 build.py 的导入表 ----
build = Path('/root/autodl-tmp/PointNeXt/openpoints/dataset/build.py')
s = build.read_text()
if 'cabbage' not in s:
    # 在文件末尾之前插入导入
    import re
    m = re.search(r'from \.(\w+) import \*', s)
    if m:
        s = s + '\nfrom .cabbage import *\n'
    else:
        s = s.replace("DATASETS = registry.Registry('dataset')",
                      "DATASETS = registry.Registry('dataset')")
        s = s + '\nfrom .cabbage import *\n'
    build.write_text(s)
    print('registered cabbage dataset in build.py')
else:
    print('build.py already imports cabbage')

# ---- 3) 配置文件 ----
cfg_dir = Path('/root/autodl-tmp/PointNeXt/cfgs/cabbage')
cfg_dir.mkdir(parents=True, exist_ok=True)
base = Path('/root/autodl-tmp/PointNeXt/cfgs/s3dis/pointnext-l.yaml')
cfg = base.read_text()
cfg = cfg.replace('num_classes: 13', 'num_classes: 2')
cfg = cfg.replace('in_channels: 4', 'in_channels: 3')
# 数据段
cfg += '''

# ---------------- Cabbage / Crops3D ----------------
dataset:
  NAME: CabbageSeg
  data_root: data/cabbage_seg
  split: train
  voxel_size: 0.02
  voxel_max: 65536

dataloader:
  batch_size: 4
  num_workers: 4

test_dataset:
  NAME: CabbageSeg
  data_root: data/cabbage_seg
  split: test
  voxel_size: 0.02
  voxel_max: 65536
'''
(cfg_dir / 'pointnext-l.yaml').write_text(cfg)
print('config written:', cfg_dir / 'pointnext-l.yaml')