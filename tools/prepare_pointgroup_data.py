import os
import glob
import numpy as np
import torch
from pathlib import Path

def process_cabbage_to_pointgroup(input_dir, output_dir):
    """
    将 cabbage_pheno 提取的 .npy 训练数据转换为 PointGroup 官方代码支持的 .pth 格式。
    
    预期输入格式: .npy 文件。
    假设 .npy 数据为 N x M 的数组，其中包含:
    - [:, 0:3]: X, Y, Z
    - [:, 3:6]: R, G, B (如果没有颜色，就用全 0 或 Z 归一化伪彩)
    - [:, 6]: 语义标签 semantic_label (0: background, 1: cabbage)
    - [:, 7]: 实例标签 instance_label (0 通常为背景或未聚类，1, 2, 3... 为每一株甘蓝)
    
    输出格式: 包含 tuple(xyz, rgb, semantic_label, instance_label) 的 .pth 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(input_dir, '*.npy'))
    print(f"找到 {len(npy_files)} 个 .npy 文件以供转换...")
    
    for file_path in npy_files:
        data = np.load(file_path)
        
        # 提取各个通道的特征 (请根据您的实际 .npy 列索引进行修改)
        # 假设: 0,1,2=XYZ, 3,4,5=RGB, 6=Sem_Label, 7=Inst_Label
        xyz = data[:, 0:3].astype(np.float32)
        
        if data.shape[1] >= 6:
            rgb = data[:, 3:6].astype(np.float32)
            # 如果 RGB 在 [0, 1] 范围内，通常将其转换到 [-1, 1] 供 PointGroup 使用 (这取决于 PointGroup 的配置)
            # 如果是在 0-255 范围内，则保留或做相应归一化
        else:
            # 如果没有 RGB 信息，构造全 0 或者基于 Z 高度生成的颜色特征
            rgb = np.zeros_like(xyz)
            
        semantic_label = np.round(data[:, 6]).astype(np.int64)
        instance_label = np.round(data[:, 7]).astype(np.int64)
        
        # 将背景的 instance_label 统一设置为 -100 或 PointGroup 支持的 ignore_label
        # (PointGroup 在计算 instance loss 时通常忽略 -100)
        instance_label[semantic_label == 0] = -100

        # 获取保存文件名
        base_name = Path(file_path).stem
        save_path = os.path.join(output_dir, base_name + '.pth')
        
        # 保存为 .pth 格式 (供 DataLoader 使用)
        # PointGroup Dataset 中__getitem__通常期望加载出这些 Numpy array，然后再转 Tensor
        torch.save((xyz, rgb, semantic_label, instance_label), save_path)
        print(f"已转换: {base_name} -> {save_path}")

if __name__ == '__main__':
    # 路径配置
    INPUT_TRAIN_DIR = 'data/processed_blocks/train'
    OUTPUT_TRAIN_DIR = 'data/pointgroup_format/train'
    
    process_cabbage_to_pointgroup(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR)
    print("全部格式转换完成！您可以将 PointGroup 数据目录指向 ", OUTPUT_TRAIN_DIR)
