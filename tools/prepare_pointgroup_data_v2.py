import os
import glob
import numpy as np
import torch
import open3d as o3d
from pathlib import Path

def process_ply_txt_to_pointgroup(input_dir, output_dir):
    """
    将包含 .ply (点云) 和 .txt (标注) 的数据转换为 PointGroup 支持的 .pth 格式。
    数据来源: E:\\Cabbage\\e_data\\train
    
    规则假设:
    1. ply 文件和 txt 文件前缀名称相同，例如 cloud01.ply 和 cloud01.txt   
    2. txt 文件的行数 = ply 文件的点数。
    3. txt 文件中包含实例标注 (例如 0 为背景，1, 2, ... 为甘蓝单株)。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 ply 文件
    ply_files = glob.glob(os.path.join(input_dir, '*.ply'))
    print(f"在 {input_dir} 找到 {len(ply_files)} 个 .ply 文件以供转换...")
    
    for ply_path in ply_files:
        base_name = Path(ply_path).stem
        
        # 尝试匹配对应的 txt 标注文件
        # 支持 {base_name}.txt 或者 {base_name}_gt.txt 这种主流命名法
        txt_path = os.path.join(input_dir, f"{base_name}.txt")
        if not os.path.exists(txt_path):
            txt_path = os.path.join(input_dir, f"{base_name}_gt.txt")
            if not os.path.exists(txt_path):
                print(f"[警告] 找不到 {base_name} 对应的 .txt 文件，跳过此文件。")
                continue
        
        # 1. 读取点云 (.ply)
        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points).astype(np.float32)
        
        # 读取颜色，如果没有颜色就置 0
        if len(pcd.colors) > 0:
            # 默认 open3d 读入颜色在 [0, 1] 之间，可以转成 [-1, 1] 适应常见网络
            rgb = (np.asarray(pcd.colors) * 2.0 - 1.0).astype(np.float32)
        else:
            rgb = np.zeros_like(xyz)
            
        # 2. 读取标注 (.txt)
        labels = np.loadtxt(txt_path)
        
        if labels.ndim < 2:
            print(f"[错误] {base_name}.txt 列数不足，无法提取语义和实例标签，跳过！")
            continue
            
        # 根据设定的规则：倒数第二列为语义，倒数第一列为实例
        semantic_label = np.round(labels[:, -2]).astype(np.int64)
        instance_label = np.round(labels[:, -1]).astype(np.int64)
            
        # PointGroup 计算实例 loss 的机制：背景点的 instance id 必须设置为特定的 ignore 标签（通常是 -100）
        # 将原始数据中背景的 -1 统一转为 -100
        instance_label[semantic_label == 0] = -100

        # 防御性检查：确保点云点数与标签行数一致
        if xyz.shape[0] != instance_label.shape[0]:
            print(f"[错误] 点数不匹配！{base_name}.ply 点数: {xyz.shape[0]}, 标签行数: {instance_label.shape[0]}")
            continue

        # 3. 保存为 DataLoader 支持的 .pth
        save_path = os.path.join(output_dir, base_name + '.pth')
        torch.save((xyz, rgb, semantic_label, instance_label), save_path)
        print(f"已转换: {base_name} -> {save_path}")

if __name__ == '__main__':
    # 路径配置
    INPUT_TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'e_data', 'train')
    OUTPUT_TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'e_data', 'pointgroup_format', 'train')
    
    process_ply_txt_to_pointgroup(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR)
    print("全部格式转换完成！您可以将 PointGroup数据目录指向 ", OUTPUT_TRAIN_DIR)
