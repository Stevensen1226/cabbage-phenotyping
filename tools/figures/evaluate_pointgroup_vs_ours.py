import numpy as np
import os
import glob
from scipy.optimize import linear_sum_assignment

def compute_iou(gt_mask, pred_mask):
    intersection = np.sum(np.logical_and(gt_mask, pred_mask))
    union = np.sum(np.logical_or(gt_mask, pred_mask))
    return intersection / union if union > 0 else 0

def evaluate_scene(gt_inst_labels, pred_inst_labels, iou_threshold=0.5):
    """
    评估单场点云的实例分割质量。
    gt_inst_labels: N 维数组 (ground truth)，背景为 0 或 -100
    pred_inst_labels: N 维数组 (预测结果)，背景为 0 或 -100
    """
    # 提取有效的实例 ID (排除背景)
    gt_ids = np.unique(gt_inst_labels)
    gt_ids = gt_ids[(gt_ids != 0) & (gt_ids != -100)]
    
    pred_ids = np.unique(pred_inst_labels)
    pred_ids = pred_ids[(pred_ids != 0) & (pred_ids != -100)]
    
    num_gt = len(gt_ids)
    num_pred = len(pred_ids)
    
    if num_gt == 0 and num_pred == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'mean_iou': 0.0, 'gt_count': 0, 'pred_count': 0}
        
    # 计算代价矩阵 (1 - IoU)
    cost_matrix = np.ones((num_gt, num_pred))
    iou_matrix = np.zeros((num_gt, num_pred))
    
    for i, gt_id in enumerate(gt_ids):
        gt_mask = (gt_inst_labels == gt_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_inst_labels == pred_id)
            iou = compute_iou(gt_mask, pred_mask)
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = 1 - iou
            
    # scipy 匈牙利算法进行最大 IoU 匹配 (Min-Cost Bipartite Matching)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    tp = 0
    matched_ious = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            tp += 1
            matched_ious.append(iou_matrix[r, c])
            
    fp = num_pred - tp
    fn = num_gt - tp
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'mean_iou': mean_iou,
        'gt_count': num_gt,
        'pred_count': num_pred
    }

def main():
    # 对比实验：读取真值和两侧的预测结果
    # 这里使用伪代码逻辑模拟文件读取，请根据实际文件路径和格式调整
    # 比如我们比较 PointGroup 和 现有传统聚类方法(Ours)
    
    gt_dir = 'e_data/test'
    ours_pred_dir = 'output/instance_eval'
    pg_pred_dir = 'PointGroup/eval/output'
    
    print("=" * 40)
    print("比较方法：[Ours (PointNet+几何聚类)] vs [PointGroup]")
    print("评测指标：Precision(准确度), Recall(召回率), F1-Score, Mean_IoU")
    print("=" * 40)
    
    # 因为此脚本作为骨架提供，下方直接用随机生成的假数据进行一次运行展示
    # 替换下面的 gt_labels 和 pred_labels 加载即可运行真实评测
    
    methods = {'Ours': None, 'PointGroup': None}
    
    for method_name in methods.keys():
        total_tp, total_fp, total_fn = 0, 0, 0
        all_mean_iou = []
        
        # ====== 模拟读取 5 个测试点云 ======
        for idx in range(5):
            # *真实场景替换区* 
            # gt_labels = np.loadtxt(f"{gt_dir}/cloudR{idx+1}_gt.txt")[:, -1]
            # pred_labels = np.load(f"path_to_{method_name}_output/cloudR{idx+1}.npy")
            
            # 以下为模拟数据生成
            N = 10000
            gt_labels = np.zeros(N)
            gt_labels[1000:3000] = 1
            gt_labels[5000:8000] = 2
            
            pred_labels = np.zeros(N)
            if method_name == 'Ours':
                # 模拟 Ours 欠分割现象较多
                pred_labels[1100:2900] = 1
                pred_labels[5500:7800] = 2
                pred_labels[9000:9500] = 3 # False positive
            else:
                # 模拟 PointGroup 稍微准一点
                pred_labels[950:3050] = 1
                pred_labels[4900:8200] = 2
            
            res = evaluate_scene(gt_labels, pred_labels, iou_threshold=0.5)
            
            total_tp += res['tp']
            total_fp += res['fp']
            total_fn += res['fn']
            if res['mean_iou'] > 0:
                all_mean_iou.append(res['mean_iou'])
        
        # 计算综合指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_instance_iou = np.mean(all_mean_iou) if all_mean_iou else 0
        
        print(f"[{method_name}] 性能指标:")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  Mean IoU  : {mean_instance_iou:.4f}")
        print("-" * 40)

if __name__ == '__main__':
    main()
