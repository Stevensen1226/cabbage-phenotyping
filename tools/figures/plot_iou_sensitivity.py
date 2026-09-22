
import os
import sys
import yaml
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.getcwd())

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.preprocess import PointCloudCleaner, GroundSegmentor
from cabbage_pheno.segmentation import PointNetSegmentor
from cabbage_pheno.instance import InstanceClusterer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("IoUSensitivity")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_ground_truth(file_path, label_col=6, instance_col=7):
    # Try different suffixes
    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Common patterns
    candidates = [
        os.path.join(dir_name, base_name + ".txt"),
        os.path.join(dir_name, base_name + "_gt.txt"),
        file_path.replace(".ply", ".txt")
    ]
    
    gt_path = None
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            gt_path = p
            break
            
    if gt_path is None:
        return None, None, None

    try:
        data = np.loadtxt(gt_path)
        points = data[:, 0:3]
        sem_labels = (data[:, label_col] > 0).astype(int)
        inst_labels = data[:, instance_col].astype(int)
        return points, sem_labels, inst_labels
    except Exception as e:
        logger.error(f"GT Load Error: {e}")
        return None, None, None

def compute_matches(pred_inst, gt_inst, iou_thresh):
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]
    
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]
    
    tp = 0
    matched_gt = set()
    
    for pid in pred_ids:
        p_mask = (pred_inst == pid)
        p_area = p_mask.sum()
        
        best_iou = 0
        best_gid = -1
        
        # Optimization: Only check GTs that overlap
        gt_under = gt_inst[p_mask]
        cands = np.unique(gt_under)
        cands = cands[cands > 0]
        
        for gid in cands:
            intersection = np.sum((gt_under == gid))
            # Get union total area without constructing full mask if possible?
            g_area = np.sum(gt_inst == gid)
            
            union = p_area + g_area - intersection
            iou = intersection / union
            
            if iou > best_iou:
                best_iou = iou
                best_gid = gid
                
        if best_iou >= iou_thresh:
            if best_gid not in matched_gt:
                tp += 1
                matched_gt.add(best_gid)
                
    return tp, len(pred_ids), len(gt_ids)

def main():
    # settings
    CONFIG_PATH = "configs/default.yaml"
    DATA_DIR = "e_data/test" # As per user history
    OUTPUT_PLOT = "output/iou_sensitivity_curve.png"
    
    if not os.path.exists(DATA_DIR):
        print(f"Data dir {DATA_DIR} not found.")
        return

    # Load Config
    cfg = load_config(CONFIG_PATH)
    
    # Init Pipeline
    cleaner = PointCloudCleaner(cfg)
    ground_seg = GroundSegmentor(cfg)
    clusterer = InstanceClusterer(cfg)
    
    # Check segmentation mode
    seg_method = cfg.get('segmentation', {}).get('method', 'none')
    print(f"Running Analysis with Segmentation Method: {seg_method}")
    if seg_method == 'pointnet':
        segmentor = PointNetSegmentor(cfg)
    else:
        segmentor = None

    # Load Files
    files = glob.glob(os.path.join(DATA_DIR, "*.ply")) # Assuming main files are .ply
    # Filter out _gt files just in case
    files = [f for f in files if "_gt" not in f]
    
    if not files:
        print("No files found.")
        return

    print(f"found {len(files)} files. Pre-calculating predictions...")
    
    # Store (Pred, GT) pairs
    results_cache = []
    
    for f in tqdm(files):
        # 1. Load Data
        pcd = read_point_cloud(f)
        if pcd is None: continue
        
        gt_pts, gt_sem, gt_inst = load_ground_truth(f)
        if gt_inst is None: continue
        
        # 2. Pipeline
        # A. Clean
        if cfg['preprocess']['sor_enable']:
            pcd = cleaner.remove_outliers(pcd)
        if len(pcd.points) == 0: continue
            
        points = np.asarray(pcd.points)
        
        # B. Ground
        ground_pcd, non_ground_pcd, _ = ground_seg.segment_ground(pcd)
        
        # C. Segment (PointNet or Heuristic)
        # Simplify: Just stick to the "Vegetation Extraction" logic used in evaluate
        # Identify "Cabbage Points"
        if len(non_ground_pcd.points) == 0:
            continue
            
        veg_points = np.asarray(non_ground_pcd.points)
        
        # Instance Clustering
        if len(veg_points) < 50:
            pred_inst_local = np.zeros(len(veg_points))
        else:
            # Cluster ONLY the vegetation
            pred_inst_local, _ = clusterer.cluster(non_ground_pcd)
            
        # 3. Alignment
        # We need to compute IoU on the *Processed Points* (veg_points)
        # So we align GT to veg_points
        tree = cKDTree(gt_pts)
        _, indices = tree.query(veg_points, k=1)
        aligned_gt_inst = gt_inst[indices]
        
        results_cache.append((pred_inst_local, aligned_gt_inst))

    print(f"Data collected. Simulating thresholds...")
    
    thresholds = np.arange(0.25, 0.85, 0.05)
    recalls = []
    precisions = []
    f1s = []
    
    for thresh in thresholds:
        total_tp = 0
        total_pred = 0
        total_gt = 0
        
        for pred, gt in results_cache:
            tp, n_p, n_g = compute_matches(pred, gt, thresh)
            total_tp += tp
            total_pred += n_p
            total_gt += n_g
            
        rec = total_tp / total_gt if total_gt else 0
        prec = total_tp / total_pred if total_pred else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
        print(f"IoU={thresh:.2f} | R={rec:.3f} P={prec:.3f} F1={f1:.3f}")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, 'b-o', label='Recall (Detection Rate)', linewidth=2)
    plt.plot(thresholds, f1s, 'g--s', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, 'r-.^', label='Precision', linewidth=2, alpha=0.5)
    
    plt.title("Impact of IoU Threshold on Segmentation Performance", fontsize=14, fontweight='bold')
    plt.xlabel("IoU Threshold (Overlap Requirement)", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0, 1.05)
    
    # Annotate the drop
    idx_50 = np.argmin(np.abs(thresholds - 0.50))
    idx_75 = np.argmin(np.abs(thresholds - 0.75))
    
    if idx_50 < len(recalls) and idx_75 < len(recalls):
        val_50 = recalls[idx_50]
        val_75 = recalls[idx_75]
        drop = val_50 - val_75
        
        plt.annotate(f"Detection Stability\n(Rec={val_50:.2f})",
                     xy=(thresholds[idx_50], val_50),
                     xytext=(thresholds[idx_50], val_50+0.18),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     ha='center', fontsize=10, color='green')

        plt.annotate(f"“Boundary-stringent regime\n(-{drop*100:.1f}%)",
                     xy=(thresholds[idx_75], val_75),
                     xytext=(thresholds[idx_75], val_75+0.15),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     ha='center', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
