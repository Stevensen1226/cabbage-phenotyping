
import os
import sys
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabbage_pheno.instance import clustering
from cabbage_pheno.preprocess import cleaning, ground
from cabbage_pheno.viz import visualizer

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_colors_from_labels(labels, colormap='tab20'):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    colors = np.zeros((len(labels), 3))
    noise_mask = (labels == -1)
    colors[noise_mask] = [0.1, 0.1, 0.1]
    valid_mask = ~noise_mask
    if np.any(valid_mask):
        norm_labels = (labels[valid_mask] % 20) / 20.0
        colors[valid_mask] = cmap(norm_labels)[:, :3]
    return colors

def run_step_by_step_comparison(input_path, config_path, output_dir):
    """
    Generate intermediate results for Figure 3:
    1. Preprocessed (Ground Removed)
    2. Coarse Segmentation (Watershed Only)
    3. Refined Segmentation (Recursive Split + PostProc)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"[1/4] Loading configuration from {config_path}...")
    cfg = load_config(config_path)
    
    # Force enable PCA split to ensure we have a 'Refined' stage
    # But for the Coarse stage, we will temporarily disable it or call the internal method
    cfg['instance']['pca_split']['enable'] = True
    
    # 1. Load Data
    print(f"[2/4] Loading Point Cloud: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        print("Error: Empty point cloud.")
        return

    # ---------------------------------------------------------
    # Step A: Preprocessing (Simulate main.py pipeline)
    # ---------------------------------------------------------
    print(" >>> Step A: Preprocessing (Cleaning + Ground Removal)...")
    
    # ROI Crop if needed
    roi_cfg = cfg.get('preprocess', {}).get('roi', {})
    if roi_cfg.get('enable', False):
        pcd = cleaning.crop_roi(pcd, roi_cfg.get('range', [0, 100, 0, 100, -2, 2]))

    # SOR
    sor_cfg = cfg.get('preprocess', {}).get('sor', {})
    if sor_cfg.get('enable', False):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=sor_cfg.get('nb_neighbors', 50),
            std_ratio=sor_cfg.get('std_ratio', 2.0)
        )

    # Downsample
    # voxel_size = cfg.get('voxel_size', 0.005)
    # pcd = pcd.voxel_down_sample(voxel_size)

    # Ground Removal (CSF)
    ground_cfg = cfg.get('preprocess', {}).get('ground', {})
    if ground_cfg.get('enable', True):
        # Use the GroundSegmentor class instead of direct function call
        # ground_idx, veg_idx = ground.classify_ground_csf(...)
        
        # instantiate segmentor
        segmentor = ground.GroundSegmentor(cfg)
        # Note: segment_ground returns (ground_pcd, non_ground_pcd, plane_model) 
        # based on inspecton of ground.py line 106.
        ground_pcd, veg_pcd, _ = segmentor.segment_ground(pcd)
        
        # Colorize for Step A Visualization
        # Ground = Brown/Gray, Veg = Green
        ground_pcd.paint_uniform_color([0.6, 0.5, 0.4]) # Soil color
        veg_pcd.paint_uniform_color([0.0, 0.8, 0.0])   # Green
        
        step_a_pcd = ground_pcd + veg_pcd
        output_path_a = os.path.join(output_dir, "fig3_step1_preprocess.ply")
        o3d.io.write_point_cloud(output_path_a, step_a_pcd)
        print(f" Saved Step 1 to {output_path_a}")
        
    else:
        veg_pcd = pcd # Assume input is already veg?

    # ---------------------------------------------------------
    # Step B: Coarse Segmentation (Watershed Only)
    # ---------------------------------------------------------
    print(" >>> Step B: Coarse Segmentation (Watershed Only)...")
    clusterer = clustering.InstanceClusterer(cfg)
    
    # We call the internal method _cluster_watershed_3d directly to get the "Coarse" result
    # note: clustering.py line 85: labels, num_seeds = self._cluster_watershed_3d(pcd)
    coarse_labels, _ = clusterer._cluster_watershed_3d(veg_pcd)
    
    # Create colored visualization for coarse result
    # Filter out -1 (noise)
    coarse_pcd = copy.deepcopy(veg_pcd)
    # Colorize
    # coarse_colors = visualizer.labels_to_colors(coarse_labels)
    coarse_colors = get_colors_from_labels(coarse_labels)
    coarse_pcd.colors = o3d.utility.Vector3dVector(coarse_colors)
    
    output_path_b = os.path.join(output_dir, "fig3_step2_coarse.ply")
    o3d.io.write_point_cloud(output_path_b, coarse_pcd)
    print(f" Saved Step 2 to {output_path_b}")

    # ---------------------------------------------------------
    # Step C: Refined Segmentation (Recursive Split)
    # ---------------------------------------------------------
    print(" >>> Step C: Refined Segmentation (Full Pipeline)...")
    
    # Now run the specific split and merge functions on the coarse labels
    points = np.asarray(veg_pcd.points)
    
    # 2. Enhanced Split
    refined_labels = clusterer._apply_skeleton_split(coarse_labels, points)
    
    # 3. Merge Fragments
    refined_labels = clusterer._merge_fragments(refined_labels, points)
    
    # 4. Cleanup
    refined_labels = clusterer._cleanup_tiny_fragments(refined_labels, points, min_size=3000) # Ensure sync with config if possible
    refined_labels = clusterer._simple_filter(refined_labels)
    final_labels, _ = clusterer._renumber_labels(refined_labels)
    
    # Colorize
    refined_pcd = copy.deepcopy(veg_pcd)
    # refined_colors = visualizer.labels_to_colors(final_labels)
    refined_colors = get_colors_from_labels(final_labels)
    refined_pcd.colors = o3d.utility.Vector3dVector(refined_colors)
    
    output_path_c = os.path.join(output_dir, "fig3_step3_refined.ply")
    o3d.io.write_point_cloud(output_path_c, refined_pcd)
    print(f" Saved Step 3 to {output_path_c}")

    # ---------------------------------------------------------
    # Generate 2D Panel Image using Matplotlib
    # ---------------------------------------------------------
    print(" >>> Generating 2D Comparison Figure (Figure 3)...")
    generate_matplotlib_figure(
        [step_a_pcd, coarse_pcd, refined_pcd],
        ["(a) Preprocessing\n(Ground Removal)", "(b) Coarse Segmentation\n(3D Watershed)", "(c) Recursive Refinement\n(Final Result)"],
        os.path.join(output_dir, "paper_figure_3_comparison.png")
    )


def generate_matplotlib_figure(pcds, titles, save_path):
    fig = plt.figure(figsize=(18, 6), dpi=150)
    
    # Setup view angle
    elev = 60
    azim = -45
    
    for i, (pcd, title) in enumerate(zip(pcds, titles)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Decimate for faster rendering if too large
        if len(points) > 50000:
            choice = np.random.choice(len(points), 50000, replace=False)
            points = points[choice]
            colors = colors[choice]
            
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=0.5, alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold', y=-0.05)
        
        # Hide axes for cleaner look
        ax.set_axis_off()
        
        # Set standardized axis limits to keep zoom consistent
        # We use the bounds of the first PCD (Preprocessed) as reference
        if i == 0:
            min_bound = points.min(0)
            max_bound = points.max(0)
            center = (max_bound + min_bound) / 2
            range_max = (max_bound - min_bound).max()
        
        ax.set_xlim(center[0] - range_max/2, center[0] + range_max/2)
        ax.set_ylim(center[1] - range_max/2, center[1] + range_max/2)
        ax.set_zlim(center[2] - range_max/2, center[2] + range_max/2)
        
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input point cloud (raw ply or processed block)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="output/figure3")
    args = parser.parse_args()
    
    run_step_by_step_comparison(args.input, args.config, args.output)
