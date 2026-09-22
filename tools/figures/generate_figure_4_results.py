
import os
import sys
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import yaml
import copy

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabbage_pheno.viz import visualizer

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def crop_center_box(pcd, center, size):
    """Crop a cube around center."""
    min_bound = center - size/2
    max_bound = center + size/2
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(bbox)

def generate_figure_4(input_ply, output_dir):
    """
    Generate Figure 4: Global View + Local Zoom-ins
    Assume input_ply is the FINALLY SEGMENTED result (instance_colored.ply) from main.py
    Or we can run inference if needed, but better to use existing result to ensure consistency.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading segmented result: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        print("Error: Empty point cloud.")
        return

    # 1. Setup Plot
    fig = plt.figure(figsize=(20, 10), dpi=150)
    
    # Layout:
    # Top: Global View (Wide)
    # Bottom: 3 Zoom-in views
    
    # GridSpec
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    # --- Global View ---
    ax_global = fig.add_subplot(gs[0, :], projection='3d')
    
    # Downsample for global view performance
    ds_ratio = 5 # 1/5 points
    if len(points) > 100000:
        idx_global = np.random.choice(len(points), len(points)//ds_ratio, replace=False)
        pts_g = points[idx_global]
        col_g = colors[idx_global]
    else:
        pts_g, col_g = points, colors
        
    ax_global.scatter(pts_g[:,0], pts_g[:,1], pts_g[:,2], c=col_g, s=0.5, alpha=0.9)
    ax_global.set_title("(a) Global View of Segmentation Result", fontsize=16, fontweight='bold', y=-0.05)
    ax_global.set_axis_off()
    
    # Adjust global view camera
    min_b = pts_g.min(0)
    max_b = pts_g.max(0)
    # Slightly angled top-down
    ax_global.view_init(elev=70, azim=-45)
    
    # Auto-scale
    mid = (min_b + max_b) / 2
    span = (max_b - min_b).max() / 2
    ax_global.set_xlim(mid[0]-span, mid[0]+span)
    ax_global.set_ylim(mid[1]-span, mid[1]+span)
    ax_global.set_zlim(mid[2]-span, mid[2]+span)

    # --- Zoom-ins ---
    # We need to find "interesting" areas.
    # Heuristic: Find dense centers.
    # Or just pick 3 distinct locations along the diagonal.
    
    # Split the field into grid and pick populated cells
    # or just random sampling of points and crop around them.
    
    # Let's pick 3 random points that are actual vegetation (not noise)
    # Avoiding black/noise points if possible (though instance_colored usually handles this)
    
    # Pick 3 centers roughly equidistant
    # Diagonal strategy
    p1 = min_b + (max_b - min_b) * 0.25
    p2 = min_b + (max_b - min_b) * 0.50
    p3 = min_b + (max_b - min_b) * 0.75
    
    centers = [p1, p2, p3]
    labels = ["(b) Detail View 1", "(c) Detail View 2", "(d) Detail View 3"]
    zoom_size = 0.8 # 0.8m radius -> 1.6m box
    
    for i, center in enumerate(centers):
        ax_zoom = fig.add_subplot(gs[1, i], projection='3d')
        
        # Find nearest actual point to this theoretical center to ensure we look at plants
        dists = np.linalg.norm(points - center, axis=1)
        nearest_idx = np.argmin(dists)
        real_center = points[nearest_idx]
        
        # Crop
        mask = (np.abs(points[:,0]-real_center[0]) < zoom_size) & \
               (np.abs(points[:,1]-real_center[1]) < zoom_size)
               
        pts_z = points[mask]
        col_z = colors[mask]
        
        if len(pts_z) < 100:
             # Try random pick if diagonal hits empty space
             rand_idx = np.random.randint(0, len(points))
             real_center = points[rand_idx]
             mask = (np.abs(points[:,0]-real_center[0]) < zoom_size) & \
                    (np.abs(points[:,1]-real_center[1]) < zoom_size)
             pts_z = points[mask]
             col_z = colors[mask]
        
        if len(pts_z) == 0:
            ax_zoom.text(0,0,0, "Empty Region")
        else:
            ax_zoom.scatter(pts_z[:,0], pts_z[:,1], pts_z[:,2], c=col_z, s=2.0, alpha=1.0)
            
            # Formatting
            ax_zoom.set_title(labels[i], fontsize=12, fontweight='bold', y=-0.1)
            ax_zoom.set_axis_off()
            ax_zoom.view_init(elev=60, azim=-30)
            
            # Limits
            ax_zoom.set_xlim(real_center[0]-zoom_size, real_center[0]+zoom_size)
            ax_zoom.set_ylim(real_center[1]-zoom_size, real_center[1]+zoom_size)
            # Z is usually smaller range
            ax_zoom.set_zlim(real_center[2]-0.5, real_center[2]+0.5)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "paper_figure_4_results.png")
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.1)
    print(f"Figure 4 saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # It expects instance_colored.ply, NOT raw .ply
    parser.add_argument("--input", type=str, required=True, help="Path to 'instance_colored.ply' generated by main.py")
    parser.add_argument("--output", type=str, default="output/figure4")
    args = parser.parse_args()
    
    generate_figure_4(args.input, args.output)
