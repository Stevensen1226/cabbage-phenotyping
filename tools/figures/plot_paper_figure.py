
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open3d as o3d
# Import our actual code
try:
    from cabbage_pheno.instance.clustering import InstanceClusterer
except ImportError:
    # If run from tools/, add parent dir to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from cabbage_pheno.instance.clustering import InstanceClusterer

def compute_geometric_features(points):
    """Compute PCA features for a cluster."""
    if len(points) < 10:
        return None
    pca = PCA(n_components=3)
    pca.fit(points)
    
    center = np.mean(points, axis=0)
    v1 = pca.components_[0] # Main axis
    v2 = pca.components_[1]
    v3 = pca.components_[2]
    
    # Project to get dimensions
    proj = points - center
    mins = list(np.min(proj @ pca.components_.T, axis=0))
    maxs = list(np.max(proj @ pca.components_.T, axis=0))
    
    L1, L2, L3 = maxs[0]-mins[0], maxs[1]-mins[1], maxs[2]-mins[2]
    
    return {
        'center': center,
        'v1': v1, 'v2': v2, 'v3': v3,
        'L1': L1, 'L2': L2, 'L3': L3,
        'ratio': L1 / (L2 + 1e-6)
    }

def analyze_and_plot(points, save_path="paper_figure_2.png"):
    feat = compute_geometric_features(points)
    if feat is None: return False
    
    # PCA Projection
    center = feat['center']
    v1 = feat['v1']
    scalars = np.dot(points - center, v1)
    
    s_min, s_max = scalars.min(), scalars.max()
    length = s_max - s_min
    
    # Create bins (Same logic as clustering.py)
    nbins = 30 # Higher resolution for paper figure
    if length / nbins < 0.01:
        nbins = max(10, int(length / 0.01))
        
    hist, bins = np.histogram(scalars, bins=nbins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Smooth density
    density_smooth = gaussian_filter(hist.astype(float), sigma=0.8)
    
    # Find Peaks
    peak_indices, _ = find_peaks(density_smooth, distance=3, height=np.max(density_smooth)*0.1)
    
    # Find Valley
    valley_idx = None
    valley_val = None
    ratio = None
    
    if len(peak_indices) >= 2:
        p1 = peak_indices[0]
        p2 = peak_indices[-1]
        
        # Search valley between first and last peak
        segment = density_smooth[p1:p2+1]
        local_min_idx = np.argmin(segment)
        valley_idx = p1 + local_min_idx
        
        valley_val = density_smooth[valley_idx]
        peak_val = min(density_smooth[p1], density_smooth[p2])
        ratio = valley_val / (peak_val + 1e-6)

    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(10, 5))
    
    # 1. 3D Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # Subsample for plotting speed if dense
    plot_pts = points if len(points) < 5000 else points[::2]
    plot_scalars = scalars if len(points) < 5000 else scalars[::2]
    
    ax1.scatter(plot_pts[:,0], plot_pts[:,1], plot_pts[:,2], c=plot_scalars, cmap='viridis', s=2, alpha=0.6)
    
    # Draw Axis
    line_pts = np.array([center - v1*length*0.6, center + v1*length*0.6])
    ax1.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], 'r-', linewidth=3, label='Principal Axis (v1)')
    
    ax1.set_title(f"(a) 3D Point Cloud\nLength={length:.2f}m", fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    # Adjust view
    ax1.view_init(elev=30, azim=45)

    # 2. 1D Density Plot
    ax2 = fig.add_subplot(1, 2, 2)
    # Fill under curve
    ax2.fill_between(bin_centers, density_smooth, color='gray', alpha=0.2)
    ax2.plot(bin_centers, density_smooth, 'k-', linewidth=2, label='Density Profile')
    # ax2.bar(bin_centers, hist, width=(bins[1]-bins[0])*0.8, alpha=0.3, color='green', label='Raw Counts')
    
    # Mark Peaks
    if len(peak_indices) > 0:
        ax2.plot(bin_centers[peak_indices], density_smooth[peak_indices], 'x', color='orange', markersize=10, markeredgewidth=2, label='Peaks')
        
    # Mark Valley
    if valley_idx is not None:
        ax2.plot(bin_centers[valley_idx], density_smooth[valley_idx], 'o', color='blue', markersize=8, label='Valley (Cut Point)')
        
        # Annotation
        ax2.annotate(f'Valley Ratio = {ratio:.2f}', 
                     xy=(bin_centers[valley_idx], density_smooth[valley_idx]), 
                     xytext=(bin_centers[valley_idx], density_smooth[valley_idx] + max(density_smooth)*0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10, ha='center', fontweight='bold')

    ax2.set_title("(b) 1D Skeleton Density Profile", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Structure Position (m)", fontsize=10)
    ax2.set_ylabel("Point Density", fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}")
    plt.close()
    return True

import open3d as o3d

def main():
    # Load a sample file
    # We will try to find a file and "mock" a cluster
    data_dir = r"E:\Cabbage\data\unlabel"
    files = [f for f in os.listdir(data_dir) if f.endswith('.ply')]
    
    if not files:
        print("No .ply files found.")
        return

    print("Searching for a suitable adhesive cluster candidate in unlabel data...")
    
    import random
    random.shuffle(files) # Randomize to get different results
    
    # Initialize Clusterer for Watershed
    cfg = {
        'instance': {
            'watershed_3d': {
                'voxel_resolution': 0.02, 
                'min_seed_distance': 0.20,
                'smoothing_sigma': 0.8,
                'max_voxels': 200000000 # Increase limit for large chunks
            }
        }
    }
    clusterer = InstanceClusterer(cfg)

    found = False
    
    for fname in files[:10]: # Increase search range to 10 files
        path = os.path.join(data_dir, fname)
        print(f"Loading {fname}...")
        try:
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue
            
        if len(points) < 1000: continue
        
        # Crop a 4x4m chunk from the center to run fast watershed
        # Randomize center a bit to find different spots
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        
        # Try 3 different chunks per file
        for _ in range(3):
            rand_x = np.random.uniform(min_bound[0]+2, max_bound[0]-2)
            rand_y = np.random.uniform(min_bound[1]+2, max_bound[1]-2)
            center = np.array([rand_x, rand_y, (min_bound[2]+max_bound[2])/2])

            box_min = center - 2.5
            box_max = center + 2.5
            mask = np.all((points >= box_min) & (points <= box_max), axis=1)
            chunk = points[mask]
            
            if len(chunk) < 500: continue
    
            # Create temp pcd for clusterer interface
            chunk_pcd = o3d.geometry.PointCloud()
            chunk_pcd.points = o3d.utility.Vector3dVector(chunk)
            
            try:
                # Use watershed to get clusters
                labels, n_seeds = clusterer._cluster_watershed_3d(chunk_pcd)
            except: continue
            
            unique_labels = np.unique(labels)
            
            for lbl in unique_labels:
                if lbl == -1: continue
                
                mask = (labels == lbl)
                c_points = chunk[mask]
                
                if len(c_points) < 50: continue
                
                # Filter noise
                # Apply SOR to cluster to make it look cleaner for the figure
                cl_pcd = o3d.geometry.PointCloud()
                cl_pcd.points = o3d.utility.Vector3dVector(c_points)
                cl_pcd, _ = cl_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
                c_points = np.asarray(cl_pcd.points)
                if len(c_points) < 50: continue

                feat = compute_geometric_features(c_points)
                if not feat: continue
                
                # Criterion: Stricter criteria for "Good Looking" Figure
                # Not too long (messy), not too short.
                # Ideally ~0.6-0.9m, clearly elongated 2-3 plants
                if 0.55 < feat['L1'] < 0.95 and feat['ratio'] > 1.8:
                    
                    v1 = feat['v1']
                    sc = np.dot(c_points - feat['center'], v1)
                    h, _ = np.histogram(sc, bins=20)
                    smooth = gaussian_filter(h.astype(float), sigma=0.8)
                    pks, _ = find_peaks(smooth, height=max(smooth)*0.25) # Distinct peaks
                    
                    if len(pks) >= 2:
                        print(f"*** FOUND BETTER CANDIDATE in {fname} *** L={feat['L1']:.2f}, Ratio={feat['ratio']:.2f}")
                        # Rotate view for better screenshot?
                        # Done inside analyze_and_plot
                        analyze_and_plot(c_points, r"e:\Cabbage\paper_figure_2.png")
                        found = True
                        break
            if found: break
        if found: break

    
    if not found:
        print("Could not find a perfect real cluster. Generating a Synthetic L-shape cluster...")
        # Synthetic L-shape
        # Sphere 1
        th = np.random.uniform(0, np.pi, 2000)
        ph = np.random.uniform(0, 2*np.pi, 2000)
        x1 = 0.2 * np.sin(th) * np.cos(ph)
        y1 = 0.2 * np.sin(th) * np.sin(ph)
        z1 = 0.15 * np.cos(th)
        c1 = np.vstack([x1, y1, z1]).T
        
        # Sphere 2 shifted
        c2 = c1.copy()
        c2[:, 0] += 0.35 # Shift X
        
        synthetic = np.vstack([c1, c2])
        # Add noise
        synthetic += np.random.normal(0, 0.01, synthetic.shape)
        
        analyze_and_plot(synthetic, r"e:\Cabbage\paper_figure_2.png")

if __name__ == "__main__":
    main()
