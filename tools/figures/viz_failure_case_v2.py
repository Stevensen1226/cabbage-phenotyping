
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import logging
import yaml

# Ensure project root is in path
sys.path.append(os.getcwd())

from cabbage_pheno.instance.clustering import InstanceClusterer
from cabbage_pheno.preprocess.ground import GroundSegmentor
from cabbage_pheno.io.pcd_io import read_point_cloud


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VizFailure")
logger.setLevel(logging.WARNING) # Reduce log spam

def compute_pca_features(points):
    """Compute PCA features for a set of points."""
    if len(points) < 3:
        return None
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return {
        'center': centroid,
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'v1': eigvecs[:, 0], # Principal axis
        'v2': eigvecs[:, 1],
        'v3': eigvecs[:, 2],
        'L1': np.sqrt(eigvals[0]) * 2 if eigvals[0] > 0 else 0, # Approx length
    }

def analyze_cluster_for_split(points, cluster_id):
    """
    Replicates the RSS logic to see if this cluster would split, 
    and returns visualization data if it does.
    """
    if len(points) < 50:
        return None

    feat = compute_pca_features(points)
    if feat is None:
        return None

    # Constants from config (approx defaults)
    MIN_PEAK_DIST_M = 0.15
    SKEL_BINS = 50 # Higher resolution for smoother viz
    
    # Project to 1D
    v1 = feat['v1']
    center = feat['center']
    scalars = np.dot(points - center, v1)
    
    s_min, s_max = scalars.min(), scalars.max()
    length = s_max - s_min
    
    if length < 0.2: # Too small to split
        return None

    # Bins
    nbins = SKEL_BINS
    if length / nbins < 0.01:
        nbins = max(10, int(length / 0.01))
            
    bins = np.linspace(s_min, s_max, nbins + 1)
    bin_centers_x = (bins[:-1] + bins[1:]) / 2
    
    hist, _ = np.histogram(scalars, bins=bins)
    
    # Smooth heavily to get the "Macro" shape
    density_smooth = gaussian_filter(hist.astype(float), sigma=1.5)
    
    # Peaks
    bin_width = length / nbins
    min_dist_bins = max(1, int(MIN_PEAK_DIST_M / bin_width))
    
    peaks, _ = find_peaks(density_smooth, distance=min_dist_bins, height=np.max(density_smooth)*0.1)
    
    if len(peaks) >= 2:
        # Check for valley
        first_p = peaks[0]
        last_p = peaks[-1]
        
        # We need peaks to be somewhat separated
        if last_p <= first_p:
            return None

        # Valid region between peaks
        region = density_smooth[first_p:last_p+1]
        if len(region) == 0: return None
        
        min_val = np.min(region)
        min_idx_local = np.argmin(region)
        valley_idx = first_p + min_idx_local
        
        # Calculate Valley Ratio (Valley Height / Lower Peak Height)
        h1 = density_smooth[first_p]
        h2 = density_smooth[last_p]
        lower_peak_h = min(h1, h2)
        
        # Avoid division by zero
        if lower_peak_h <= 0.01: lower_peak_h = 0.01

        valley_ratio = min_val / lower_peak_h
        
        # We want "Bad Splits" -> High Valley Ratio (Shallow Valley)
        # But not > 0.95 (that's just flat)
        
        return {
            'points': points,
            'v1': v1,
            'center': center,
            'scalars': scalars,
            's_min': s_min,
            's_max': s_max,
            'bins': bins,
            'bin_centers_x': bin_centers_x,
            'hist': hist,
            'density_smooth': density_smooth,
            'peaks': peaks,
            'valley_idx': valley_idx,
            'valley_ratio': valley_ratio,
            'feature': feat
        }
        
    return None

def main():
    # 1. Config load
    with open("configs/default.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. Iterate Files
    ply_files = glob.glob("data/unlabel/cloudR*.ply")
    if not ply_files:
        print("No ply files found in data/unlabel/")
        return
    
    candidates = []

    print(f"Scanning {len(ply_files)} files for best failure case...")

    for ply_path in ply_files:
        print(f"Checking {os.path.basename(ply_path)}...")
        pcd = read_point_cloud(ply_path)
        if pcd is None: continue

        # Ground Removal
        ground_est = GroundSegmentor(cfg)
        _, veg_pcd, _ = ground_est.segment_ground(pcd)
        
        # Initial Clustering
        clusterer = InstanceClusterer(cfg)
        labels, num_seeds = clusterer._cluster_watershed_3d(veg_pcd)
        
        points = np.asarray(veg_pcd.points)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        # Check large clusters
        for lbl in unique_labels:
            mask = labels == lbl
            size = np.sum(mask)
            if size < 5000: continue # Only large enough clusters
            
            cluster_pts = points[mask]
            
            # Simple bounding box check before expensive PCA
            dims = cluster_pts.max(0) - cluster_pts.min(0)
            if np.max(dims) < 0.3: continue # Too small spatial extent

            res = analyze_cluster_for_split(cluster_pts, lbl)
            
            if res:
                ratio = res['valley_ratio']
                # Search for ratios between 0.6 and 0.9 (Shallow)
                if 0.60 < ratio < 0.95:
                    candidates.append({
                        'ratio': ratio,
                        'size': size,
                        'file': os.path.basename(ply_path),
                        'lbl': lbl,
                        'data': res
                    })
                    print(f"  -> Candidate: {os.path.basename(ply_path)} L:{lbl} Ratio:{ratio:.3f} Size:{size}")
    
    if not candidates:
        print("No high-ratio split candidates found. Trying lower threshold...")
        # Fallback? No, the user wants "more obvious failure", meaning clearer adhesion.
        # Maybe I should just take the max ratio found regardless of threshold?
        return

    # Sort by ratio descending (shallower valleys first = 'worse' splits)
    candidates.sort(key=lambda x: x['ratio'], reverse=True)
    
    # Pick the top one
    best = candidates[0]
    print(f"\nSelected BEST Failure Candidate: {best['file']} Cluster {best['lbl']} with Ratio {best['ratio']:.3f}")
    
    # 6. Plotting
    print("Generating visualization...")
    res = best['data']
    
    fig = plt.figure(figsize=(14, 6))
    
    # (a) 3D Plot
    ax1 = fig.add_subplot(121, projection='3d')
    pts = res['points']
    
    # Color by Scalar to visualize the split direction
    scalars = res['scalars']
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=scalars, s=2, cmap='viridis', alpha=0.8)
    
    # Draw Principal Axis
    center = res['center']
    v1 = res['v1']
    length_vis = (res['s_max'] - res['s_min']) / 2.0
    p1 = center - v1 * length_vis
    p2 = center + v1 * length_vis
    
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=5, label='Principal Axis')
    
    # Draw Cut Plane
    cut_scalar = res['bin_centers_x'][res['valley_idx']]
    cut_pt_3d = center + cut_scalar * v1
    ax1.scatter([cut_pt_3d[0]], [cut_pt_3d[1]], [cut_pt_3d[2]], c='red', s=300, marker='X', label='Cut Position', zorder=100)
    
    ax1.set_title(f"(a) 3D Interlocking Cluster\nSource: {best['file']} (L{best['lbl']})", fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    
    # Set equal aspect ratio approximate
    max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0
    mid_x = (pts[:,0].max()+pts[:,0].min()) * 0.5
    mid_y = (pts[:,1].max()+pts[:,1].min()) * 0.5
    mid_z = (pts[:,2].max()+pts[:,2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # (b) Density Profile
    ax2 = fig.add_subplot(122)
    
    # Plot smoothed density
    ax2.plot(res['bin_centers_x'], res['density_smooth'], 'k-', linewidth=3.0, label='Density Profile')
    
    # Fill area
    ax2.fill_between(res['bin_centers_x'], res['density_smooth'], color='lightgray', alpha=0.5)
    
    # Peaks
    peak_x = res['bin_centers_x'][res['peaks']]
    peak_y = res['density_smooth'][res['peaks']]
    ax2.scatter(peak_x, peak_y, marker='^', s=200, color='green', zorder=5, label='Signif. Peaks')
    
    # Valley
    valley_x = res['bin_centers_x'][res['valley_idx']]
    valley_y = res['density_smooth'][res['valley_idx']]
    ax2.scatter(valley_x, valley_y, marker='v', s=200, color='red', zorder=5, label='Weak Valley')
    
    # Annotation for Valley Ratio
    ax2.annotate(f"Valley Ratio = {res['valley_ratio']:.2f}\n(Indistinct Boundary)", 
                 xy=(valley_x, valley_y), 
                 xytext=(valley_x, valley_y + np.max(res['density_smooth'])*0.25),
                 ha='center',
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=13, fontweight='bold', color='red')
                 
    ax2.set_title("(b) 1D Skeleton Density Profile", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Structure Position (m)", fontsize=12)
    ax2.set_ylabel("Point Density", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Save
    out_file = "output/failure_case_analysis_v2.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    main()
