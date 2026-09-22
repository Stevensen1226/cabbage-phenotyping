import open3d as o3d
import numpy as np
import os
import argparse
import glob
from tqdm import tqdm

def process_file(input_file, output_dir, block_size, stride, label_col=3, start_count=0):
    """
    Process a single file and save blocks.
    Returns the number of blocks generated.
    """
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return 0

    print(f"Loading {input_file}...")
    try:
        pcd = o3d.io.read_point_cloud(input_file)
        points = np.asarray(pcd.points)
    except Exception as e:
        print(f"Failed to read {input_file} with Open3D: {e}")
        # Try loading as txt directly if extension matches
        if input_file.endswith('.txt') or input_file.endswith('.xyz'):
             try:
                data = np.loadtxt(input_file)
                points = data[:, :3]
             except Exception as e2:
                 print(f"Failed to load as txt: {e2}")
                 return 0
        else:
            return 0
    
    # Try to get labels
    labels = None
    
    # Heuristic: Check if it's a .txt/.xyz file with enough columns
    if input_file.endswith('.txt') or input_file.endswith('.xyz'):
        try:
            data = np.loadtxt(input_file)
            if data.shape[1] > label_col:
                points = data[:, :3]
                labels = data[:, label_col]
                print(f"Extracted labels from column {label_col}")
            else:
                print(f"Warning: File has {data.shape[1]} columns, but label_col is {label_col}.")
        except:
            pass
    
    # Handle ASCII PLY with custom scalar_label
    if input_file.endswith('.ply') and labels is None:
        try:
            # Check if it's ASCII PLY
            with open(input_file, 'rb') as f:
                header = f.read(1000).decode('ascii', errors='ignore')
            
            if 'format ascii' in header:
                # Find where header ends
                with open(input_file, 'r') as f:
                    lines = f.readlines()
                
                header_end_idx = 0
                properties = []
                for i, line in enumerate(lines):
                    if line.startswith('property'):
                        properties.append(line.split()[-1])
                    if line.strip() == 'end_header':
                        header_end_idx = i + 1
                        break
                
                # Load data part
                # We can use np.loadtxt skipping header
                data = np.loadtxt(input_file, skiprows=header_end_idx)
                
                # Find label column index in properties
                # User provided label_col is for TXT which usually has X Y Z R G B Label
                # For PLY, we should look for 'scalar_label' or 'label' or use the label_col if it matches index
                
                # Map properties to columns
                # properties list: ['x', 'y', 'z', 'red', 'green', 'blue', 'scalar_label']
                # data shape: (N, 7)
                
                # If user specified label_col=6, and properties has 7 items, index 6 is the last one.
                if len(properties) > label_col:
                    labels = data[:, label_col]
                    points = data[:, :3] # Assuming x,y,z are first 3
                    print(f"Extracted labels from ASCII PLY column {label_col} ({properties[label_col]})")
                elif 'scalar_label' in properties:
                    idx = properties.index('scalar_label')
                    labels = data[:, idx]
                    points = data[:, :3]
                    print(f"Extracted labels from ASCII PLY property 'scalar_label' (index {idx})")
                elif 'label' in properties:
                    idx = properties.index('label')
                    labels = data[:, idx]
                    points = data[:, :3]
                    print(f"Extracted labels from ASCII PLY property 'label' (index {idx})")
                    
        except Exception as e:
            print(f"Failed to parse ASCII PLY: {e}")
            
    if labels is None:
        print(f"Warning: Could not automatically extract labels from {os.path.basename(input_file)}.")
        print("Generating DUMMY labels for demonstration purposes...")
        labels = np.zeros(len(points))
        # Mock some data: z < 0.1 is ground/background (0), else cabbage (1)
        labels[points[:, 2] < 0.1] = 0
        labels[points[:, 2] >= 0.1] = 1

    print(f"Total points: {len(points)}")
    
    # Create output dirs
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    # Grid slicing
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    count = start_count
    blocks_generated = 0
    
    # Iterate over X and Y
    for x in np.arange(min_bound[0], max_bound[0], stride):
        for y in np.arange(min_bound[1], max_bound[1], stride):
            # Define block bounds
            x_min, x_max = x, x + block_size
            y_min, y_max = y, y + block_size
            
            # Mask points in this block
            mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                   (points[:, 1] >= y_min) & (points[:, 1] < y_max)
            
            block_points = points[mask]
            block_labels = labels[mask]
            
            if len(block_points) > 100: # Skip empty or sparse blocks
                # Save as .npy
                # Format: [x, y, z, label]
                save_data = np.column_stack((block_points, block_labels))
                # Use filename prefix to avoid overwriting if multiple files processed
                file_prefix = os.path.splitext(os.path.basename(input_file))[0]
                save_path = os.path.join(train_dir, f"{file_prefix}_block_{count:04d}.npy")
                np.save(save_path, save_data)
                count += 1
                blocks_generated += 1
                
    print(f"Generated {blocks_generated} blocks from {os.path.basename(input_file)}")
    return blocks_generated

def prepare_data(args):
    """
    Slices large labeled point clouds into smaller blocks for training.
    Input: Folder containing labeled files OR single file path.
    Output: Folder with .npy files
    """
    input_path = args.input
    output_dir = args.output
    block_size = args.block_size
    stride = args.stride
    
    files_to_process = []
    
    if os.path.isdir(input_path):
        # If input is a directory, look for common point cloud formats
        extensions = ['*.ply', '*.pcd', '*.txt', '*.xyz']
        for ext in extensions:
            files_to_process.extend(glob.glob(os.path.join(input_path, ext)))
    elif os.path.isfile(input_path):
        files_to_process.append(input_path)
    else:
        # Try glob pattern directly (e.g. data/raw/*.txt)
        files_to_process = glob.glob(input_path)
        
    if not files_to_process:
        print(f"No files found at {input_path}")
        return

    print(f"Found {len(files_to_process)} files to process.")
    
    total_blocks = 0
    for f in files_to_process:
        # We reset count for each file, but include filename in output to avoid collision
        # Or we could pass a global counter. Let's use local count + filename prefix.
        total_blocks += process_file(f, output_dir, block_size, stride, label_col=args.label_col)
        
    print(f"All done. Total blocks generated: {total_blocks}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for PointNet++ training")
    parser.add_argument("--input", required=True, help="Input file, folder, or glob pattern (e.g. data/raw/*.txt)")
    parser.add_argument("--output", default="data/processed_blocks", help="Output directory")
    parser.add_argument("--block_size", type=float, default=1.0, help="Size of block in meters")
    parser.add_argument("--stride", type=float, default=1.0, help="Stride for sliding window")
    parser.add_argument("--label_col", type=int, default=3, help="Column index for labels (0-based)")
    
    args = parser.parse_args()
    prepare_data(args)
