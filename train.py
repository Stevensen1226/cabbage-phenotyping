import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg
from cabbage_pheno.segmentation.dataset import CabbageDataset
from tqdm import tqdm
import argparse
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")

def compute_class_weights(dataset, num_classes=2, max_samples=100):
    """
    Estimate class weights by sampling the dataset.
    """
    logger.info("Estimating class weights...")
    counts = np.zeros(num_classes)
    
    # Sample a subset to save time
    indices = np.random.choice(len(dataset), min(len(dataset), max_samples), replace=False)
    
    for idx in indices:
        _, target = dataset[idx]
        unique, c = np.unique(target.numpy(), return_counts=True)
        for u, count in zip(unique, c):
            if u < num_classes:
                counts[u] += count
                
    total = np.sum(counts)
    if total == 0:
        return torch.ones(num_classes)
        
    # Inverse frequency weights
    weights = total / (len(counts) * counts + 1e-6)
    
    # Normalize
    weights = weights / np.sum(weights) * len(counts)
    
    logger.info(f"Estimated Class Counts: {counts}")
    logger.info(f"Calculated Class Weights: {weights}")
    
    return torch.from_numpy(weights).float()

def augment_batch(points):
    """
    Apply augmentation to a batch of points.
    points: (B, 3, N)
    """
    B, C, N = points.shape
    device = points.device
    
    # 1. Random Rotation around Z-axis
    theta = np.random.uniform(0, 2*np.pi, B)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.zeros((B, 3, 3))
    rotation_matrix[:, 0, 0] = cos_theta
    rotation_matrix[:, 0, 1] = -sin_theta
    rotation_matrix[:, 1, 0] = sin_theta
    rotation_matrix[:, 1, 1] = cos_theta
    rotation_matrix[:, 2, 2] = 1
    
    rotation_matrix = torch.from_numpy(rotation_matrix).float().to(device)
    
    # (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    points = torch.bmm(rotation_matrix, points)
    
    # 2. Random Scaling (0.8 to 1.25)
    scales = np.random.uniform(0.8, 1.25, B)
    scales = torch.from_numpy(scales).float().to(device).view(B, 1, 1)
    points = points * scales
    
    # 3. Jitter (Gaussian Noise)
    noise = torch.randn(B, C, N).to(device) * 0.01 # std=0.01
    points = points + noise
    
    return points

def calculate_metrics(pred, target, num_classes):
    """
    Calculate IoU and Accuracy.
    pred: (B, N)
    target: (B, N)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    iou_list = []
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            iou_list.append(float('nan')) # Ignore if class not present
        else:
            iou_list.append((intersection / union).item())
            
    # Accuracy
    correct = (pred == target).sum().float()
    total = target.numel()
    accuracy = (correct / total).item()
    
    return iou_list, accuracy

def validate(model, loader, device, num_classes, criterion):
    model.eval()
    total_loss = 0
    all_ious = []
    total_acc = 0
    
    with torch.no_grad():
        for points, target in loader:
            points = points.permute(0, 2, 1).to(device)
            target = target.to(device)
            
            pred_logits, _ = model(points)
            loss = criterion(pred_logits, target)
            total_loss += loss.item()
            
            pred_labels = pred_logits.max(dim=1)[1]
            ious, acc = calculate_metrics(pred_labels, target, num_classes)
            all_ious.append(ious)
            total_acc += acc
            
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    
    # Average IoU per class
    all_ious = np.array(all_ious)
    avg_class_iou = np.nanmean(all_ious, axis=0)
    mIoU = np.nanmean(avg_class_iou)
    
    return avg_loss, mIoU, avg_class_iou, avg_acc

def train(args):
    # --- Configuration ---
    DATA_PATH = args.data_path
    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    NPOINTS = args.npoints
    MODEL_SAVE_PATH = args.save_path

    # --- Check Data ---
    if not os.path.exists(os.path.join(DATA_PATH, 'train')):
        logger.error(f"Train data directory not found: {os.path.join(DATA_PATH, 'train')}")
        return

    # --- Dataset & Loader ---
    logger.info("Loading datasets...")
    train_dataset = CabbageDataset(DATA_PATH, npoints=NPOINTS, split='train')
    
    # Check for validation set, otherwise split train
    val_path = os.path.join(DATA_PATH, 'test')
    if os.path.exists(val_path) and len(os.listdir(val_path)) > 0:
        val_dataset = CabbageDataset(DATA_PATH, npoints=NPOINTS, split='test')
        logger.info(f"Using 'test' folder as validation set ({len(val_dataset)} samples).")
    else:
        logger.warning("No 'test' folder found. Splitting 'train' dataset 90/10 for validation.")
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model & Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = PointNet2SemSeg(num_classes=NUM_CLASSES).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # --- Class Weights ---
    # Calculate weights from training data (using the underlying dataset if it's a Subset)
    base_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
    class_weights = compute_class_weights(base_dataset, NUM_CLASSES).to(device)
    criterion = torch.nn.NLLLoss(weight=class_weights)

    # --- Training Loop ---
    best_miou = 0.0
    
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for points, target in progress_bar:
            # points: (B, N, 3) -> Transpose to (B, 3, N)
            points = points.permute(0, 2, 1).to(device)
            target = target.to(device)
            
            # Augmentation
            points = augment_batch(points)

            optimizer.zero_grad()
            
            # Forward
            pred, _ = model(points) # pred: (B, NumClasses, N)
            
            # Loss
            loss = criterion(pred, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        val_loss, val_miou, val_class_iou, val_acc = validate(model, val_loader, device, NUM_CLASSES, criterion)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Val Acc: {val_acc:.4f}")
        logger.info(f"Class IoUs: {val_class_iou}")

        # Save Best Model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"New best model saved to {MODEL_SAVE_PATH} (mIoU: {best_miou:.4f})")
            
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'checkpoint_latest.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PointNet++ for Cabbage Segmentation")
    parser.add_argument("--data_path", default="data/processed_blocks", help="Path to processed .npy data")
    parser.add_argument("--save_path", default="models/best_model.pth", help="Path to save trained model")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (0:background, 1:cabbage)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--npoints", type=int, default=4096, help="Number of points per block")
    
    args = parser.parse_args()
    
    # Create models directory if needed
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    train(args)
