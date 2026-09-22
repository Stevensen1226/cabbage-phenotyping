#!/usr/bin/env python3
"""
训练可视化脚本 —— 从日志文件中提取 loss 曲线，在终端/文件中查看
用法:
    python tools/view_training.py                          # 查看最新训练
    python tools/view_training.py --plot                   # 生成 PNG 图表
"""
import re
import sys
import os
import glob
import argparse

def parse_log(log_path):
    """解析训练日志，提取 loss"""
    epochs = []
    train_losses = []
    val_losses = []
    semantic_losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # epoch: 1/384, train loss: 1.2345
            m = re.search(r'epoch:\s+(\d+)/\d+,\s+train loss:\s+([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
            
            # epoch: 32/384, val loss: 6.0859
            m = re.search(r'epoch:\s+(\d+)/\d+,\s+val loss:\s+([\d.]+)', line)
            if m:
                if len(val_losses) == 0 or int(m.group(1)) != (val_losses[-1][0] if val_losses else -1):
                    val_losses.append((int(m.group(1)), float(m.group(2))))
    
    return epochs, train_losses, val_losses

def print_summary(epochs, train_losses, val_losses):
    """打印训练摘要"""
    print("=" * 60)
    print("  PointGroup Training Summary")
    print("=" * 60)
    
    if not epochs:
        print("  No training data found.")
        return
    
    n = len(epochs)
    print(f"  Total epochs parsed: {n}")
    print(f"  First epoch: {epochs[0]}, Last epoch: {epochs[-1]}")
    
    # 最近 10 个 epoch
    print(f"\n  Last 10 epochs:")
    print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")
    
    val_dict = {e: v for e, v in val_losses}
    recent = epochs[-10:]
    for e in recent:
        tl = train_losses[epochs.index(e)]
        vl = val_dict.get(e, float('nan'))
        print(f"  {e:>6} | {tl:>10.4f} | {vl:>10.4f}")
    
    # 损失变化趋势
    first_loss = train_losses[0]
    last_loss = train_losses[-1]
    change = (last_loss - first_loss) / first_loss * 100
    print(f"\n  Train loss: {first_loss:.4f} → {last_loss:.4f} ({change:+.1f}%)")
    
    if val_losses:
        first_vl = val_losses[0][1]
        last_vl = val_losses[-1][1]
        v_change = (last_vl - first_vl) / first_vl * 100
        print(f"  Val loss:   {first_vl:.4f} → {last_vl:.4f} ({v_change:+.1f}%)")

def plot_curves(epochs, train_losses, val_losses, output_path):
    """生成 loss 曲线图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed. Run: pip install matplotlib")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Val loss
    if val_losses:
        val_epochs = [v[0] for v in val_losses]
        val_vals = [v[1] for v in val_losses]
        ax2.plot(val_epochs, val_vals, 'r-o', markersize=4, label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"\n  Chart saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointGroup Training Visualizer')
    parser.add_argument('--log', type=str, default='', help='Path to training log')
    parser.add_argument('--plot', action='store_true', help='Generate PNG chart')
    args = parser.parse_args()
    
    # 查找最新日志
    if args.log:
        log_path = args.log
    else:
        exp_dir = os.path.join(os.path.dirname(__file__), '..', 'PointGroup_Ours', 
                               'exp', 'cabbage_dataset', 'pointgroup', 'pointgroup_cabbage')
        logs = sorted(glob.glob(os.path.join(exp_dir, 'train-*.log')))
        if not logs:
            print("No training logs found!")
            sys.exit(1)
        log_path = logs[-1]
    
    print(f"Reading: {log_path}\n")
    epochs, train_losses, val_losses = parse_log(log_path)
    print_summary(epochs, train_losses, val_losses)
    
    if args.plot:
        output = os.path.splitext(log_path)[0] + '.png'
        plot_curves(epochs, train_losses, val_losses, output)
