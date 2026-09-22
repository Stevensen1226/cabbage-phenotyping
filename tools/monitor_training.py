#!/usr/bin/env python3
"""终端实时训练监控 - 无需 TensorBoard，直接在终端看训练曲线"""
import os, sys, time, glob, re
from collections import defaultdict

LOG_DIR = "/home/stevensen/Cabbage/PointGroup_Ours/exp/cabbage_dataset/pointgroup/pointgroup_cabbage"

def find_latest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "train-*.log")))
    return logs[-1] if logs else None

def parse_log(filepath):
    """解析日志，提取 loss"""
    epochs, train_loss, val_loss = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            m_train = re.search(r'epoch:\s*(\d+)/(\d+),\s*train loss:\s*([\d.]+)', line)
            m_val = re.search(r'epoch:\s*(\d+)/(\d+),\s*val loss:\s*([\d.]+)', line)
            if m_train:
                epochs.append(int(m_train.group(1)))
                train_loss.append(float(m_train.group(3)))
            elif m_val:
                val_loss.append((int(m_val.group(1)), float(m_val.group(3))))
    return epochs, train_loss, val_loss

def get_thermal(loss, min_v=0.5, max_v=2.0):
    """根据 loss 值返回热力条"""
    level = max(0, min(10, int((max_v - loss) / (max_v - min_v) * 10)))
    bar = "█" * level + "░" * (10 - level)
    return bar

def main():
    log_file = find_latest_log()
    if not log_file:
        print("未找到训练日志")
        return

    print(f"\033[2J\033[H", end="")  # 清屏
    print("=" * 60)
    print("  🥬 PointGroup 甘蓝训练 - 终端实时监控")
    print("=" * 60)

    last_size = 0
    last_train_count = 0

    while True:
        time.sleep(2)

        # 检查日志是否还在更新
        try:
            current_size = os.path.getsize(log_file)
        except:
            print("\n⏳ 等待日志文件...")
            continue

        epochs, train_l, val_l = parse_log(log_file)

        if len(epochs) == 0:
            print("\r⏳ 训练刚开始，等待第一个 epoch...", end="")
            continue

        # 移动到开头重新绘制
        print(f"\033[3H", end="")  # 跳到第3行
        latest_epoch = epochs[-1]
        latest_train = train_l[-1]
        latest_val = val_l[-1] if val_l else (0, 0)

        # 状态条
        progress = latest_epoch / 384 * 100
        progress_bar = "█" * int(progress / 2) + "░" * (50 - int(progress / 2))

        print(f"  进度: [{progress_bar}] {progress:.0f}%  (Epoch {latest_epoch}/384)")
        print(f"  ├─ Train Loss: {latest_train:.4f}  {get_thermal(latest_train, 0.3, 2.0)}")
        print(f"  ├─ Val Loss:   {latest_val[1]:.4f}  (epoch {latest_val[0]})" if val_l else "  ├─ Val Loss:   待评估...")
        print(f"  └─ 日志文件:  {os.path.basename(log_file)}")
        print(f"  └─ 最后更新:  {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(log_file)))}")
        print()
        
        # 最近 5 个 epoch 的 loss 趋势
        recent = list(zip(epochs[-5:], train_l[-5:]))
        print("  最近 5 个 epoch:")
        for ep, loss in recent:
            bar = "█" * int((2.0 - loss) * 10) if loss < 2.0 else "█"
            print(f"    Epoch {ep:>4d}: {loss:.4f}  {get_thermal(loss, 0.3, 2.0)}")
        
        print("\n  Ctrl+C 退出监控（不影响训练）")

        # 向上移动光标
        lines_to_clear = 11 + len(recent)
        print(f"\033[{lines_to_clear}A", end="")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n监控已停止，训练仍在后台运行。")
        print(f"查看日志: tail -f {find_latest_log()}")
