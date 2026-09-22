#!/usr/bin/env python
"""
SoftGroup 训练脚本 — 甘蓝数据集.
用法: python train.py --config config/softgroup_cabbage.yaml
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import random
import logging
from torch.utils.data import DataLoader

from util.config import cfg
from dataset.cabbage_dataset import CabbageDataset, cabbage_collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SoftGroup")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(cfg.manual_seed)

    # Dataset
    train_dataset = CabbageDataset(cfg, split='train')
    val_dataset = CabbageDataset(cfg, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.train_workers, collate_fn=cabbage_collate_fn, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=cfg.test_workers, collate_fn=cabbage_collate_fn)

    # Model
    from model.softgroup.softgroup import SoftGroup, model_fn_decorator
    model = SoftGroup(cfg).cuda()
    model_fn = model_fn_decorator(test=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # Exp path
    exp_path = os.path.join('exp', cfg.dataset, cfg.model_name, 
                            os.path.basename(cfg.config)[:-5])
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, f'train-{cfg.model_name}.log')

    logger.info(f"Training {cfg.model_name} on {cfg.dataset}")
    logger.info(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}")
    logger.info(f"Exp path: {exp_path}")

    best_val_loss = float('inf')

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = {}

        for batch in train_loader:
            loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in meter_dict.items():
                train_losses[k] = train_losses.get(k, 0.0) + v[0]

        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, _, loss_out, _ = model_fn(batch, model, epoch)
                val_loss_total += loss.item()

        val_loss_avg = val_loss_total / max(len(val_loader), 1)

        # Log
        log_str = f"Epoch {epoch:3d}/{cfg.epochs} | "
        for k, v in train_losses.items():
            log_str += f"{k}: {v/len(train_loader):.4f} "
        log_str += f"| val_loss: {val_loss_avg:.4f}"
        logger.info(log_str)

        with open(log_file, 'a') as f:
            f.write(log_str + '\n')

        # Save
        if epoch % cfg.save_freq == 0 or epoch == cfg.epochs:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, os.path.join(exp_path, f'checkpoint_epoch{epoch}.pth'))

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
            }, os.path.join(exp_path, 'best_model.pth'))

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    args = parser.parse_args()

    # Load config (same mechanism as PointGroup)
    cfg.config = args.config
    import yaml
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    for section in cfg_dict:
        if isinstance(cfg_dict[section], dict):
            for k, v in cfg_dict[section].items():
                setattr(cfg, k, v)
        else:
            setattr(cfg, section, cfg_dict[section])

    train()
