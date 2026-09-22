#!/usr/bin/env python
"""PointNet2 纯语义训练 — 甘蓝数据集"""
import os, sys, torch, logging, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from dataset.cabbage_dataset import CabbageDataset, cabbage_collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("PointNet2Sem")

def main():
    from util.config import cfg
    # Load YAML
    with open(sys.argv[sys.argv.index('--config') + 1]) as f:
        d = yaml.safe_load(f)
    for s in d:
        if isinstance(d[s], dict):
            for k, v in d[s].items(): setattr(cfg, k, v)
        else:
            setattr(cfg, s, d[s])

    import random, numpy as np
    random.seed(cfg.manual_seed); np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

    ds_train = CabbageDataset(cfg, 'train'); ds_val = CabbageDataset(cfg, 'val')
    dl_train = DataLoader(ds_train, cfg.batch_size, True, num_workers=cfg.train_workers, collate_fn=cabbage_collate_fn, drop_last=True)
    dl_val = DataLoader(ds_val, 1, False, num_workers=cfg.test_workers, collate_fn=cabbage_collate_fn)

    from model.pointnet2_sem import PointNet2Sem, model_fn_decorator
    model = PointNet2Sem(cfg).cuda()
    mfn = model_fn_decorator(test=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    exp = os.path.join('exp', cfg.dataset, cfg.model_name, os.path.basename(cfg.config)[:-5])
    os.makedirs(exp, exist_ok=True)
    logger.info(f"PointNet2-semantic train: {len(ds_train)} samples, exp={exp}")

    best = float('inf')
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for batch in dl_train:
            loss, _, lo, _ = mfn(batch, model, ep)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in dl_val:
                loss, _, _, _ = mfn(batch, model, ep)
                vl += loss.item()
        vl /= max(len(dl_val), 1)
        logger.info(f"Epoch {ep}/{cfg.epochs} val_loss={vl:.4f}")

        if ep % cfg.save_freq == 0:
            torch.save({'epoch': ep, 'model': model.state_dict()}, os.path.join(exp, f'ckpt_{ep}.pth'))
        if vl < best:
            best = vl; torch.save({'epoch': ep, 'model': model.state_dict()}, os.path.join(exp, 'best.pth'))
    logger.info(f"Done. Best val_loss={best:.4f}")

if __name__ == '__main__':
    main()
