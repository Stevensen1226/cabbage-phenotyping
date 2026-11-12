# Crops3D generalization experiment

## Important scope

Crops3D_IS is the instance-segmentation subset of Crops3D and, according to the official repository, contains plot-scale annotations for maize, potato, and rapeseed. It should be used as an external cross-crop/foreground generalization set, not as a cabbage-specific instance segmentation benchmark.

The full Crops3D dataset contains cabbage among eight crop types, but those files are primarily for organism/organ segmentation and do not replace a cabbage-specific instance test set.

## Data format

The PLY files contain:

```text
x y z r g b sf
```

For Crops3D_IS:

```text
sf = 0   background/ground
sf >= 1  unique plant instance ID
```

For organ segmentation:

```text
sf = 0  soil
sf = 1  stem
sf = 2  leaf
sf = 3  fruit
```

## Conversion

Run:

```bash
python tools/convert_crops3d_to_cabbage_gt.py \
  --input /path/to/Crops3D/Crops3D_IS \
  --output data/external/crops3d_is_converted \
  --mode instance \
  --splits /path/to/Crops3D/Crops3D_IS
```

The converter produces the project's canonical files:

```text
<scene>.ply
<scene>_gt.txt
```

with columns:

```text
x y z r g b semantic instance
```

Background uses `semantic=0, instance=-1`; plant instances use `semantic=1, instance=1..N`.

## Recommended experiments

1. Zero-shot cross-crop foreground generalization:
   train on the cabbage paper split, evaluate directly on Crops3D_IS test plots.
2. Few-shot domain adaptation:
   fine-tune on a small subset of Crops3D_IS train plots and keep all Crops3D_IS test plots untouched.
3. Cabbage organ/full-Crops3D auxiliary experiment:
   use the Cabbage crop directory only after confirming whether per-plant instance IDs are available. Do not fabricate instance labels from organ labels.

## Leakage rule

Never include Crops3D_IS test plots in training or checkpoint selection. Select any adapted checkpoint on a held-out subset of the external training plots, then report the external test split once.
