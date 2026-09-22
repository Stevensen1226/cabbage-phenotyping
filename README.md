# Model-Agnostic Adaptive Density Peak-Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation

Official code and dataset release for the paper:

> **Model-Agnostic Adaptive Density Peak-Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation**  
> Yonglong Zhang, Liwen Shi, Jia Jia, Xiangying Xu, Muhammad Jawaad Atif, Zhiping Zhang

This repository focuses on **instance segmentation of adhered cabbage plants in dense-field 3D point clouds**. The proposed **ADPV-Framework** is a lightweight, model-agnostic post-processing framework that refines initial semantic/instance segmentation results.

## Highlights

- **ADPV-Framework**: a model-agnostic post-processing module for dense cabbage point clouds.
- **DPVIS**: Density Peak-Valley Instance Segmentation is used to decouple adhered cabbage clusters.
- **LCP**: Local coordinate system projection based on PCA.
- **SBD**: Structural boundary detection from one-dimensional projection density.
- **VISRR**: Valley-guided instance separation with recursive re-segmentation.
- **k-NN distance-weighted voting**: repairs fragmented instances after initial segmentation.
- Consistent improvements across multiple baseline models, with the reported F1-score increasing from `0.108` to `0.780` in the paper.
- Low additional inference overhead (`0.5–8.6%`) and limited FPS reduction (`≤7.8%`).

## Dataset

The released dataset contains:

- **23 spatially contiguous sub-regions**
- **365 cabbage plants**
- **4,581,227 annotated points**
- Training: **13 sub-regions, 203 plants**
- Validation: **4 sub-regions, 60 plants**
- Test: **6 sub-regions, 102 plants**

The article-level data split is strictly isolated by sub-region to prevent information leakage.

### Download

The complete dataset is available from the GitHub Release:

- [Download dataset Release v1.0.0](https://github.com/Stevensen1226/cabbage-phenotyping/releases/tag/v1.0.0)
- [Raw PLY + point-wise ground truth](https://github.com/Stevensen1226/cabbage-phenotyping/releases/download/v1.0.0/cabbage_paper_dataset_raw_20260922.zip)
- [PointGroup-format data](https://github.com/Stevensen1226/cabbage-phenotyping/releases/download/v1.0.0/cabbage_pointgroup_format_20260922.zip)
- [SHA256 checksums](https://github.com/Stevensen1226/cabbage-phenotyping/releases/download/v1.0.0/SHA256SUMS.txt)

## Dataset Format

Each point cloud is released as a `.ply` file and a corresponding point-wise `.txt` annotation:

```text
x y z r g b semantic instance
```

Label conventions:

```text
semantic = 0  -> background, ground, weeds, noise, or non-cabbage points
semantic = 1  -> cabbage points

instance = -1 -> background or non-cabbage points
instance = 1..N -> individual cabbage plants
```

The PointGroup-format data use the tuple:

```python
(xyz, rgb, semantic, instance)
```

with background instance labels set to `-100` for the prepared training files.

## Repository Structure

```text
cabbage_pheno/
├── preprocess/       # SOR, ground removal, point cloud cleaning
├── segmentation/     # semantic segmentation models and inference
├── instance/         # clustering, PCA peak-valley splitting, fragment merging
├── traits/           # morphology and phenotype calculations
├── service/          # shared inference pipeline
└── viz/              # point cloud visualization

configs/              # training, inference, ablation, and paper configurations
tools/                # data preparation, evaluation, figure generation, packaging
docs/                 # method, experiment, annotation, and data documentation
train.py              # training entry point
evaluate.py           # evaluation and comparison entry point
main.py               # single-scene inference entry point
requirements.txt      # Python dependencies
```

## Installation

```bash
git clone https://github.com/Stevensen1226/cabbage-phenotyping.git
cd cabbage-phenotyping
pip install -r requirements.txt
pip install -e .
```

The deep-learning components require PyTorch and a compatible CUDA environment. CUDA extensions for PointGroup/SoftGroup should be compiled according to their corresponding directories before training or inference.

## Quick Start

### Generate all pipeline stages for one point cloud

```bash
python tools/generate_steps.py \
  --input evalaute_test/cloudR1.ply \
  --config configs/randla_watershed3d_ransac025.yaml \
  --steps all \
  --out_dir output/step_preview/cloudR1
```

This saves intermediate PLY files for preprocessing, semantic segmentation, coarse clustering, skeleton splitting, fragment merging, and final instance cleanup.

### Evaluate the paper split

```bash
python evaluate.py \
  --input evalaute_test \
  --config configs/randla_watershed3d_ransac025.yaml \
  --split test
```

### Run the full ADPV refinement

The ADPV-related implementation is located in:

```text
cabbage_pheno/instance/clustering.py
```

The main refinement stages are:

1. PCA-based local coordinate projection (`LCP`)
2. Projection-density peak/valley detection (`SBD`)
3. Cut-plane generation and recursive separation (`VISRR`)
4. Fragmented-instance repair using distance-weighted neighbor voting
5. Final instance filtering and cleanup

## Preprocessing

The manuscript uses:

- Statistical Outlier Removal: `k = 10`, `sigma = 3.5`
- RANSAC ground removal: distance threshold `0.025 m`

The corresponding configuration is:

```text
configs/randla_watershed3d_ransac025.yaml
```

## Data Availability

Source code:

https://github.com/Stevensen1226/cabbage-phenotyping

Dataset:

https://github.com/Stevensen1226/cabbage-phenotyping/releases/tag/v1.0.0

## Citation

If you use this repository or dataset, please cite the paper. The final bibliographic information will be updated after publication.

```bibtex
@article{zhang_cabbage_adpv,
  title   = {Model-Agnostic Adaptive Density Peak-Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation},
  author  = {Zhang, Yonglong and Shi, Liwen and Jia, Jia and Xu, Xiangying and Atif, Muhammad Jawaad and Zhang, Zhiping},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  note    = {Manuscript}
}
```

## Contact

For questions about the method or dataset, please open a GitHub issue or contact the corresponding author:

```text
Zhiping Zhang
zhangzp@yzu.edu.cn
Yangzhou University
```

## License

A formal license will be added before the public paper release. The dataset and code may not be redistributed without permission until the license is finalized.
