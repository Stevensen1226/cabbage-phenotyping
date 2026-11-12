# Model-Agnostic Adaptive Density Peak–Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation

Official code, documentation, and released results for the paper:

> **Model-Agnostic Adaptive Density Peak–Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation**  
> Yonglong Zhang, Liwen Shi, Jia Jia, Xiangying Xu, Muhammad Jawaad Atif, Zhiping Zhang

This repository provides the complete ADPV post-processing framework, baseline training/inference code, preprocessing tools, evaluation scripts, paper documentation, final tables, figures, and result files.

## Highlights

- Model-agnostic ADPV refinement for adhered cabbage point clouds.
- PCA-based local projection and density peak–valley instance separation.
- k-NN distance-weighted fragment reassignment.
- Unified evaluation across Watershed-3D, MeanShift, HDBSCAN, and PointGroup-native configurations.
- Fixed scene-level train/validation/test split with a 102-plant test registry.

## Dataset

- 23 spatially contiguous scenes
- 365 annotated cabbage plants
- 4,581,227 annotated points
- Training: 13 scenes / 203 plants
- Validation: 4 scenes / 60 plants
- Test: 6 scenes / 102 plants

The split is isolated at the scene level. Test scenes are cloudR2, cloudR5, cloudR9, cloudR12, cloudR14, and cloudR18.

Dataset release:

- [Dataset Release v1.0.0](https://github.com/Stevensen1226/cabbage-phenotyping/releases/tag/v1.0.0)
- [Raw PLY + point-wise ground truth](https://github.com/Stevensen1226/cabbage-phenotyping/releases/download/v1.0.0/cabbage_paper_dataset_raw_20260922.zip)
- [PointGroup-format data](https://github.com/Stevensen1226/cabbage-phenotyping/releases/download/v1.0.0/cabbage_pointgroup_format_20260922.zip)
- [Dataset checksums](https://github.com/Stevensen1226/cabbage-phenotyping/releases/download/v1.0.0/SHA256SUMS.txt)

## Current Paper Results

The current fixed-cohort main comparison contains 26 configurations: 18 primary configurations and eight HDBSCAN configurations.

| Method | ADPV F1 range | Mean Base F1 | Mean ADPV F1 |
|---|---:|---:|---:|
| Watershed-3D | 0.7946–0.9079 | 0.4924 | 0.8709 |
| MeanShift | 0.8586–0.9082 | 0.8091 | 0.8725 |
| HDBSCAN | 0.7967–0.8507 | 0.0493 | 0.8231 |
| PointGroup native | 0.8648 | 0.4859 | 0.8648 |

- Highest ADPV F1: **0.9082** (RandLA-Net + MeanShift).
- PointNet2-RGB + Watershed-3D: **0.9079**.
- Complete scene–plant–method–metric mapping: **26 × 102 = 2,652 rows**.
- Paired trait sample sizes: **50 / 45 / 36 / 36**.
- Crown diameter improved significantly across all four backbones; height changes were not significant.
- Full ADPV overhead: **117.6–176.7 ms/scene** (5.3–14.8%).

## Repository Structure

```text
cabbage_pheno/       Core preprocessing, segmentation, clustering, traits, and evaluation code
configs/             Training, inference, ablation, and paper configurations
tools/               Data preparation, evaluation, figure generation, packaging, and revision scripts
docs/                Method, data, experiment, paper, and audit documentation
revision_data/       Final split manifests, tables, figures, and audited result files
train.py             Training entry point
evaluate.py          Evaluation entry point
main.py              Single-scene inference entry point
requirements.txt     Python dependencies
```

## Installation

```bash
git clone https://github.com/Stevensen1226/cabbage-phenotyping.git
cd cabbage-phenotyping
pip install -r requirements.txt
pip install -e .
```

Deep-learning components require PyTorch and a compatible CUDA environment. PointGroup/SoftGroup CUDA extensions should be compiled according to their corresponding source directories.

## Quick Start

### Evaluate the fixed paper split

```bash
python evaluate.py \
  --input evalaute_test \
  --config configs/randla_watershed3d_ransac025.yaml \
  --split test
```

### Generate pipeline stages for one point cloud

```bash
python tools/generate_steps.py \
  --input evalaute_test/cloudR1.ply \
  --config configs/randla_watershed3d_ransac025.yaml \
  --steps all \
  --out_dir output/step_preview/cloudR1
```

### ADPV implementation

The main ADPV implementation is in:

```text
cabbage_pheno/instance/clustering.py
```

## Paper Documents and Final Tables

- Manuscript replacement text: `docs/COMPAG-D-26-10878_Revised_Manuscript_Text.md`
- Fixed-cohort result report: `docs/COMPAG-D-26-10878_Fixed_Cohort_Results_CN.md`
- Final main table: `revision_data/final_v6_ransac/manuscript_tables_20261111/Table2_main_results.csv`
- Paired trait table: `revision_data/final_v6_ransac/manuscript_tables_20261111/Table3_paired_traits.csv`
- Ablation table: `revision_data/final_v6_ransac/manuscript_tables_20261111/Table4_ablation.csv`
- Sensitivity table: `revision_data/final_v6_ransac/manuscript_tables_20261111/Table5_sensitivity.csv`
- Efficiency table: `revision_data/final_v6_ransac/manuscript_tables_20261111/Table6_efficiency.csv`
- Scene–plant–method–metric mapping: `revision_data/final_v6_ransac/manuscript_tables_20261111/TableS_scene_plant_method_metric_mapping.csv`

## Data Availability

Source code:

https://github.com/Stevensen1226/cabbage-phenotyping

Dataset:

https://github.com/Stevensen1226/cabbage-phenotyping/releases/tag/v1.0.0

## Citation

If you use this repository or dataset, please cite the paper. Final bibliographic information will be updated after publication.

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

For questions about the method or dataset, please open a GitHub issue or contact:

```text
Zhiping Zhang
zhangzp@yzu.edu.cn
Yangzhou University
```

## License

A formal license will be added before the public paper release. The dataset and code may not be redistributed without permission until the license is finalized.
