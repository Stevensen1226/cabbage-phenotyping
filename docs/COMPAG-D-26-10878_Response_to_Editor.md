# Response to the Editor

**Manuscript title:** Model-Agnostic Adaptive Density Peak–Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation  
**Previous manuscript number:** COMPAG-D-26-10878  
**Date:** 12 November 2026

Dear Editor,

We thank you for the editorial assessment. We have not submitted a response-only rebuttal. Instead, we have prepared a new manuscript and revised the experiments, figures, tables, and supporting data so that both concerns can be checked directly. The previous manuscript number, **COMPAG-D-26-10878**, is stated in the cover letter.

## Point-by-point response

### Comment 1 — Validation inconsistency

> The manuscript describes different test cohorts across its methods and results. A consistent map from test scenes to plants is needed to verify whether the reported segmentation gains come from the tested scenes and plants.

**Response**

We have rebuilt the evaluation around one frozen cohort and one evaluation protocol.

- The complete dataset contains 23 scenes and 365 plants.
- The scene-level split is fixed as 13 training scenes / 203 plants, four validation scenes / 60 plants, and six test scenes / 102 plants.
- The six test scenes are cloudR2, cloudR5, cloudR9, cloudR12, cloudR14, and cloudR18.
- All 26 reported model/refinement configurations use the same six test scenes, the same 102-plant registry, the same annotations, and the same unified KNN-R instance-evaluation path.
- Hyperparameters, checkpoints, and post-processing parameters were selected using training and validation data only. The test set was not used for model selection.

We now provide a complete scene–plant–method–metric mapping containing:

- 26 configurations × 102 plants = **2,652 rows**;
- scene ID, plant ID, and GT instance ID;
- model, clustering method, and Base/ADPV configuration;
- matched status, predicted instance ID, and per-plant IoU;
- corresponding scene-level F1, instance mIoU, and count MAE.

We also report scene-level and plant-level gains explicitly. Under Watershed-3D, mean F1 across the four backbones increased from 0.4924 to 0.8709. Under MeanShift, mean F1 increased from 0.8091 to 0.8725. The highest ADPV F1 was 0.9082 for RandLA-Net + MeanShift, followed by 0.9079 for PointNet2-RGB + Watershed-3D.

The scene-level analysis also identifies where refinement is not effective. For MeanShift, negative F1 changes occurred on cloudR9 and cloudR14 for PointGroup, cloudR2 and cloudR9 for PointNet2-RGB, and cloudR9 and cloudR14 for RandLA-Net. These cases are reported rather than averaged away. Watershed-3D improved F1 in 6/6 scenes for PointGroup, PointNet2-RGB, and PointNeXt-L, and in 5/6 scenes with one tie for RandLA-Net.

**Supporting files**

- `revision_data/final_v6_ransac/main_unified_20261112/scene_method_metrics.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/scene_plant_method_metrics.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/scene_gain_table.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/plant_gain_table.csv`
- `revision_data/splits/paper_split_v2.json`
- `revision_data/splits/test_plant_registry.csv`

### Comment 2 — Trait comparisons used different plant groups

> Trait comparisons use different plant groups before and after refinement. Comparisons using the same plants are needed to establish whether refinement improves trait estimates.

**Response**

We agree. The previous unpaired comparison has been removed. The revised analysis uses the identical `(scene_id, plant_id, gt_instance_id)` key on both sides.

A plant was included only when:

1. it was matched at IoU ≥ 0.5 under both Base and ADPV;
2. the four reported traits were valid under both methods;
3. the predicted and reference plant point sets contained sufficient points.

The pairing flow is:

| Model | Base matched | ADPV matched | Paired plants |
|---|---:|---:|---:|
| RandLA-Net | 50 | 85 | 50 |
| PointNet2-RGB | 45 | 91 | 45 |
| PointNeXt-L | 36 | 73 | 36 |
| PointGroup | 36 | 87 | 36 |

For every trait we report the paired mean difference (ADPV minus Base), bootstrap 95% confidence interval, paired t-test, Wilcoxon test, Holm-adjusted p value, and Cohen's dz. The same GT trait values are used for both methods.

The revised conclusions are limited to the paired subsets:

- Crown diameter improved significantly for all four backbones.
- Plant-height changes were not significant for any backbone.
- Alpha-shape volume improved directionally for all four backbones and remained significant after Holm correction for RandLA-Net, PointNeXt-L, and PointGroup; PointNet2-RGB was not significant.
- Compactness improved significantly for RandLA-Net and PointNeXt-L only.
- We no longer claim universal trait improvement or significant height deterioration.

**Supporting files**

- `revision_data/final_v6_ransac/paired_rerun_20261111/v6_paired_trait_statistics.csv`
- `revision_data/final_v6_ransac/paired_rerun_20261111/v6_paired_trait_deltas.csv`
- `revision_data/final_v6_ransac/paired_rerun_20261111/pairing_flow.csv`
- `revision_data/final_v6_ransac/figures/fig7_paired_trait_delta_rerun_20261111.png`
- `revision_data/final_v6_ransac/figures/figS_paired_trait_identity_rerun_20261111.png`

## Additional revisions requested in the editorial suggestions

### Fixed split and leakage control

The Methods now state the 13/4/6 scene-level split, provide the complete scene list, and confirm that no scene occurs in more than one split. All plants within a scene are assigned to the same split. Test scenes are used only for final evaluation.

### Overall, per-scene, and per-plant reporting

The Results now report:

- overall fixed-cohort metrics;
- per-scene ΔF1, Δinstance mIoU, and ΔMAE;
- per-configuration positive, tie, and negative scene counts;
- newly matched plants, lost matches, and same-plant IoU changes;
- explicit failure cases for MeanShift.

### Controlled ablation and parameter selection

The controlled Base, split-only, and split-plus-merge ablation was rerun on the fixed split. Parameter sensitivity was rerun on 21 adhered clusters from 17 training and validation scenes, with all six test scenes excluded. The deployed value `min_peak_dist_m = 0.30 m` was retained based on full-density count error and end-to-end validation F1. The `valley_depth_rel` values 0.85–0.95 formed an end-to-end plateau; 0.90 was retained.

### Efficiency

Efficiency was remeasured on the same six test scenes with the complete ADPV path, including KNN-R. The ADPV increment was 117.6–176.7 ms per scene, corresponding to 5.3–14.8% of full-pipeline latency.

### HDBSCAN in the main comparison

We also included HDBSCAN in the main comparison on the same six test scenes and 102-plant registry with the final mixed checkpoints. Its F1 was 0.0278–0.0581 without ADPV and 0.7967–0.8507 with ADPV. The complete main comparison now contains 26 configurations and a 2,652-row scene–plant–method–metric mapping.

### Figure and table revisions

- Figure 6 now uses fixed test scenes and identifies scene IDs, cluster IDs, and plant IDs.
- Figure 7 reports same-plant paired trait differences with 95% confidence intervals and Holm correction.
- Figure 8 reports sensitivity using training and validation scenes only.
- Figure 9 is retained as exploratory cross-crop generalization evidence, illustrating qualitative transferability to maize and rapeseed. These samples are outside the fixed 23-scene cabbage cohort and were not used for tuning or quantitative model selection, so the figure supports only a preliminary observation rather than a broad generalization claim.
- Figure 10 reports fixed-cohort efficiency with the complete ADPV path.
- Tables 2–6 and the supplementary mapping/gain tables have been regenerated from the unified evaluation.

### Cover letter and new-submission statement

The cover letter states the previous manuscript number and summarizes the changes made in response to both editorial concerns. The submission is a new manuscript, not a response-only rebuttal.

## Summary of the main changes

1. One fixed six-scene, 102-plant test cohort is used for all 26 configurations.
2. A complete 2,652-row scene–plant–method–metric mapping is provided.
3. Scene-level and plant-level gains, including negative cases, are reported explicitly.
4. Trait comparisons use identical plants before and after refinement.
5. The manuscript now reports paired differences, confidence intervals, statistical tests, sample sizes, and inclusion/exclusion criteria.
6. Conclusions have been narrowed to the evaluated cohort, backbones, scenes, and traits.

We hope that the revised manuscript and supporting materials address the editorial concerns directly and provide the requested level of traceability.

Sincerely,

The Authors
