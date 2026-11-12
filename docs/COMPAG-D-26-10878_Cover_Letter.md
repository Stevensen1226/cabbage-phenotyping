# Cover Letter — New Submission

**Manuscript title:** Model-Agnostic Adaptive Density Peak–Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation

**Previous manuscript number:** COMPAG-D-26-10878

Dear Editor,

We are submitting a new manuscript, not a response-only rebuttal. The previous manuscript number is **COMPAG-D-26-10878**. In response to the editorial assessment, we rebuilt the segmentation, ablation, trait, and efficiency experiments around one frozen test cohort and revised the manuscript, figures, tables, and supplementary material accordingly.

## 1. Consistent test cohort and scene-to-plant mapping

We fixed the dataset split at the scene level:

- 23 scenes / 365 plants in total;
- 13 training scenes / 203 plants;
- 4 validation scenes / 60 plants;
- 6 test scenes / 102 plants.

All 26 reported model/refinement configurations use the same six test scenes, the same 102 reference plants, the same annotation registry, and the same unified KNN-R instance-evaluation path. The test scenes are cloudR2, cloudR5, cloudR9, cloudR12, cloudR14, and cloudR18. Hyperparameters and checkpoints were selected using training and validation data only; the test set was reserved for final evaluation.

To make the segmentation gains auditable, we now provide:

- a complete scene-to-plant-to-method-to-metric mapping (26 configurations × 102 plants = 2,652 rows);
- per-plant audit of newly matched plants, lost matches, and IoU change on plants matched by both methods;
- scene-level metrics for all 26 configurations;
- per-scene ΔF1, Δinstance mIoU, and ΔMAE relative to the paired Base configuration;
- the 102-plant registry and fixed split manifest.

Under Watershed-3D, the four-model mean ADPV F1 increased from 0.4924 to 0.8709. Under MeanShift, it increased from 0.8091 to 0.8725. The highest ADPV F1 was 0.9082 for RandLA-Net with MeanShift, followed by 0.9079 for PointNet2-RGB with Watershed-3D. HDBSCAN was also included in the same main comparison: its F1 was 0.0278–0.0581 without ADPV and 0.7967–0.8507 with ADPV. The final main table therefore contains 26 configurations. Scene-level results were uniformly non-negative for Watershed-3D and PointGroup native, whereas MeanShift showed negative F1 changes in specific scenes (cloudR2, cloudR9, or cloudR14 depending on backbone). These failure cases are reported explicitly.

## 2. Same-plant paired trait comparison

The previous trait comparison used different plant groups before and after refinement. We replaced it with a paired analysis using identical `(scene_id, plant_id, gt_instance_id)` observations on both sides. All four backbones were rerun through the same instance-evaluation path.

The revised analysis:

- uses the same GT trait values for Base and ADPV errors;
- reports the paired sample size, mean difference, bootstrap 95% confidence interval, paired t-test, Wilcoxon test, Holm correction, and Cohen's dz;
- uses 50 paired plants for RandLA-Net, 45 for PointNet2-RGB, 36 for PointNeXt-L, and 36 for PointGroup;
- explicitly reports plants retained and excluded at the matching and trait-validity stages.

The revised results are interpreted within these paired subsets. Crown diameter improved consistently across all four backbones and remained significant after Holm correction. Volume improved directionally in all four and remained significant for RandLA-Net, PointNeXt-L, and PointGroup. No backbone showed a significant height change. Compactness improved significantly only for RandLA-Net and PointNeXt-L. We therefore no longer claim universal trait improvement or significant height deterioration.

## 3. Controlled ablation and parameter selection

We reran the controlled Base, split-only, and split-plus-merge ablation on the fixed validation and test scenes using the final mixed-checkpoint protocol and the corrected KNN-R evaluation path. For PointNeXt-L, test F1 changed from 0.4527 (Base) to 0.7466 (split only) and 0.7946 (split plus merge). For PointGroup, the corresponding values were 0.4045, 0.8531, and 0.9033.

Parameter sensitivity was rerun on 21 adhered clusters from the 13 training and four validation scenes; all six test scenes were excluded. `min_peak_dist_m = 0.30 m` was retained based on full-density count error and end-to-end validation F1. The values `valley_depth_rel = 0.85, 0.90, and 0.95` formed an end-to-end plateau, and the existing value 0.90 was retained.

## 4. Efficiency

Efficiency was remeasured on the same six fixed test scenes with the complete deployed ADPV path, including KNN-R. The ADPV increment was 117.6–176.7 ms per scene, corresponding to 5.3–14.8% of full-pipeline latency. Peak GPU memory remained below 6.1 GB.

## 5. Manuscript changes

We have revised the Methods, Results, Discussion, and Conclusion to:

- describe the fixed scene-level split and leakage controls;
- replace the old unpaired trait results and Figure 7;
- replace the sensitivity and efficiency figures with fixed-cohort versions;
- report per-scene heterogeneity rather than only cohort-level averages;
- align the abstract, main table, paired-trait table, ablation table, sensitivity table, efficiency table, figures, and conclusions with the unified evaluation;
- remove claims that all metrics improve consistently or that the method generalizes broadly beyond the evaluated cohort.

We hope the revised manuscript now allows the segmentation and phenotypic gains to be checked at the scene, plant, configuration, and metric levels.

Sincerely,

The Authors
