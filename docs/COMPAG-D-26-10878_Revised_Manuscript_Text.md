# COMPAG-D-26-10878 — Drop-in revised manuscript text

> This file contains replacement text for sections affected by the fixed-cohort revision. The source manuscript PDF was available, but no editable LaTeX/Word source was found in the workspace. Table and figure numbers should be checked once the editable source is available.
>
> **Update 2026-11-11:** Sections 3.3, 3.4, 3.5, and 3.7 have been synchronized with the unified KNN-R evaluation, paired-trait rerun, ablation, sensitivity, and efficiency benchmarks. Table and figure numbers should still be checked against the final editable source.

## Revised Abstract

Severe plant adhesion and canopy occlusion in densely planted open-field cabbage cause 3D point cloud instance segmentation models to merge neighbouring plants, fragment instances, and lose individual-plant coverage. We propose the Adaptive Density Peak-Valley (ADPV) refinement framework, a lightweight post-processing module that decouples adhered instances using density peak-valley analysis and repairs fragmented instances using k-nearest-neighbour distance-weighted voting. The framework is evaluated on a manually annotated cabbage dataset containing 23 scenes and 365 plants, partitioned at the scene level into 13 training scenes (203 plants), four validation scenes (60 plants), and six test scenes (102 plants). All compared configurations use the same test scenes, 102-plant reference registry, annotations, and upstream checkpoints. Across four backbones and two clustering paradigms, ADPV increased instance F1 and plant coverage, with the highest fixed-cohort ADPV F1 reaching 0.9082 for RandLA-Net with MeanShift and 0.9079 for PointNet2-RGB with Watershed-3D. Pairwise comparison on identical plants further showed model- and trait-dependent effects. Crown diameter improved consistently, volume and compactness improved for several backbones, and no backbone showed a significant change in plant height; therefore, ADPV is not a universal trait-correction method. The complete ADPV path added 117.6–176.7 ms per scene, corresponding to 5.3–14.8% of end-to-end latency, while peak GPU memory remained below 6.1 GB. These results support ADPV as a practical refinement step for dense cabbage instance segmentation, while limiting the conclusion to the fixed six-scene evaluation cohort.

## Revised Section 2.1 — Data acquisition and split

The field point cloud was divided into 23 spatially contiguous sub-regions containing 365 manually annotated cabbage plants. The data were partitioned at the sub-region level into 13 training scenes (203 plants), four validation scenes (60 plants), and six test scenes (102 plants). The test scenes were cloudR2, cloudR5, cloudR9, cloudR12, cloudR14, and cloudR18. No scene occurred in more than one split, and all plants within a scene were assigned to the same split. The test set was used only for final evaluation. Model selection, checkpoint selection, and post-processing parameter selection used training and validation data only. The complete scene list and 102-plant test registry are provided in the supplement. A 2,652-row scene–plant–method–metric mapping for all 26 configurations (18 primary configurations plus eight HDBSCAN configurations), together with scene-level and plant-level gain tables, is included as Supplementary Data.

## Revised Section 3.3 — Quantitative comparison

Table 2 summarizes the unified evaluation of 26 configurations on the same six test scenes and the same 102-plant reference registry. All Watershed-3D and MeanShift configurations were rerun through the same preprocessing, semantic-backbone, clustering, and KNN-R evaluation path. Under Watershed-3D, ADPV increased F1 from 0.5914 to 0.8777 for RandLA-Net, from 0.5211 to 0.9079 for PointNet2-RGB, from 0.4527 to 0.7946 for PointNeXt-L, and from 0.4045 to 0.9033 for PointGroup. The four-model mean increased from 0.4924 to 0.8709 (+0.3785).

The MeanShift results were also positive but more heterogeneous. F1 increased from 0.8473 to 0.9082 for RandLA-Net, from 0.8473 to 0.8631 for PointNet2-RGB, from 0.6955 to 0.8586 for PointNeXt-L, and from 0.8464 to 0.8599 for PointGroup. The four-model mean increased from 0.8091 to 0.8725 (+0.0634). The highest ADPV F1 in the complete comparison was 0.9082 for RandLA-Net with MeanShift, followed closely by 0.9079 for PointNet2-RGB with Watershed-3D.

For the native PointGroup comparison, F1 increased from 0.4859 to 0.8648, instance mIoU from 0.7570 to 0.7635, plant coverage from 39.2% to 84.3%, and count MAE decreased from 5.67 to 1.50. These results indicate that ADPV primarily recovers missing and merged plants and reduces fragmentation. Improvements are nevertheless not uniform across every scene or metric, so the claim of consistent improvement across all segmentation metrics remains removed.

HDBSCAN was also included in the main fixed-cohort comparison (`min_cluster_size=500`, `min_samples=10`, EOM). It produced F1 = 0.0278–0.0581 without ADPV. ADPV increased HDBSCAN F1 to 0.7967–0.8507, demonstrating that the refinement can recover severe over-merging, but HDBSCAN+ADPV remained below Watershed or MeanShift in most backbone–dataset combinations. HDBSCAN is therefore reported alongside the other clustering methods in the 26-configuration main table.

Scene-level audit confirmed this heterogeneity. Watershed-3D improved F1 in 6/6 test scenes for PointGroup, PointNet2-RGB, and PointNeXt-L, and in 5/6 scenes with one tie for RandLA-Net; PointGroup native improved in 6/6 scenes. MeanShift was less stable: PointGroup decreased on cloudR9 (-0.213 F1) and cloudR14 (-0.068), PointNet2-RGB decreased on cloudR2 (-0.133) and cloudR9 (-0.115), and RandLA-Net decreased on cloudR14 (-0.068) and cloudR9 (-0.017). The complete scene-to-plant-to-method-to-metric mapping contains 26 configurations × 102 plants = 2,652 rows. Per-plant matching audit showed that Watershed-3D added 35–51 newly matched plants and lost no previously matched plant across the four backbones. MeanShift added 13–20 newly matched plants but lost 0–4, whereas PointGroup native added 46 and lost none. The improvement is therefore predominantly driven by newly recovered plants, with a smaller contribution from increased IoU on plants matched on both sides.

## Revised Section 3.4 — Paired trait extraction

Trait comparisons were restricted to identical plants observed under both Base and ADPV. The pairing key was `(scene_id, plant_id, gt_instance_id)`. A plant entered the paired analysis only when it was matched at IoU ≥ 0.5 on both sides and the four reported traits were valid on both sides. The same GT trait values were used when calculating Base and ADPV absolute errors. All four backbones were rerun through the same clustering evaluation path, including the k-nearest-neighbour distance-weighted fragment-reassignment branch used by the final ADPV configuration.

The paired sample sizes were 50 plants for RandLA-Net, 45 for PointNet2-RGB, 36 for PointNeXt-L, and 36 for PointGroup. For RandLA-Net, crown diameter changed by -5.533 cm [95% CI: -8.119, -3.128], α-shape volume by -355.478 cm³ [-568.999, -171.322], and compactness by -0.00178 [-0.00311, -0.00058]; all three improvements remained significant after Holm correction. The height difference was -0.082 cm [-0.281, +0.098] and was not significant.

For PointNet2-RGB, crown diameter changed by -4.063 cm [-6.178, -2.147] and remained significant after Holm correction. Volume changed by -143.751 cm³ [-289.887, -14.240], and compactness by -0.00047 [-0.00160, +0.00064], but neither remained significant after Holm correction. Plant height changed by -0.087 cm [-0.310, +0.108] and was not significant.

For PointNeXt-L, crown diameter changed by -4.826 cm [-7.509, -2.493], volume by -180.271 cm³ [-343.808, -40.120], and compactness by -0.00115 [-0.00214, -0.00029]; all three remained significant after Holm correction. Height changed by +0.111 cm [-0.172, +0.508] and was not significant.

For PointGroup, crown diameter changed by -4.445 cm [-7.338, -1.881] and volume by -370.771 cm³ [-678.201, -131.044], both remaining significant after Holm correction. Compactness changed by -0.00008 [-0.00126, +0.00115], and height by +0.089 cm [-0.057, +0.249]; neither was significant.

Figure 7 shows the paired absolute-error differences, and the supplementary paired-identity figure shows the Base-versus-ADPV relation for the same plants. These results show that the downstream benefit of ADPV is model- and trait-dependent. Crown diameter improved consistently across all four backbones. Volume improved directionally in all four and remained significant for three. No backbone showed a significant height change. The previous unpaired comparison based on different plant groups has been removed.

## Revised Section 3.5 — Ablation and sensitivity

The controlled ablation used the final mixed-checkpoint protocol and the fixed 13/4/6 split. `Base`, `split only`, and `split plus merge` were evaluated on the four validation scenes and the six test scenes with the same preprocessing and instance-evaluation code. The full state enabled PCA-based splitting, PCA fragment merging, fragment voting, and KNN-R reassignment.

For PointNeXt-L with Watershed-3D, F1 changed from 0.3576 at the validation baseline to 0.6240 with splitting only and 0.7098 with splitting plus merging. On the test set, the corresponding values were 0.4527, 0.7466, and 0.7946. For PointGroup with Watershed-3D, validation F1 changed from 0.4189 to 0.7223 and 0.8602, while test F1 changed from 0.4045 to 0.8531 and 0.9033. The merge stage therefore added +0.0859 and +0.1378 F1 on validation, and +0.0480 and +0.0503 on test, respectively. The gain was concentrated in splitting, while merging provided an additional improvement in instance recovery and count error.

Parameter sensitivity was evaluated on 21 adhered clusters from the 13 training and four validation scenes; all six test scenes were excluded. For `min_peak_dist_m = 0.30 m`, the full-density count MAE was 0.3333, the lowest among the tested values. End-to-end validation across the two backbones gave pooled F1 = 0.7850 and mean MAE = 1.75, compared with F1 = 0.7821 and MAE = 2.125 for 0.35 m; 0.30 m was therefore retained. The values `valley_depth_rel = 0.85, 0.90, and 0.95` produced the same end-to-end validation F1 and count MAE, forming a plateau. The existing value 0.90 was retained because the low-density sensitivity advantage of 0.85 did not translate into an end-to-end improvement. The parameter conclusion is therefore robust within the tested plateau but remains limited to the available training and validation clusters.

## Revised Section 3.6 — Exploratory cross-crop generalization

The maize and rapeseed visualizations are retained as exploratory cross-crop generalization evidence. They illustrate qualitative transferability of the post-processing concept to other crops. Because these samples are outside the fixed 23-scene cabbage cohort and were not used for training, tuning, quantitative model selection, or pixel-level evaluation, this evidence is preliminary and qualitative. It is not used to support a quantitative or broad generalization claim; the main quantitative conclusions remain restricted to the fixed six-scene cabbage test set.

## Revised Section 3.7 — Efficiency on the fixed test cohort

Efficiency was measured on the same six fixed test scenes and the same 102-plant registry used for segmentation evaluation. The benchmark included the complete deployed ADPV path, including PCA splitting, fragment voting, and KNN-R reassignment. Across scenes, ADPV added 176.7 ms per scene for RandLA-Net, 149.5 ms for PointNet2-RGB, 117.6 ms for PointNeXt-L, and 170.6 ms for PointGroup. Relative to the complete Base pipeline, these increments correspond to 10.2%, 10.1%, 5.3%, and 14.8%, respectively.

Full-pipeline latency increased from 1591.9 to 1768.6 ms for RandLA-Net, from 1340.7 to 1490.2 ms for PointNet2-RGB, from 2002.1 to 2119.7 ms for PointNeXt-L, and from 1169.8 to 1340.4 ms for PointGroup. Throughput changed from 0.772 to 0.706 FPS, 0.863 to 0.791 FPS, 0.549 to 0.524 FPS, and 1.215 to 1.064 FPS, respectively. Peak GPU memory was approximately 5.8, 6.1, 0.4, and 0.8 GB. Measurements excluded model loading and first-scene warm-up; the faster of two timed cluster runs was retained for each scene. PointGroup was evaluated in the semantic clustering branch used by the main comparison.

## Revised Discussion

The fixed-cohort analysis supports a narrower and more defensible conclusion than the original submission. ADPV is effective at recovering individual plants from merged clusters and reducing fragmentation, as reflected by the consistent F1 and coverage gains for several Watershed-3D configurations. However, improvements are not uniform across clustering methods, scenes, or phenotypic traits. Instance mIoU decreased in some PointGroup settings, and the trait analysis showed consistent crown-diameter improvement but model-dependent volume and compactness effects. No backbone showed a significant height deterioration. Therefore, ADPV should be described as a refinement strategy that improves instance recovery under dense adhesion, not as a universal geometric or phenotypic correction method.

The evaluation also has clear limits. The test cohort contains six scenes and 102 plants from one acquisition region. The paired trait analysis includes 36–50 plants per backbone because only plants matched by both Base and ADPV can be compared. The main comparison covers Watershed-3D, MeanShift, and the native PointGroup proposal path, but is not an exhaustive benchmark of all 3D instance-segmentation methods. Independent field trials and external-region validation remain necessary.

## Revised Conclusion

We presented ADPV, a model-agnostic post-processing framework for dense cabbage point cloud instance segmentation. On a fixed six-scene, 102-plant test cohort, ADPV improved instance F1 and plant coverage for multiple backbones and clustering methods, with the highest fixed-cohort F1 reaching 0.9082 for RandLA-Net with MeanShift and the Watershed-3D F1 reaching 0.9079 for PointNet2-RGB. The complete ADPV path added 117.6–176.7 ms per scene (5.3–14.8% of end-to-end latency), while peak GPU memory remained below 6.1 GB. At the same time, the results show that improvements are not universal: some instance-mIoU values decreased, and phenotypic effects varied by backbone and trait. Crown diameter improved consistently, whereas height changes were not significant and volume/compactness effects were model-dependent. These findings support ADPV as a practical instance-recovery refinement under dense planting, while restricting its claimed benefits to the evaluated cohort and traits.

## Revised Figure Captions

**Figure 6.** Qualitative instance recovery for adhered cabbage plants in the fixed test split. Rows show cluster K3 and K4 from cloudR18 and cluster K5 from cloudR14. Plant IDs are listed in the GT panels. Columns show GT, RandLA-Net Base, RandLA-Net ADPV, PointGroup Base, and PointGroup ADPV. All scenes and plants shown belong to the six fixed test scenes and are not used for training or model selection.

**Figure 7.** Paired absolute-error differences on identical plants before and after ADPV (ADPV − Base; negative indicates improvement). Points show individual plants, horizontal bars show mean differences with bootstrap 95% confidence intervals, and labels report paired sample sizes and Holm-adjusted Wilcoxon significance. Paired sample sizes are 50, 45, 36, and 36 plants for RandLA-Net, PointNet2-RGB, PointNeXt-L, and PointGroup, respectively.

**Figure 8.** Count-MAE sensitivity for `min_peak_dist_m` and `valley_depth_rel` on 21 adhered clusters from the 13 training and four validation scenes. All six test scenes are excluded. The orange box marks the deployed value.

**Figure 9.** Exploratory cross-crop generalization evidence on maize and rapeseed. These samples are outside the fixed 23-scene cabbage cohort. The figure illustrates qualitative transferability only and was not used for training, tuning, quantitative model selection, or claims of broad quantitative generalization.

**Figure 10.** Full-pipeline efficiency on the same six fixed test scenes and 102-plant registry. Measurements include the complete ADPV path, including PCA splitting, fragment voting, and KNN-R reassignment. Bars show mean per-scene latency and throughput; GPU labels show peak memory.

## Required manuscript consistency checks

- Replace the old trait figure with `fig7_paired_trait_delta` and the supplementary identity figure with `figS_paired_trait_identity`.
- Replace the old sensitivity figure with `fig8_sensitivity_trainval`.
- Replace the old efficiency figure with `fig10_efficiency_fixed_test`.
- Report paired trait sample sizes as 50/45/36/36.
- State that crown diameter improved for all four backbones, height changes were not significant, and volume/compactness effects were model-dependent.
- State that the trait analysis is restricted to the paired subset and report bootstrap 95% CIs with Holm correction.
- Remove the old statement that RandLA-Net height error increased significantly.
- Replace the old 86-to-186 coverage statement and old ablation values.
