# COMPAG-D-26-10878 Revision Package

## Submission-ready components

- `COMPAG-D-26-10878_Cover_Letter.md`
- `COMPAG-D-26-10878_Cover_Letter.docx`
- `COMPAG-D-26-10878_Revised_Manuscript_Text.md`
- `COMPAG-D-26-10878_Revised_Manuscript_Text.docx`
- `COMPAG-D-26-10878_Fixed_Cohort_Results_CN.md`
- `COMPAG-D-26-10878_Response_to_Editor.md`
- `COMPAG-D-26-10878_Response_to_Editor_CN.md`
- `COMPAG-D-26-10878_FINAL_AUDIT_20261111.md`

## Updated manuscript tables

- `revision_data/final_v6_ransac/manuscript_tables_20261111/Table2_main_results.csv|tex`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/Table3_paired_traits.csv|tex`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/Table4_ablation.csv|tex`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/Table5_sensitivity.csv|tex`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/Table6_efficiency.csv|tex`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/TableS_HDBSCAN_results.csv|tex`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/TableS_scene_plant_method_metric_mapping.csv`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/TableS_scene_gain.csv`
- `revision_data/final_v6_ransac/manuscript_tables_20261111/TableS_plant_gain.csv`

## Updated figures

- `revision_data/final_v6_ransac/figures/fig7_paired_trait_delta_rerun_20261111.png|pdf`
- `revision_data/final_v6_ransac/figures/fig8_sensitivity_trainval_rerun_20261111.png|pdf`
- `revision_data/final_v6_ransac/figures/fig10_efficiency_fixed_test_rerun_20261111.png|pdf`
- `revision_data/final_v6_ransac/figures/figS_paired_trait_identity_rerun_20261111.png|pdf`

## Core result files

- `revision_data/final_v6_ransac/main_unified_20261112/scene_method_metrics.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/macro_method_metrics.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/scene_plant_method_metrics.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/scene_gain_table.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/scene_gain_summary.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/plant_gain_table.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/plant_gain_summary.csv`
- `revision_data/final_v6_ransac/paired_rerun_20261111/v6_paired_trait_statistics.csv`
- `revision_data/final_v6_ransac/paired_rerun_20261111/v6_paired_trait_deltas.csv`
- `revision_data/final_v6_ransac/ablation_retrained/ablation_summary.csv`
- `revision_data/final_v6_ransac/ablation_retrained/sensitivity_trainval.csv`
- `revision_data/final_v6_ransac/efficiency_20261111/efficiency_summary.csv`
- `revision_data/final_v6_ransac/hdbscan_20261112/HDBSCAN_RESULTS_CN.md`
- `revision_data/final_v6_ransac/hdbscan_20261112/scene_method_metrics.csv`
- `revision_data/final_v6_ransac/hdbscan_20261112/scene_plant_method_metrics.csv`
- `revision_data/final_v6_ransac/hdbscan_20261112/hdbscan_summary_for_review.csv`

## Evidence status

- Fixed 13/4/6 split: 13 training scenes / 203 plants, 4 validation scenes / 60 plants, and 6 test scenes / 102 plants.
- All 26 main-table configurations use the same 102-plant test registry and unified KNN-R evaluation (18 primary + eight HDBSCAN).
- Complete 26 × 102 = 2,652-row scene–plant–method–metric mapping is provided.
- Scene-level positive/tie/negative gains and per-plant newly matched/lost/IoU changes are reported.
- Paired trait sample sizes are 50 / 45 / 36 / 36.
- All six test scenes are excluded from parameter selection.
- Controlled ablation, sensitivity, and efficiency were rerun on 2026-11-11.
- HDBSCAN was added to the main comparison on 2026-11-12; the main table now contains 26 configurations and the total mapping contains 2,652 rows.
- Figure, table, and numerical consistency audits pass for the current files.

## Source limitation

Only the submitted PDF was available; no editable LaTeX/Word source for the full manuscript was found in the workspace. The package therefore includes replacement manuscript text, tables, figures, and audited result files, but the final publisher-formatted PDF still needs to be assembled from the journal template.
