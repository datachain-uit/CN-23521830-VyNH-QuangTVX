# CQ V4 2 CD3 feature view

Built from frozen V4 train/dev/test/FULL_A. See run_manifest.json for provenance and limitations.
Raw: 85 CD3 features, 3 audit masks and phase_fraction, plus frozen metadata/targets.
Main learner: 76 columns; selector: 74.
QA23 held out: none; raw retains all columns.
V1 and gender maps are fitted only on Pool B train P1–P4; dev, test and FULL_A are transform-only.
Feature order is defined by *_feature_columns.json. ID, targets and masks never enter X.

raw_features/pool_b/train.parquet contains pooled early train snapshots.
raw_features/pool_b/dev_phase1.parquet through dev_phase4.parquet contain dev.
raw_features/fulltrain_a/train.parquet contains lambda=1 train snapshots.
model_ready/ mirrors these paths and contains the shared V1 learner columns.
load_cq_v4_2.py loads either view with frozen metadata and q/y aliases; test loading requires a completed test extension and an explicit phase.
The raw matrix supports independent preprocessing checks and the D13 include-held-out sensitivity run.
LR gender one-hot, D13/model training and human D06 approval belong to subsequent work.

Reproduce from data_pipeline: python 25_build_phase_features_v4_2.py
Use requirements-v4_2.txt. Existing stage checkpoints are checked against source/code signatures and file hashes.
All datasets and row-level audits remain private local artifacts.

## Test extension

Test P1–P4 was appended from the original V4 test spine by explicit user request.
raw_features/pool_b/test_phase1.parquet through test_phase4.parquet contain raw test snapshots.
model_ready/pool_b/test_phase1.parquet through test_phase4.parquet contain the frozen train transform.
Use load(split='test', phase='P1', view='learner') or view='selector'; raw=True returns raw features.
No test data were used for fitting, QA23, feature selection or model training.
Original train/dev/FULL_A files, preprocessing, feature order and QA23 reports retain their original hashes.
audit/test_extension_validation.json records these checks; provenance/pre_test_extension retains the original dev-only manifests.
Reproduce the append step: python 27_build_test_features_v4_2.py; verify with python 26_verify_phase_features_v4_2.py.

## Data card

[Data card V4.2](DATA_CARD_V4_2.md) describes the current CD3 feature view, frozen labels, Y37, context temporal lineage, missingness and test scope.
