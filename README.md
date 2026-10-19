# WAFER-GUARD — Wafer Anomaly Detector

## Overview

WAFER-GUARD is a PyTorch-based semiconductor wafer defect detection pipeline developed for the GDG Tech Wildcard AI/Machine Learning task.

The project audits and repairs a supervised autoencoder pipeline for manufacturing telemetry, focusing on data leakage, incorrect loss gradients caused by tensor broadcasting, and mishandling of legacy Probe Station PS-L data.

**Release verdict: NO-GO.** The repaired model does not meet the required F1-score of 0.90. The final evaluation is reported honestly at the natural prevalence of the chronological test set.

## Key Findings and Repairs

### 1. Data Leakage

**Original defect:** Preprocessing was fitted on the full dataset, SMOTE was applied before splitting, and a random split allowed information from future observations to influence evaluation.

**Repair:**
- Applied a chronological split: 60% training, 20% validation, and 20% testing.
- Fitted imputation, scaling, and feature selection using training data only.
- Removed SMOTE and used a class-weighted loss.
- Used validation data for model and threshold selection.

### 2. Corrupted Loss Gradients

**Original defect:** Incompatible tensor shapes caused unintended broadcasting during reconstruction loss calculation. Each sample could be compared against other samples in the batch.

**Repair:**
- Standardized model tensors to `[batch, features]`.
- Added shape assertions and loss-function tests.
- Verified gradient propagation through the encoder, decoder, and classification head.

### 3. Legacy Probe Station PS-L Handling

**Original defect:** The pipeline treated the legacy sentinel and categorical firmware codes as continuous sensor measurements.

**Repair:**
- Identified legacy wafers using `sensor_0 == 0.0`.
- Encoded `sensor_1` firmware codes `100.0`, `200.0`, and `300.0` as categorical indicators.
- Excluded sentinel values from continuous scaling statistics and the reconstruction loss.
- Documented the use of the training mean as a neutral numerical fill value, with a separate legacy indicator.

## Evaluation Results

The repaired model was evaluated on the final chronological test set without artificial class balancing.

| Metric | Measured result |
|---|---:|
| Test wafers | 314 |
| Defective wafers | 17 |
| Test defect prevalence | 5.41% |
| Precision | 0.000 |
| Recall | 0.000 |
| F1-score | 0.000 |
| ROC-AUC | 0.572 |
| Average precision | 0.118 |

**Confusion matrix**

| | Predicted non-defective | Predicted defective |
|---|---:|---:|
| Actual non-defective | 292 | 5 |
| Actual defective | 17 | 0 |

The full dataset has a defect prevalence of approximately 6.64%. The final chronological test set has a prevalence of 5.41%; it was not artificially rebalanced to force a particular prevalence.

The model failed to detect any of the 17 defective test wafers at the selected threshold. The required F1-score of at least 0.90 was not achieved. The results therefore do not support production deployment.

## Repository Contents

| File | Description |
|---|---|
| `wafer_guard_fixed.ipynb` | Corrected PyTorch notebook with preprocessing, training, and evaluation |
| `AUDIT.md` | Detailed audit findings, repairs, metrics, and limitations |
| `results.json` | Recorded evaluation results |

## Dataset

The notebook uses `secom_telemetry_modified.csv`, provided through the GDG task's supporting Google Drive folder.

The dataset is not included in this repository. Place the CSV file next to the notebook before running it.

## How to Run

1. Open `wafer_guard_fixed.ipynb` in Google Colab or a compatible Jupyter environment.
2. Install the required dependencies:

   ```bash
   pip install torch scikit-learn pandas numpy joblib
   ```

3. Place `secom_telemetry_modified.csv` in the location expected by the notebook.
4. Run all notebook cells from top to bottom.
5. Review the resulting metrics and compare them with `results.json` and `AUDIT.md`.

The audit report records a clean-kernel execution and the environment used for the reported results.

## Limitations

- Only 17 defective wafers appear in the test set, resulting in substantial uncertainty in the evaluation.
- The model's defect detection performance is weak and requires further investigation.
- Potential causes include limited positive examples, temporal drift, and weak predictive signal.
- The validation-selected threshold and training configuration may be unstable because of the small number of defective examples.
- The documented neutral fill value for legacy sensor readings is a modelling trade-off, not a genuine measurement.

## Conclusion

This project demonstrates an audit-first approach to machine learning infrastructure: identify silent defects, repair preprocessing and tensor mechanics, preserve realistic evaluation conditions, and report actual model performance.

The pipeline's implementation has been repaired, but **the detector is not approved for production release** because its measured performance fails the required release criterion.
