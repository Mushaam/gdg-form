# gdg-form
# WaferGuard — Wafer Anomaly Detection

## Overview

WaferGuard is a PyTorch-based anomaly detection pipeline for semiconductor manufacturing telemetry. This project audits and repairs a wafer defect detection notebook, focusing on data leakage, correct tensor operations, and legacy Probe Station PS-L preprocessing.

## Key Repairs

- **Data leakage:** Ensured preprocessing is fitted on training data only and evaluation data remains isolated.
- **Tensor mechanics:** Corrected prediction and target shape handling to prevent unintended broadcasting during loss calculation and ensure valid gradient computation.
- **Legacy Probe Station PS-L:** Handled `sensor_0 = 0.0` as a sentinel and treated `sensor_1` values `100.0`, `200.0`, and `300.0` as categorical firmware error codes for legacy wafers.
- **Evaluation integrity:** Evaluated the model on an imbalanced test set to reflect natural defect prevalence rather than an artificially balanced evaluation population.

## Repository Contents

| File | Description |
|---|---|
| `wafer_guard_fixed.ipynb` | Corrected PyTorch notebook containing preprocessing, model training, and evaluation |
| `AUDIT.md` | Summary of the identified defects and implemented repairs |
| `results.json` | Recorded model evaluation results |

## Dataset

The pipeline uses `secom_telemetry_modified.csv`, provided with the GDG AI/ML task.

Place the dataset in the location expected by the notebook before running it. The dataset is not included in this repository unless explicitly added.

## How to Run

1. Open `wafer_guard_fixed.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `secom_telemetry_modified.csv` or configure the notebook to read it from the appropriate location.
3. Run the notebook cells in order.
4. Review the evaluation metrics and compare them with `results.json`.

## Evaluation

The final evaluation should use the natural defect prevalence of approximately 6.6%. Refer to `results.json` and `AUDIT.md` for the **actual measured metrics** and details of the repairs.

## Objective

The goal is to produce a causality-aware, mathematically correct, and reproducible wafer defect detection pipeline suitable for technical review.

