# Fab-7 Wafer Defect Detection Pipeline

This project contains the audited and fixed machine learning pipeline for detecting defective silicon wafers using factory sensor data.

The original notebook claimed an F1 score above 0.90, but that was an artificial result caused by data leakage, a broken PyTorch loss calculation, and improper sensor handling. We fixed all three bugs and tested the model honestly on real-world factory data.

---

## What We Changed

1. **Fixed Data Leakage:**
   * **Before:** Imputation, scaling, and SMOTE were run on the entire dataset before splitting. This leaked test data into training and tested on fake synthetic wafers.
   * **Fix:** We split the data into train and test sets first. All scalers, imputers, and SMOTE are now fitted only on the training set.

2. **Fixed PyTorch Loss Calculation:**
   * **Before:** The DataLoader added an extra dimension using `.unsqueeze(1)`, creating tensors of shape `[B, 1, F]` while the model output was `[B, F]`. This triggered silent broadcasting and corrupted the training loss.
   * **Fix:** We removed `.unsqueeze(1)` so batch dimensions match properly.

3. **Handled Legacy Station PS-L Properly:**
   * **Before:** `sensor_0` used `0.0` as a sentinel flag, and `sensor_1` contained categorical status codes (`100.0`, `200.0`, `300.0`). The old code treated these as regular numbers and scaled them.
   * **Fix:** We separated the categorical codes using one-hot encoding and set the continuous values to NaN before scaling.

4. **Added Dynamic Thresholding and Class Weights:**
   * Because defect rates are only 6.6%, the model's probabilities are naturally low. Using a default 0.5 threshold made the model predict zero defects. We added positive class weighting to the loss and swept decision thresholds dynamically to catch actual defective wafers.

---

## Actual Values We Got

We evaluated the model on the natural factory defect rate (around 6.6%) without using SMOTE on the test data:

* **Optimal Decision Threshold:** 0.0000
* **F1 Score:** 0.1652
* **Precision:** 0.0909 (9.1%)
* **Recall:** 0.9048 (90.5%)
* **ROC-AUC:** 0.6507
* **PR-AUC:** 0.1106

### Confusion Matrix
* **True Negatives:** 103 (good wafers correctly identified)
* **False Positives:** 190 (good wafers flagged as bad by mistake)
* **False Negatives:** 2 (bad wafers that slipped through)
* **True Positives:** 19 (bad wafers successfully caught)
