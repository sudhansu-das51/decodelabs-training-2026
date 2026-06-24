# Iris Classification — Project Documentation

**Project:** Iris Data Classification  
**Type:** Supervised Machine Learning Pipeline (CLI)  
**Algorithm:** K-Nearest Neighbors (KNN)  
**Batch:** DecodeLabs Industrial Training | 2026  
**Language:** Python 3.8+  
**Developer:** Subin Das

---

## 1. Project Overview

This project builds a complete ML pipeline to classify iris flowers into 3 species — setosa, versicolor, and virginica — using only their physical measurements (sepal and petal dimensions). The model achieves **96.67% accuracy** on unseen test data.

The pipeline is built without any shortcuts: it scales features, tunes K automatically using the Elbow Method, evaluates with multiple metrics, and includes a live prediction demo.

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.load_iris` (built-in, no download needed) |
| Total Samples | 150 |
| Features | 4 (sepal length, sepal width, petal length, petal width) |
| Classes | 3 (setosa, versicolor, virginica) |
| Class Balance | Perfectly balanced — 50 samples per class |

---

## 3. Architecture (IPO Model)

```
[ INPUT ]                  [ PROCESS ]                     [ OUTPUT ]
Load Dataset  →  Scale Features  →  Split Data  →  Tune K  →  Train  →  Evaluate
Visualize                                                                Predict
```

---

## 4. File Breakdown

| Function | Phase | Purpose |
|----------|-------|---------|
| `load_and_explore_data()` | Phase 1 | Loads dataset, prints stats, returns X, y |
| `visualize_features()` | Phase 1B | Saves scatter plots of feature pairs |
| `scale_features()` | Phase 2A | Applies StandardScaler (mean=0, std=1) |
| `split_data()` | Phase 2B | 80/20 stratified train-test split |
| `find_optimal_k()` | Phase 2D | Elbow Method — tests K=3 to K=20, picks best |
| `train_model()` | Phase 2C | Trains KNN with the optimal K |
| `evaluate_model()` | Phase 3 | Accuracy, F1, confusion matrix, classification report |
| `predict_custom_sample()` | Bonus | Live prediction on a new unseen sample |
| `run_pipeline()` | Entry | Orchestrates all phases in order |

---

## 5. Key Technical Decisions

### Why Feature Scaling?
KNN is distance-based. Without scaling, a feature with a large range (e.g., sepal length: 4.3–7.9) would dominate the distance calculation over a feature with a small range (e.g., petal width: 0.1–2.5), even if petal width is more informative. StandardScaler fixes this by making all features comparable.

### Why K starts at 3, not 1?
K=1 means the model looks at only its single nearest neighbor — it essentially memorizes training data and is highly sensitive to noise (overfitting). The training material explicitly flags this risk, so the search begins at K=3.

### Why Stratified Split?
With only 50 samples per class, a random split could accidentally put most of one class in training and very few in testing. `stratify=y` ensures each class maintains its proportional representation in both splits.

### Why F1 Score alongside Accuracy?
Accuracy alone can be misleading. F1 Score (harmonic mean of Precision and Recall) catches cases where a model performs well on some classes but poorly on others — especially important with multi-class problems.

---

## 6. Results & Interpretation

**Optimal K found: 7**  
**Test Accuracy: 96.67%**  
**F1 Score (macro): 0.9666**

**Confusion Matrix:**
```
              Predicted
              setosa  versicolor  virginica
Actual  setosa    10           0          0
    versicolor     0          10          0
     virginica     0           1          9
```

Only 1 sample was misclassified: a virginica flower was predicted as versicolor. This is expected — these two species have overlapping petal measurements (visible in the scatter plots), making them the hardest pair to separate.

**Setosa is perfectly classified** because it is linearly separable from the other two species in petal space.

---

## 7. Visualization Outputs

**`feature_visualization.png`** — Two scatter plots side by side. The right plot (petal measurements) shows much cleaner separation between classes than the left (sepal measurements). This tells us petals are more informative features for this classification task.

**`k_tuning_plot.png`** — Error rate vs K value from K=3 to K=20. The red dotted line marks K=7, the first K where error rate drops to its minimum (0.0333). Several other K values also achieve this minimum, but K=7 is selected as it is the earliest (simplest model that still achieves peak performance).

**`confusion_matrix.png`** — Heatmap with actual vs predicted labels. The diagonal shows correct predictions. The single off-diagonal value (1 in virginica-versicolor cell) is the one misclassification.

---

## 8. Testing Summary

Pipeline was run end-to-end. All stages completed without errors.

| Stage | Result |
|-------|--------|
| Syntax & imports | ✅ Clean |
| Data load (150×4) | ✅ Correct |
| Scaling (mean≈0, std≈1) | ✅ Verified |
| Split (120 train / 30 test) | ✅ Correct |
| K tuning → K=7 | ✅ Optimal found |
| Model training | ✅ Complete |
| Accuracy = 96.67% | ✅ Verified |
| All 3 PNGs saved | ✅ Generated |
| Custom prediction → virginica | ✅ Correct |

**Zero bugs. No fixes required. Code is submission-ready.**

---

## 9. How to Run

```bash
# Install dependencies (one time)
pip install numpy pandas matplotlib seaborn scikit-learn

# Run the pipeline
python iris_classification.py
```

---

## 10. Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Array operations |
| pandas | ≥1.3 | DataFrame for data inspection |
| matplotlib | ≥3.4 | Plotting charts |
| seaborn | ≥0.11 | Enhanced visualizations |
| scikit-learn | ≥0.24 | Dataset, scaler, KNN, metrics |

---

*Documentation prepared for DecodeLabs Project Submission | Batch 2026*
