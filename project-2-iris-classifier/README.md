# 🌸 Iris Data Classification

A complete supervised machine learning pipeline that classifies iris flowers into 3 species using the K-Nearest Neighbors (KNN) algorithm — with automatic K tuning, feature scaling, and full visual evaluation.

> **Project 2 | DecodeLabs Industrial Training Kit | Batch 2026**

---

## 📌 What It Does

This project takes the classic Iris dataset (150 flower samples, 3 species) and builds a full ML pipeline from scratch — loading data, scaling features, finding the best K value automatically, training a KNN model, and evaluating it with accuracy, F1 score, confusion matrix, and live predictions.

**Final Result: 96.67% accuracy on test data.**

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Pipeline

```bash
python iris_classification.py
```

The script will print results to the terminal and save 3 PNG charts automatically.

---

## 📊 Output Files Generated

| File | What It Shows |
|------|--------------|
| `feature_visualization.png` | Scatter plots showing how the 3 species separate by measurements |
| `k_tuning_plot.png` | Error rate vs K value — shows why K=7 was chosen |
| `confusion_matrix.png` | Heatmap showing correct vs incorrect predictions per class |

---

## 🧠 How the Pipeline Works

The project follows the **IPO Model** (Input → Process → Output):

```
Load Data → Visualize → Scale Features → Split Data → Tune K → Train → Evaluate → Predict
```

### Phase 1 — Input
- Loads the Iris dataset (built into scikit-learn, no download needed)
- 150 samples, 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: setosa, versicolor, virginica — 50 samples each
- Visualizes feature distributions to understand the data before modeling

### Phase 2 — Process
- **Feature Scaling** — StandardScaler transforms all features to Mean=0, Variance=1 (required for KNN since it uses distance)
- **Train-Test Split** — 80% training / 20% testing, stratified to preserve class balance
- **K Tuning** — Tests K=3 to K=20, picks the K with the lowest error rate (Elbow Method)
- **Model Training** — KNN trained with optimal K=7

### Phase 3 — Output
- Accuracy score, F1 score, confusion matrix, full classification report
- Live prediction on a brand new unseen flower sample

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **96.67%** |
| F1 Score (macro) | **0.9666** |
| Optimal K | **7** |
| Misclassified samples | **1 out of 30** (1 virginica predicted as versicolor) |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| setosa | 1.00 | 1.00 | 1.00 |
| versicolor | 0.91 | 1.00 | 0.95 |
| virginica | 1.00 | 0.90 | 0.95 |

---

## 🗂️ Project Structure

```
iris_classification.py      ← Full pipeline (single file)
feature_visualization.png   ← Generated on run
k_tuning_plot.png           ← Generated on run
confusion_matrix.png        ← Generated on run
README.md                   ← This file
```

---

## ⚙️ Configuration

Edit these two constants at the top of the file to change behavior:

```python
RANDOM_STATE = 42     # Change for different train/test splits
TEST_SIZE    = 0.20   # Change split ratio (0.20 = 80/20)
```

K is found **automatically** — no need to set it manually.

---

## 🧪 Test Results

Fully tested by running the complete pipeline end-to-end:

| Test | Status |
|------|--------|
| Syntax check | ✅ Pass |
| All imports available | ✅ Pass |
| Data loads correctly (150 samples, 4 features) | ✅ Pass |
| Feature scaling (Mean≈0, Std≈1) | ✅ Pass |
| Train/test split (120/30) | ✅ Pass |
| K tuning finds K=7 | ✅ Pass |
| Model training completes | ✅ Pass |
| Accuracy = 96.67% | ✅ Pass |
| All 3 PNG charts saved | ✅ Pass |
| Live prediction on custom sample | ✅ Pass |

**No errors or bugs found.**

---

## 🔮 Possible Improvements (Future Scope)

- Try other classifiers: SVM, Decision Tree, Random Forest
- Add cross-validation for more reliable accuracy estimate
- Build a simple web UI with Flask or Streamlit
- Test on a real-world flower measurement dataset

---

## 👤 Author

Sudhansu Mohana Das
DecodeLabs Industrial Training | Batch 2026

---

## 📄 License

This project is part of the DecodeLabs training curriculum. For educational use only.
