# 🤖 Tech Stack Recommender

A content-based AI recommendation engine that maps a user's skills to the most relevant tech career roles using TF-IDF vectorization and Cosine Similarity.

> **Project 3 | DecodeLabs Industrial Training Kit | Batch 2026**

---

## 📌 What It Does

You type in your skills (e.g. Python, Machine Learning, Statistics) and the engine scores every job role in the dataset against your profile using real ML math — not simple keyword matching. It returns your **Top 3 most suitable career roles** with match percentages.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn
```

### Run the Recommender

```bash
python tech_stack_recommender.py
```

Then enter your skills when prompted:

```
Your skills: Python, Machine Learning, Statistics
```

---

## 💬 Example Output

```
Based on your input skills: Python, Machine Learning, Statistics

#1  Machine Learning Engineer
     Match Score : 53.6%
     Required Skills : Python, Machine Learning, TensorFlow, PyTorch, ...

#2  AI Research Engineer
     Match Score : 52.1%
     Required Skills : Python, Machine Learning, Statistics, Algorithms, ...

#3  Data Scientist
     Match Score : 51.7%
     Required Skills : Python, SQL, Machine Learning, Statistics, ...
```

---

## 📂 Files

```
tech_stack_recommender.py   ← Main engine (full pipeline)
raw_skills.csv              ← Dataset of 15 job roles with skill tags
README.md                   ← This file
```

---

## 🗄️ Dataset (`raw_skills.csv`)

Contains **15 job roles**, each with a set of required skill tags.

| Role | Key Skills |
|------|-----------|
| Data Scientist | Python, SQL, Machine Learning, Statistics |
| DevOps Engineer | AWS, Docker, Kubernetes, CI/CD, Linux |
| Frontend Developer | JavaScript, React, CSS, HTML |
| Machine Learning Engineer | Python, TensorFlow, PyTorch, Algorithms |
| Cloud Architect | AWS, Azure, Cloud, Kubernetes, Security |
| Full Stack Developer | JavaScript, Python, React, SQL, APIs |
| ... | *(15 roles total)* |

You can add more roles by simply adding rows to `raw_skills.csv`.

---

## 🧠 How It Works — 4-Step Pipeline

```
Input Skills → Ingest → TF-IDF Vectorize → Cosine Score → Sort → Top-3 Output
```

**Step 1 — Ingestion:** Validates user input (minimum 3 skills required)

**Step 2 — TF-IDF Vectorization:** Converts both user skills and job role skills into numerical vectors in the same vocabulary space. Rare/specific skills (e.g. `Kubernetes`) get higher weights than common ones (e.g. `Python`)

**Step 3 — Cosine Similarity Scoring:** Measures the angle between the user's vector and each job role's vector. Score of 1.0 = perfect match, 0.0 = no overlap

**Step 4 — Sort + Filter:** Ranks all 15 roles by score, returns Top 3 only

---

## ⚙️ Configuration

```python
DATASET_PATH        = "raw_skills.csv"   # Path to your dataset
MIN_REQUIRED_SKILLS = 3                  # Minimum skills user must enter
TOP_N               = 3                  # Number of recommendations returned
```

---

## ⚠️ Edge Cases Handled

| Scenario | Behaviour |
|----------|-----------|
| Less than 3 skills entered | Error raised with a clear message |
| Empty or blank input | Stripped and counted as 0, error raised |
| Skills not in dataset vocabulary | Cold Start detected, user notified |
| No input at all | Demo runs with example skills automatically |

---

## 🧪 Test Results

All 7 test cases passed with zero errors:

| Test | Input | Expected | Status |
|------|-------|----------|--------|
| ML skills | Python, ML, Statistics | ML Engineer #1 | ✅ Pass |
| DevOps skills | AWS, Docker, Linux | DevOps Engineer #1 (60.7%) | ✅ Pass |
| Frontend skills | JavaScript, React, CSS | Frontend Dev #1 (62.2%) | ✅ Pass |
| Cold Start | Cooking, Painting, Dancing | Cold Start warning | ✅ Pass |
| Too few skills | Python, SQL | ValueError raised | ✅ Pass |
| Empty strings | `"", " ", ""` | ValueError raised | ✅ Pass |
| CSV load | — | 15 roles loaded | ✅ Pass |

**No bugs found. Code is submission-ready.**

---

## 🔮 Possible Improvements (Future Scope)

- Add more job roles to `raw_skills.csv` to improve coverage
- Build a web UI using Flask or Streamlit
- Show skill gap — what skills user is missing for each recommended role
- Support fuzzy matching for typos (e.g. "javascrpt" → "javascript")

---

## 👤 Author

Sudhan Mohana Das
DecodeLabs Industrial Training | Batch 2026

---

## 📄 License

This project is part of the DecodeLabs training curriculum. For educational use only.
