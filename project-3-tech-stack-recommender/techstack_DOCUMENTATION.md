# Tech Stack Recommender — Project Documentation

**Project:** Tech Stack Recommender
**Type:** Content-Based AI Recommendation Engine (CLI)
**Batch:** DecodeLabs Industrial Training | 2026
**Language:** Python 3.8+
**Developer:** Subin Das

---

## 1. Project Overview

This project builds a content-based recommendation engine that takes a user's current tech skills and recommends the Top 3 most suitable job roles from a dataset of 15 roles. It uses TF-IDF vectorization and Cosine Similarity — the same math used in real-world recommendation systems.

It does NOT use collaborative filtering (requires user history) or simple binary overlap matching (not nuanced enough). It follows the exact 4-step pipeline mandated by the project brief.

---

## 2. Architecture (IPO Model)

```
[ INPUT ]              [ PROCESS ]                          [ OUTPUT ]
User Skills  →  Ingest → TF-IDF Vectorize → Cosine Score  →  Top-3 Roles
                          (Same vocab space)   (Sort + Filter)
```

---

## 3. File Breakdown

| Function | Step | Purpose |
|----------|------|---------|
| `load_role_dataset()` | Step 0 | Reads `raw_skills.csv`, normalizes skill strings |
| `ingest_user_skills()` | Step 1 | Validates input, enforces minimum 3 skills |
| `build_tfidf_vector_space()` | Step 2A | Builds shared TF-IDF vocabulary, returns item + user vectors |
| `score_roles_by_similarity()` | Step 2B | Computes cosine similarity: user vector vs every role |
| `sort_by_relevance()` | Step 3 | Sorts all roles by descending similarity score |
| `filter_top_n()` | Step 4 | Truncates to Top-N (default: 3) |
| `check_cold_start()` | Safety | Detects if all scores are 0.0, warns user |
| `display_recommendations()` | Output | Prints ranked results with match % |
| `run_recommender()` | Orchestrator | Runs all steps end-to-end |
| `main()` | Entry Point | CLI prompt, handles empty input with demo |

---

## 4. Key Technical Decisions

### Why TF-IDF and not binary overlap?
Binary matching treats every skill equally — a skill like `git` (appears in 9 out of 15 roles) gets the same weight as `kubernetes` (appears in only 3 roles). TF-IDF penalizes common terms via the Inverse Document Frequency component, so specific/rare skills like `kubernetes` or `cryptography` carry more weight than generic ones like `python` or `git`. This gives more meaningful match scores.

### Why Cosine Similarity and not Euclidean Distance?
Euclidean distance measures the absolute gap between vectors. A role with 10 skills would always appear "farther" from a user who listed only 3, even if those 3 skills are a perfect subset. Cosine similarity measures only the angle between vectors — it's magnitude-independent, making it the correct choice when comparing profiles of different lengths.

### Why multi-word skills get underscore-joined?
`TfidfVectorizer` tokenizes by word. "Machine Learning" would split into `machine` and `learning` — two separate tokens that could each partially match unrelated skills. Joining it to `machine_learning` makes it one atomic token, preserving the skill's identity exactly as required by the "same vocabulary space" requirement.

### Why minimum 3 skills?
With fewer than 3 skills, the user's TF-IDF vector is sparse — the similarity scores become unreliable because there isn't enough data to distinguish meaningful matches from noise. This is explicitly mandated in the project spec.

---

## 5. TF-IDF Weight Interpretation

When you enter `Python, Machine Learning, Statistics`, the weights assigned are:

| Skill | TF-IDF Weight | Why |
|-------|--------------|-----|
| machine_learning | 0.6570 | Specific — appears in fewer roles |
| statistics | 0.6032 | Specific — appears in fewer roles |
| python | 0.4522 | Generic — appears in many roles, down-weighted |

This means the engine correctly prioritizes roles where machine learning and statistics are listed, over roles that merely contain python.

---

## 6. Cold Start Detection

If the user enters skills with zero vocabulary overlap with the dataset (e.g. "Cooking, Painting, Dancing"), every cosine similarity score collapses to 0.0. Instead of silently returning a meaningless ranked list, `check_cold_start()` detects this state and surfaces a clear warning. This is the "Bypassing the Cold Start" requirement from the project brief.

---

## 7. Sample Test Results

**Input: Python, Machine Learning, Statistics**
```
#1  Machine Learning Engineer  →  53.6%
#2  AI Research Engineer       →  52.1%
#3  Data Scientist             →  51.7%
```

**Input: AWS, Docker, Linux**
```
#1  DevOps Engineer    →  60.7%
#2  Cloud Architect    →  22.8%
#3  Systems Admin      →  15.8%
```

**Input: JavaScript, React, CSS**
```
#1  Frontend Developer    →  62.2%
#2  Full Stack Developer  →  43.9%
#3  DevOps Engineer       →  0.0%   ← Cold Start boundary case
```

---

## 8. Testing Summary

All 7 test cases passed. Pipeline ran end-to-end without errors.

| Test | Result |
|------|--------|
| Syntax & imports | ✅ Clean |
| CSV loads (15 roles) | ✅ Correct |
| ML skills → ML Engineer #1 | ✅ Correct |
| DevOps skills → DevOps Engineer #1 | ✅ Correct |
| Frontend skills → Frontend Dev #1 | ✅ Correct |
| Cold Start (non-tech skills) | ✅ Detected & warned |
| < 3 skills → ValueError | ✅ Raised correctly |
| Empty strings → ValueError | ✅ Raised correctly |

**Zero bugs. No fixes required. Code is submission-ready.**

---

## 9. How to Run

```bash
# Install dependencies (one time)
pip install pandas scikit-learn

# Run the recommender
python tech_stack_recommender.py
```

---

## 10. Dependencies

| Library | Purpose |
|---------|---------|
| pandas | Load and manage the CSV dataset |
| scikit-learn (TfidfVectorizer) | Build TF-IDF vector representations |
| scikit-learn (cosine_similarity) | Compute similarity scores between vectors |

---

*Documentation prepared for DecodeLabs Project Submission | Batch 2026*
