# 💬 Comment Category Prediction

Predicting the category a platform assigns to a user comment, using text, engagement, and system-flag features

![Python](https://img.shields.io/badge/Python-3.12-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-boosting-green) ![Pandas](https://img.shields.io/badge/Pandas-data-informational) ![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)

**Best Model: Coordinate-Ascent Ensemble (5 models) | Validation Macro-F1: 0.7981 | Leaderboard Score: 0.79640**

## 📌 Overview

This project predicts `label` — the final category a platform's moderation system assigns to a comment — from the comment's text, its engagement metrics (upvotes/downvotes), timestamp, and a set of system-detected topic flags (race, religion, gender, disability references). It's a 4-class, heavily imbalanced classification problem from the [Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge) on Kaggle.

The pipeline combines TF-IDF text vectorization with engineered engagement/text-shape features, trains five diverse classifiers, and blends them with a coordinate-ascent search that optimizes directly for the competition's macro-F1-based custom metric rather than raw accuracy.

## 🗂️ Repository Structure

```
comment-category-prediction/
├── comment_category_solution_v6_professional.ipynb   # Main notebook: EDA, features, models, ensemble
├── Comment_Category_Prediction_Project_Report.pdf     # Full project report
├── submission.csv                                      # Example output format
└── README.md
```

## 📊 Dataset

| File | Description |
|---|---|
| `train.csv` | 198,000 labeled comments with all feature columns |
| `test.csv` | 102,000 unlabeled comments (same features, no `label`) |
| `Sample.csv` | Submission format template |

**Columns:** `comment` (text), `created_date`, `post_id`, `emoticon_1–3`, `upvote`, `downvote`, `if_1`/`if_2` (hidden internal features), `race`, `religion`, `gender`, `disability` (system-detected topic flags), `label` (target, 4 classes).

**Class balance is heavily skewed:** 57.7% / 8.0% / 31.5% / 2.8% across the four label values — the central challenge of this dataset, and the reason accuracy alone is a misleading metric here.

## 🔧 Pipeline

1. **Data Loading** — load train/test/sample submission; parse `created_date`; sanity-check shapes and class balance.
2. **EDA** — missing values, target class imbalance, comment length/word count by class, engagement (votes) by class, topic-flag prevalence by class, feature correlation heatmap.
3. **Feature Engineering** — 18 engineered features: text-shape stats (length, word count, punctuation/uppercase/digit ratios), date parts (hour/day-of-week/month), vote ratios, log-transformed engagement, and a leak-safe `post_id` thread-activity count.
4. **Text Vectorization** — word-level TF-IDF (1–2 grams, 10,000 features, English stopwords removed).
5. **Train/Val Split** — stratified 88/12 split.
6. **Model Training** — 5 candidate classifiers: Logistic Regression, Complement Naive Bayes, SGD (log-loss), LightGBM, Random Forest.
7. **Model Comparison** — accuracy vs. macro-F1 per model, confusion matrix for the top individual model.
8. **Ensembling** — coordinate-ascent weight search across all 5 models, optimizing validation macro-F1 directly; any model that never improves the blend settles at weight 0.
9. **Submission** — refit selected models on full training data, blend test predictions, write `submission.csv`.

## 🧠 Models & Results

| Model | Notes |
|---|---|
| Logistic Regression | TF-IDF + numeric features, `class_weight="balanced"` |
| Complement Naive Bayes | Text-only, designed for imbalanced text classification |
| SGD (log-loss) | TF-IDF + numeric features, `class_weight="balanced"` |
| LightGBM ⭐ | Trained directly on the sparse TF-IDF + numeric matrix, `class_weight="balanced"` |
| Random Forest | Numeric/engineered features only, added for blend diversity |

**Best result:** coordinate-ascent ensemble of the above, selected and weighted by validation macro-F1.

| Metric | Score |
|---|---|
| Validation Macro-F1 | 0.7981 |
| Leaderboard Score | 0.79640 |

## 🔍 Key finding

Early iterations optimized for accuracy and a 5-model blend scored *worse* on the leaderboard (0.775 → 0.740) despite higher validation accuracy. Comparing local metrics against leaderboard scores across submissions showed the "Custom Metric" tracks **macro-F1**, not accuracy — expected, given the ~58/8/32/3% class split. Switching every modeling decision (class weights, model selection, ensemble weighting) to macro-F1 recovered the regression and then beat the original single-model baseline.

## ✨ Key Features Engineered

- **Text-shape:** character length, word count, average word length, unique-word ratio, uppercase/digit ratios, punctuation counts
- **Temporal:** hour, day-of-week, month extracted from `created_date`
- **Engagement:** total votes, vote ratio (upvote − downvote, normalized), log-transformed total votes
- **Thread activity:** comment count per `post_id`, computed across train + test to avoid leakage
- **Text vector:** word-level TF-IDF, 1–2 grams, 10,000 features

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/comment-category-prediction.git
cd comment-category-prediction

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm

# Launch the notebook
jupyter notebook comment_category_solution_v6_professional.ipynb
```

Built to run against the competition data at `/kaggle/input/competitions/comment-category-prediction-challenge/` — run as a Kaggle notebook for the full experience (or point the `DATA_DIR` variable at a local copy of the three CSVs). Full run (EDA + all 5 models + ensemble) takes roughly 12–15 minutes on a standard CPU instance.

## 🛠️ Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn · LightGBM

## 📁 Outputs

- `submission.csv` — final `label` predictions in the required submission format
- Inline EDA and model-comparison charts (rendered directly in the notebook)

## 📈 Future Improvements

- Grouped (by `post_id`) train/validation split, to rule out thread-level leakage inflating validation scores
- Character-level TF-IDF or subword embeddings alongside word-level TF-IDF
- Stacking with a meta-learner in place of coordinate-ascent weight search
- Threshold/decision-boundary tuning per class, given the metric weights all four classes equally
- Broader hyperparameter search (grid/Bayesian) for LightGBM and Random Forest

## 📄 License

This project is available under the MIT License.
