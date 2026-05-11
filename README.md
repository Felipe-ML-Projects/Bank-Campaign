# Bank-Campaign - — Term Deposit Subscription Classifier

Classifying whether bank customers' will subscribe to a bank term deposit based on historical data. Marketing campaigns were based on phone calls.

A couple classification algorithms were developed:
1) Logistic Regression
2) K Nearest Neighbor
3) XGBoost
4) Decision Tree
5) Random Forest
6) Neural Network

# Table of Contents

- [Business Case](#-business-case)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Data Preprocessing](#-data-preprocessing)
- [Models Developed](#-models-developed)
- [Evaluation Metrics](#-evaluation-metrics)
- [Key Findings](#-key-findings)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)
- [Results Summary](#-results-summary)

# Business Case

A retail bank runs recurring **telephone-based marketing campaigns** to promote term deposit products to its existing customer base. A term deposit is a fixed-interest financial product where a customer locks in a sum of money for a defined period in exchange for a guaranteed return.

# The Problem with Traditional Campaigns

Traditionally, campaign teams call a broad list of customers without a targeting strategy. This approach leads to:

- **High call volume with low conversion rates** — most contacted customers do not subscribe.
- **Wasted operational costs** — agent time, telephony costs, and team resources are spent on unlikely prospects.
- **Customer experience degradation** — repeatedly contacting uninterested customers damages brand reputation and increases churn risk.

# The Opportunity

By applying machine learning to historical campaign data, the bank can:

1. **Score and rank customers** by their likelihood to subscribe before the next campaign.
2. **Prioritise outreach** to high-probability segments, cutting call volume without sacrificing conversion numbers.
3. **Reduce cost-per-acquisition** by focusing agent time where it counts.
4. **Improve customer satisfaction** by contacting only those most likely to be receptive.

# Business Impact Potential

| Metric | Without ML | With ML (Target) |
| Calls per conversion | ~12–15 | ~4–6 |
| Campaign ROI | Baseline | +30–50% |
| Agent productivity | Low | High |
| Customer churn from over-contact | Elevated | Reduced |

# Problem Statement

**Binary Classification:** Predict whether a customer will subscribe (`yes`) or not subscribe (`no`) to a bank term deposit following a phone-based marketing campaign.

- **Target Variable:** `y` — whether the customer subscribed (`yes` / `no`)
- **Task Type:** Supervised Binary Classification
- **Evaluation Focus:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

# Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

**File:** `bank-full.csv` 

### Features

| Feature | Type | Description |
| `age` | Numeric | Customer age |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status |
| `education` | Ordinal | Education level (`primary`, `secondary`, `tertiary`) |
| `default` | Binary | Has credit in default? |
| `balance` | Numeric | Average yearly balance (€) |
| `housing` | Binary | Has housing loan? |
| `loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `day` | Numeric | Last contact day of month |
| `month` | Ordinal | Last contact month of year |
| `duration` | Numeric | Last contact duration (seconds) |
| `campaign` | Numeric | Number of contacts during this campaign |
| `pdays` | Numeric | Days since last contact from a previous campaign |
| `previous` | Numeric | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous campaign *(dropped — high missingness)* |
| **`y`** | **Binary** | **Target: subscribed to term deposit?** |

# Class Imbalance Note

The dataset exhibits significant class imbalance — the majority of customers do **not** subscribe. This was especially impactful for the Neural Network models, which required careful handling via scaled inputs and early stopping.

## Project Structure

```
bank-campaign/
│
├── Bank_Campaign.ipynb        # Main analysis and modelling notebook
├── bank-full.csv              # Raw dataset (not included — download from UCI)
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## Data Preprocessing

The following preprocessing steps were applied before model training:

### 1. Missing Value Treatment
- `unknown` and `Unknown` string values were replaced with `NaN`.
- `education`, `job`, and `contact` were imputed using the **most frequent value** strategy (`SimpleImputer`).
- `poutcome` was **dropped entirely** due to excessive missing values.

### 2. Encoding
- **One-Hot Encoding** applied to nominal categorical columns: `job`, `marital`, `default`, `housing`, `loan`, `contact` (with `drop_first=True` to avoid multicollinearity).
- **Ordinal Encoding** applied to ordered variables:
  - `education`: `primary → secondary → tertiary`
  - `month`: `jan → feb → … → dec` (encoded as 1–12)

### 3. Target Encoding
- The target column `y` (`yes`/`no`) was binary-encoded using `LabelEncoder`.

### 4. Feature Scaling
- `StandardScaler` was applied to all features for distance-based models (K-NN) and Neural Networks.
- Tree-based models (Decision Tree, Random Forest, XGBoost) used unscaled features.

### 5. Train-Test Split
- **70% training / 30% testing**, with `random_state=42` for reproducibility.

### 6. Correlation Analysis
- A heatmap was generated to assess feature correlations and confirm no irrelevant or highly redundant features required removal.

---

## Models Developed

Six classification algorithms were developed and evaluated:

### 1. Logistic Regression
- **Library:** `sklearn.linear_model.LogisticRegression`
- **Key Setting:** `max_iter=2000` to ensure convergence
- **Evaluation:** Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Log Loss
- **ROC-AUC Note:** The model achieved an AUC of ~0.80, indicating it can rank a subscriber above a non-subscriber 80% of the time — strong performance for a linear baseline.

### 2. Decision Tree
- **Library:** `sklearn.tree.DecisionTreeClassifier`
- **Hyperparameter Tuning:** `GridSearchCV` with 5-fold cross-validation
- **Parameters Tuned:**
  - `criterion`: `['entropy', 'gini']`
  - `max_depth`: `[3, 4, 5, 6, None]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 5]`
- **Visualisation:** Best tree plotted with feature names and class labels.

### 3. Random Forest
- **Library:** `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameter Tuning:** `GridSearchCV`
- **Parameters Tuned:**
  - `n_estimators`: `[100, 200, 300, 500]`
  - `max_depth`: `[3, 5, 8, None]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 4]`
- **Visualisation:** One sample tree from the ensemble plotted for interpretability.

### 4. XGBoost
- **Library:** `xgboost.XGBClassifier`
- **Settings:** `objective='binary:logistic'`, `eval_metric='logloss'`
- **Note:** Used default hyperparameters as a strong initial baseline; suitable for further tuning.

### 5. K-Nearest Neighbors (K-NN)
- **Library:** `sklearn.neighbors.KNeighborsClassifier`
- **Settings:** `n_neighbors=5`, `metric='euclidean'`
- **Scaling:** `StandardScaler` applied — essential for distance-based methods.

### 6. Neural Network
Two implementations were built:

**A. PyTorch (SimpleANN)**
```
Input → Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1)
```
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam (lr=1e-3)`
- Epochs: 20
- Batch Size: 32

**B. TensorFlow / Keras**
```
Input → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)
```
- Loss: `binary_crossentropy`
- Optimizer: `Adam (lr=1e-3)`
- Callback: `EarlyStopping(patience=5, restore_best_weights=True)`
- Class imbalance caused initial instability — scaling inputs and early stopping were critical.

---

# Evaluation Metrics

All models were evaluated on the held-out test set using:

| Metric | Description |
| **Accuracy** | Proportion of correctly classified samples |
| **Precision** | Of predicted positives, how many are actually positive |
| **Recall** | Of all actual positives, how many were correctly predicted |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve — measures ranking ability |
| **Log Loss** | Penalises confident wrong predictions (Logistic Regression) |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN |

> **Business Priority:** In a marketing campaign context, **Recall** is especially important — missing a genuine subscriber (false negative) means lost revenue. The model should be optimised to minimise false negatives while keeping precision acceptable to avoid wasting call agent resources.


# Key Findings

- **Class Imbalance** is a core challenge — most customers do not subscribe, which skews naive accuracy upward and hurts recall on the minority (subscribed) class.
- **Call Duration** (`duration`) is a strong predictor, but is only known *after* a call is made, limiting its utility for pre-call scoring in production deployment.
- **Logistic Regression** performed surprisingly well as a baseline with AUC ~0.80, making it a strong candidate for a transparent, interpretable production model.
- **Neural Networks** suffered initially due to class imbalance, but improved with proper scaling and early stopping.
- **Tree-based ensemble methods** (Random Forest, XGBoost) are expected to yield the highest predictive accuracy with proper hyperparameter tuning.

---

# Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bank-campaign.git
cd bank-campaign
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `bank-full.csv` from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place it in the project root directory.

### 4. Run the Notebook

```bash
jupyter notebook Bank_Campaign.ipynb
```

# Dependencies

```
pandas
numpy
scikit-learn
xgboost
torch
tensorflow
matplotlib
seaborn
jupyter
```

Or install all at once:

```bash
pip install pandas numpy scikit-learn xgboost torch tensorflow matplotlib seaborn jupyter
```

---

# Results Summary

| Model | Train Accuracy | Test Accuracy | Notes |

| Logistic Regression - ~90% | AUC ≈ 0.80; strong interpretable baseline |

| Decision Tree | Tuned via GridSearchCV - Best params selected via 5-fold CV |

| Random Forest - GridSearchCV tuned; ensemble robustness |

| XGBoost - Default params; strong gradient boosting baseline |

| K-NN (k=5) - Euclidean distance; feature scaling required |

| Neural Network (PyTorch) 2-layer ANN; impacted by class imbalance |

| Neural Network (TensorFlow) Early stopping applied; scaled input required |

> **Note:** Exact numeric results depend on the runtime environment and any randomness in model fitting. Run the notebook to reproduce outputs.

---

# Future Work

- [ ] Apply **SMOTE** or **class weighting** to address class imbalance across all models
- [ ] Remove `duration` from features for a production-ready pre-call scoring model
- [ ] Add **threshold tuning** for Logistic Regression to optimise for Recall
- [ ] Build a **model comparison dashboard** with unified evaluation metrics across all models
- [ ] Explore **feature importance** plots for Random Forest and XGBoost
- [ ] Deploy the best model as a **REST API** for integration with CRM systems

---

# License

This project is for educational and portfolio purposes. The dataset is publicly available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

---

> *Built with Python, scikit-learn, XGBoost, PyTorch, and TensorFlow.*
