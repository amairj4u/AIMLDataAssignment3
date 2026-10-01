# Comparing Classifiers: Predicting Bank Term-Deposit Subscriptions

**Practical Application Assignment 3** comparing K-Nearest Neighbors, Logistic Regression,
Decision Trees, and Support Vector Machines on a Portuguese bank's telemarketing data.

---

**Data source:** UCI Machine Learning Repository — *Bank Marketing* dataset
(Moro, S., Cortez, P. & Rita, P., 2014).

---

## Business problem

A Portuguese bank sells term deposits by calling customers on the phone. **Only about 11 in
every 100 calls ends in a sale**, so roughly 89% of the call centre's effort produces nothing.

**Can we predict, before we dial, which customers are likely to say yes — so agents spend their
limited hours on the most promising people?**

This is a binary classification problem. The target is `y` (did the customer subscribe?), and the
model's job is to **rank** customers so the bank can call the top of the list first.

---

## Method

| Stage | What was done |
|---|---|
| **Cleaning** | Removed 12 duplicate rows; kept `"unknown"` as a valid category rather than discarding ~25% of the data; recoded `pdays = 999` into a clear `contacted_before` flag; grouped the three `basic.*` education levels |
| **Leakage control** | **Dropped `duration`** (call length). It is only known *after* a call ends, so a usable model cannot have it. Including it would have inflated every score |
| **Transformation** | `OneHotEncoder` + `StandardScaler` inside a `ColumnTransformer`, wrapped in a `Pipeline` so no information leaks from validation folds into training |
| **Split** | 80/20 stratified train/test split (32,940 / 8,236 rows) |
| **Models** | Logistic Regression, KNN, Decision Tree, SVM — first with defaults, then tuned |
| **Tuning** | `GridSearchCV` with 5-fold stratified cross-validation, scored on ROC-AUC. `class_weight='balanced'` was included in the grids to address the 89/11 class imbalance |

### Evaluation metric: ROC-AUC

Accuracy is actively misleading here. A model that predicts **"no" for everyone** scores
**88.7% accuracy** and finds **zero** customers.

ROC-AUC was chosen because it measures **ranking quality**, which is exactly what a call list
needs; it is **not fooled by class imbalance**; and it is **threshold-independent**, so the bank
can decide later how deep into the list to call. Recall and precision are reported alongside it
because those translate directly into calls made and sales closed.

---

## Results

### Default settings

| Model | Train time (s) | Test accuracy | Test ROC-AUC | Test recall |
|---|---|---|---|---|
| Logistic Regression | 0.25 | 0.899 | **0.801** | 0.208 |
| KNN | 0.09 | 0.894 | 0.738 | 0.295 |
| Decision Tree | 0.29 | 0.836 | 0.618 | 0.330 |
| SVM | 54.96 | 0.900 | 0.689 | 0.239 |

Note how all four sit at 84–90% accuracy while their AUC ranges from 0.62 to 0.80 — accuracy
hides the difference that matters.

### After cross-validation and grid search

| Model | CV AUC | Test ROC-AUC | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.791** | 0.801 | 0.830 | **0.648** | 0.360 | **0.463** |
| Decision Tree | 0.786 | **0.806** | 0.899 | 0.263 | 0.627 | 0.371 |
| KNN | 0.769 | 0.784 | 0.900 | 0.241 | 0.659 | 0.353 |
| SVM | 0.714 | 0.701 | 0.899 | 0.179 | 0.692 | 0.284 |

**Winner: Logistic Regression.** The tuned Decision Tree is a statistical tie on the test set
(0.806 vs 0.801 — smaller than sampling noise), so the model was chosen on the
**cross-validation** score, leaving the test set as an honest final exam. Logistic Regression is
also the better operational pick: it trains in seconds, outputs calibrated probabilities, and can
explain why it scored a customer the way it did.

Tuning turned the Decision Tree from the worst model into a contender (AUC 0.62 → 0.81) by
capping `max_depth` at 7 — the untuned tree had been memorising the training data at 99.5%
training accuracy.

---

## Key findings

### 1. The model works, and the payoff is large

| Strategy | Sales per 100 calls |
|---|---|
| Calling in random order | ~11 |
| Calling the model's top-ranked 10% | **~50** |

Calling only the top **30%** of the ranked list still reaches **73% of all subscribers**. The
call centre could cut calling volume by 70% while keeping roughly three-quarters of its sales.

### 2. The economy decides more than the customer does
Interest rates and employment levels dominate every model. `emp.var.rate` has an odds ratio of
**0.20** — a one-standard-deviation rise cuts the odds of subscribing to one fifth. The decision
tree independently spends **61%** of its splitting power on `nr.employed` alone. Age, job and
education barely register.

### 3. Past customers are the best customers
Customers who accepted a previous campaign subscribe again **65%** of the time — about six times
the average. This group is small, easy to identify, and under-used.

### 4. The calendar is being used backwards
May receives the most calls of any month (13,767) yet converts at only **6.4%**. March (50.5%),
December (48.9%), September (44.9%) and October (43.9%) convert brilliantly but receive very few
calls.

### 5. Repeated calling destroys value
The first call converts at 13%, the fourth or fifth at 8.7%, the eleventh at 3.1%. One customer
in this dataset was called **56 times**.

### 6. Mobile beats landline roughly three to one
Cellular contacts convert at 14.7%, landline at 5.2%.

---

**[Link Jupyter Notebook →](bank_marketing_classifier_comparison.ipynb)**
