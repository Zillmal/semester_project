#!/usr/bin/env python
# coding: utf-8

# ## Stage 11 — mRNA-only LASSO-Cox baseline (survival)
# 
# 
# **Goal:** Survival prediction based on mRNA of PAM50 genes only. This is the *baseline* against which the multi-omics model can be compared.
# 
# **Implementation:** *Cox* proportional-hazards model with *LASSO* (L1) penalty, fit with `CoxnetSurvivalAnalysis`. Nested 5-fold CV: outer = shared folds (comparable across models), inner = alpha selection.
# 
# - *LASSO* shrinks unimportant gene coefficients to exactly 0 (feature selection)
# - evaluation by *C-index* (ranks predicted vs. observed survival; 1 = perfect predictions, 0.5 = random chance level)

# ## Imports & setup
# 

# In[1]:


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

warnings.simplefilter("ignore") # silences Coxnet convergence warnings
Path("../results/tables").mkdir(parents=True, exist_ok=True)


# ## Load and align data
# 
# `rna` expression data (patients x genes)
# 
# `surv` table (rows patients, columns: subtype, vital status, days_to_death, days_to_last_follow_up, event, time) 
# 
# `folds` predefined CV-fold assignments for each patient (shared across models).
# 
# - `set_index("patient")` makes patient ID the index for later combination
# - `surv["time"] > 0` keep only patients with valid, positive follow-up time
# - `intersection` only keep patients present in all three tables
# - patient alignment with `loc`
# - `fold_id` outer CV fold per patient (same splits as all other models)

# In[5]:


rna = pd.read_csv("../data/processed/rna_pam50.csv").set_index("patient")
surv = pd.read_csv("../data/processed/survival_luminal_clean.csv").set_index("patient")
folds = pd.read_csv("../data/processed/cv_fold_assignments.csv").set_index("patient")

surv = surv[surv["time"].notna() & (surv["time"] > 0)]
patients = rna.index.intersection(surv.index).intersection(folds.index)
rna, surv = rna.loc[patients], surv.loc[patients]
fold_id = folds.loc[patients, "fold"]
print(f"Patients: {len(patients)} | genes: {rna.shape[1]} | folds: {sorted(fold_id.unique())}")


# ## Helper function to construct survival target
# 
# For each patient
# - `event` status (False = No observed death, True = Dead)
# - and `time` (days between diagnosis and death, or days between diagnosis and last contact)
# 
# is collected and paired. 
# 

# In[ ]:


def survival_y(ids):
    return Surv.from_arrays(event=surv.loc[ids, "event"].astype(bool).values,
                            time=surv.loc[ids, "time"].values)


# ## Helper function to construct fold-safe features
# 
# Scaling of gene expression data is necessary because LASSO is sensitive to different scales.
# 
# Scaling function is convenient to easily scale anew within each CV-fold (where different samples are test samples and could lead to data leakage)
# 
# 
# - `.fit(rna.loc[train_ids])` scaler is only fitted with training data to prevent *data leakage*
# - scaling is then applied (`.transform`) to train and test samples
# - scaling matters because LASSO's penalty is scale-sensitive (features must be comparable)

# In[8]:


def build_features(train_ids, test_ids):
    scaler = StandardScaler().fit(rna.loc[train_ids])
    X_tr = scaler.transform(rna.loc[train_ids])
    X_te = scaler.transform(rna.loc[test_ids])
    return X_tr, X_te


# ## Function for alpha tuning
# 
# Nested CV: inner 5-fold CV used for tuning/optimizing L1 strength (`alpha`) separate from outer test folds (no data leakage during tuning)
# 
# 
# - `l1_ratio = 1.0` = *LASSO* regularization (sets most coefficients to 0, e.g. feature selection)
# - per inner fold: fit the whole L1 path, score each `alpha` on the inner-validation fold by C-index
# - `try/except` Coxnet can fail to converge at very small `alpha` (almost no penalty) --> skips failing convergence
# - return the `alpha` with the best mean inner C-index

# In[9]:


def select_alpha(X, y, alphas):
    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    score_sum = np.zeros(len(alphas))
    for i_tr, i_va in inner.split(X):
        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=alphas, max_iter=100000)
        try:
            model.fit(X[i_tr], y[i_tr])
        except (ArithmeticError, ValueError):  # Coxnet may fail to converge at small alphas
            continue
        for j, a in enumerate(alphas):
            risk = model.predict(X[i_va], alpha=a)
            score_sum[j] += concordance_index_censored(
                y[i_va]["event"], y[i_va]["time"], risk)[0]
    return alphas[int(np.argmax(score_sum))]


# ## Nested CV 
# 
# combines helper functions `build_features`, `survival_y` and `select_alpha`
# 
# Outer loop over the shared folds = honest evaluation on unseen patients.
# 
# - `train_ids`/`test_ids` outer split per fold
# `fold_id==f` for samples that are used as test set in respective fold
# 
# - `buils_features` for scaling
# - `survival_y` for target construction
# 
# - `.alphas_` fit one coxnet model on training data for each alpha
# - tests the 50 alphas between alpha_max (the alpha that results in every coefficient being 0) and alpha_max x 0.01
# - `select_alpha` picks the best
# - `final` fit on full training fold with the best alpha, predict risk on the test fold
# - `n_sel` count how many genes are kept after LASSO regularization
# - `lasso_cox_cv_results.csv` results table with one row per fold (best alpha, number selected genes, C-index)

# In[10]:


rows = []
for f in sorted(fold_id.unique()):
    train_ids = fold_id.index[fold_id != f]
    test_ids = fold_id.index[fold_id == f]
    X_tr, X_te = build_features(train_ids, test_ids)
    y_tr, y_te = survival_y(train_ids), survival_y(test_ids)

    alphas = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=50,
                                    alpha_min_ratio=0.01, max_iter=100000).fit(X_tr, y_tr).alphas_
    best_alpha = select_alpha(X_tr, y_tr, alphas)

    final = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[best_alpha],
                                   max_iter=100000).fit(X_tr, y_tr)
    risk = final.predict(X_te)
    ci = concordance_index_censored(y_te["event"], y_te["time"], risk)[0]
    n_sel = int((final.coef_.ravel() != 0).sum())
    rows.append({"fold": f, "alpha": best_alpha, "n_features_total": X_tr.shape[1],
                 "n_features_selected": n_sel, "test_c_index": ci, "n_test": len(test_ids)})
    print(f"Fold {f}: C-index={ci:.3f} | alpha={best_alpha:.4g} | selected={n_sel}")

cv = pd.DataFrame(rows)
cv.to_csv("../results/tables/lasso_cox_cv_results.csv", index=False)

mean_ci, sd_ci = cv["test_c_index"].mean(), cv["test_c_index"].std()
print(f"\nBenchmark mRNA-only LASSO-Cox C-index: {mean_ci:.3f} +/- {sd_ci:.3f} (5-fold CV)")

