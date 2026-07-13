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
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
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

# In[2]:


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

# In[3]:


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

# In[4]:


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

# In[ ]:


# selected alpha using the 1 standard error rule 

def select_alpha(X, y, alphas):
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.full((inner.get_n_splits(), len(alphas)), np.nan)

    for fold_idx, (i_tr, i_va) in enumerate(inner.split(X, y["event"])):
        model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=alphas,
            max_iter=100000
        )

        try:
            model.fit(X[i_tr], y[i_tr])

            for j, alpha in enumerate(alphas):
                risk = model.predict(X[i_va], alpha=alpha)
                scores[fold_idx, j] = concordance_index_censored(
                    y[i_va]["event"],
                    y[i_va]["time"],
                    risk
                )[0]

        except (ArithmeticError, ValueError) as e:
            print(f"Warning: inner fold {fold_idx} failed ({e})")
            continue

    mean_scores = np.nanmean(scores, axis=0)
    std_scores = np.nanstd(scores, axis=0, ddof=1)
    n_scores = np.sum(~np.isnan(scores), axis=0)
    se_scores = std_scores / np.sqrt(n_scores)

    best_idx = np.nanargmax(mean_scores)
    threshold = mean_scores[best_idx] - se_scores[best_idx]

    candidate_idx = np.where(mean_scores >= threshold)[0]

    chosen_idx = candidate_idx[0]

    return alphas[chosen_idx]


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

# In[ ]:


# switch alpha if too small and fails to converge
def fit_final_model_with_fallback(X_tr, y_tr, selected_alpha, alphas):
    sorted_alphas = np.sort(alphas)

    candidate_alphas = sorted_alphas[sorted_alphas >= selected_alpha]

    for alpha in candidate_alphas:
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=1.0,
                alphas=[alpha],
                max_iter=100000
            )
            model.fit(X_tr, y_tr)
            return model, alpha
        except ArithmeticError:
            print(f"Alpha {alpha:.4g} failed; trying larger alpha...")

    raise RuntimeError("No alpha worked. Try increasing alpha_min_ratio.")


# In[ ]:


rows = []
path_models = {}
# reset risk-score file once per run to avoid duplicate-fold appends
risk_path = Path("../results/tables/lasso_cox_mrna_risk_scores.csv")
risk_path.unlink(missing_ok=True)

for f in sorted(fold_id.unique()):
    train_ids = fold_id.index[fold_id != f]
    test_ids = fold_id.index[fold_id == f]
    X_tr, X_te = build_features(train_ids, test_ids)
    y_tr, y_te = survival_y(train_ids), survival_y(test_ids)

    alphas = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=100,
                                    alpha_min_ratio=0.01, max_iter=100000).fit(X_tr, y_tr).alphas_
    best_alpha = select_alpha(X_tr, y_tr, alphas)
    
    final, used_alpha = fit_final_model_with_fallback(X_tr, y_tr, best_alpha, alphas)
    
    path_model = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=100, alpha_min_ratio=0.01, max_iter=100000)
    path_model.fit(X_tr, y_tr)
    path_models[f] = path_model

    train_risk = final.predict(X_tr)
    risk = final.predict(X_te)
    train_ci = concordance_index_censored(y_tr["event"], y_tr["time"], train_risk)[0]
    ci = concordance_index_censored(y_te["event"], y_te["time"], risk)[0]
    n_sel = int((final.coef_.ravel() != 0).sum())
    rows.append({"fold": f, "used alpha": used_alpha, "n_features_total": X_tr.shape[1],
                 "n_features_selected": n_sel, "train_c_index": train_ci, "test_c_index": ci, "n_test": len(test_ids)})
    print(f"Fold {f}: Train C-index={train_ci:.3f} | Test C-index={ci:.3f} | alpha={used_alpha:.4g} | features={X_tr.shape[1]} | selected={n_sel}")

    # Save risk scores
    risk_rows = [{"patient": pid, "fold": f, "risk_score": float(r)}
             for pid, r in zip(test_ids, risk)]
    risk_path = Path("../results/tables/lasso_cox_mrna_risk_scores.csv")
    pd.DataFrame(risk_rows).to_csv(risk_path, mode="a", header=not risk_path.exists(), index=False)
    
cv = pd.DataFrame(rows)
cv.to_csv("../results/tables/lasso_cox_cv_results.csv", index=False)

best_fold = cv.loc[cv["test_c_index"].idxmax(), "fold"]
path_model = path_models[best_fold]

mean_ci, sd_ci = cv["test_c_index"].mean(), cv["test_c_index"].std()
print(f"\nBenchmark mRNA-only LASSO-Cox C-index: {mean_ci:.3f} +/- {sd_ci:.3f} (5-fold CV)")


# In[13]:


# Plotting the coefficient paths for the best fold
best_alpha_for_plot = cv.loc[cv["fold"] == best_fold, "used alpha"].iloc[0]

plt.figure(figsize=(10, 6))

importance = np.max(np.abs(path_model.coef_), axis=1)
top_genes = np.argsort(importance)[-10:]

for i in top_genes:
    plt.plot(np.log10(path_model.alphas_), path_model.coef_[i], linewidth=1, label=rna.columns[i])

plt.axvline(np.log10(best_alpha_for_plot), color='red', linestyle='--', label=f"Selected alpha = {best_alpha_for_plot:.4g}")

plt.xlabel("log10(alpha)")
plt.ylabel("Coefficient")
plt.title(f"LASSO-Cox Coefficient Paths (mRNA, Best Fold = {best_fold})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("../results/figures/lasso_cox_mRNA_coefficient_paths.png", dpi=300)
plt.show()


# In[ ]:




