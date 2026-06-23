#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

Path("../results/figures").mkdir(parents=True, exist_ok=True)
Path("../results/tables").mkdir(parents=True, exist_ok=True)

meth = pd.read_csv("../data/processed/meth_pam50.csv", index_col=0)

print("Methylation matrix shape:", meth.shape)
print("Total missing beta values:", meth.isna().sum().sum())
print("Overall missing fraction:", meth.isna().mean().mean())

display(meth.iloc[:5, :5])


# In[2]:


# record missingness 
missing_before = pd.DataFrame({
    "missing_count": meth.isna().sum(axis=1),
    "missing_fraction": meth.isna().mean(axis=1)
})

missing_before.to_csv(
    "../results/tables/methylation_missingness_per_patient_before_imputation.csv"
)

cpg_missing_before = pd.DataFrame({
    "missing_count": meth.isna().sum(axis=0),
    "missing_fraction": meth.isna().mean(axis=0)
})

cpg_missing_before.to_csv(
    "../results/tables/methylation_missingness_per_cpg_before_imputation.csv"
)

print(missing_before["missing_fraction"].describe())
print(cpg_missing_before["missing_fraction"].describe())


# In[3]:


# Find CpGs that are completely missing
all_missing_cpgs = meth.columns[meth.isna().all()]

print("All-missing CpGs:", len(all_missing_cpgs))

pd.Series(all_missing_cpgs, name="cpg_id").to_csv(
    "../results/tables/all_missing_cpgs_removed_before_knn.csv",
    index=False
)

# Remove them before KNN
meth_for_imputation = meth.drop(columns=all_missing_cpgs)

print("Original shape:", meth.shape)
print("Shape used for KNN:", meth_for_imputation.shape)


# In[4]:


# test out different K values for KNN imputation
X = meth_for_imputation.copy()

# Reproducible random masking
rng = np.random.default_rng(42)

# Only choose cells that were originally observed
observed_positions = np.argwhere(~X.isna().to_numpy())

# Hide 2% of originally observed cells
n_mask = int(0.02 * len(observed_positions))
masked_positions = observed_positions[
    rng.choice(len(observed_positions), size=n_mask, replace=False)
]

# Make a copy and hide those known values
X_masked = X.copy()

true_values = []

for row_idx, col_idx in masked_positions:
    true_values.append(X.iat[row_idx, col_idx])
    X_masked.iat[row_idx, col_idx] = np.nan

true_values = np.array(true_values)

# Test several K values
k_values = [3, 5, 10]
results = []

for k in k_values:
    imputer = KNNImputer(
        n_neighbors=k,
        weights="distance"
    )

    X_imputed_array = imputer.fit_transform(X_masked)

    X_imputed = pd.DataFrame(
        X_imputed_array,
        index=X.index,
        columns=X.columns
    )

    predicted_values = np.array([
        X_imputed.iat[row_idx, col_idx]
        for row_idx, col_idx in masked_positions
    ])

    mae = mean_absolute_error(true_values, predicted_values)
    rmse = mean_squared_error(true_values, predicted_values) ** 0.5

    results.append({
        "k_neighbors": k,
        "masked_cells": n_mask,
        "MAE": mae,
        "RMSE": rmse
    })

knn_comparison = pd.DataFrame(results).sort_values("MAE")

print(knn_comparison)

knn_comparison.to_csv(
    "../results/tables/knn_neighbor_sensitivity.csv",
    index=False
)


# In[5]:


# impoutation with KNN
k = 10

imputer = KNNImputer(
    n_neighbors=k,
    weights="distance"
)

meth_imputed_array = imputer.fit_transform(meth_for_imputation)

meth_imputed = pd.DataFrame(
    meth_imputed_array,
    index=meth_for_imputation.index,
    columns=meth_for_imputation.columns
)

print("Missing values before:", meth_for_imputation.isna().sum().sum())
print("Missing values after:", meth_imputed.isna().sum().sum())
print("Minimum imputed beta value:", meth_imputed.min().min())
print("Maximum imputed beta value:", meth_imputed.max().max())


# In[6]:


# check originally observed values are unchanged
observed_mask = ~meth.isna()

max_difference_observed = (
    meth_imputed[observed_mask] - meth[observed_mask]
).abs().max().max()

print("Maximum difference among originally observed values:",
      max_difference_observed)

assert np.isclose(max_difference_observed, 0), (
    "Observed methylation values changed unexpectedly."
)


# In[7]:


# save the imputed methylation matrix
meth_imputed.to_csv(
    "../data/processed/meth_pam50_knn_imputed.csv"
)

print("Saved: ../data/processed/meth_pam50_knn_imputed.csv")


# In[8]:


# show what knn actually filled in 
imputed_mask = meth_for_imputation.isna()

imputed_values = meth_imputed.where(imputed_mask).to_numpy().ravel()
imputed_values = imputed_values[~np.isnan(imputed_values)]

observed_values = meth_for_imputation.to_numpy().ravel()
observed_values = observed_values[~np.isnan(observed_values)]

plt.figure(figsize=(9, 5))

plt.hist(
    observed_values,
    bins=50,
    density=True,
    alpha=0.45,
    label="Originally observed beta values"
)

plt.hist(
    imputed_values,
    bins=50,
    density=True,
    alpha=0.65,
    label="KNN-imputed beta values"
)

plt.xlabel("Methylation beta value")
plt.ylabel("Density")
plt.title("Distribution of Observed and KNN-Imputed PAM50 Promoter Beta Values")
plt.legend()
plt.tight_layout()

plt.savefig(
    "../results/figures/observed_vs_imputed_beta_distribution.png",
    dpi=300
)

plt.show()

print("Number of imputed values:", len(imputed_values))
print(pd.Series(imputed_values).describe())


# In[9]:


# check that mean values are not shifted too much by imputation
mean_before = meth_for_imputation.mean(axis=0, skipna=True)
mean_after = meth_imputed.mean(axis=0)

mean_shift = (mean_after - mean_before).abs().sort_values(ascending=False)

print(mean_shift.describe())
print(mean_shift.head(10))


# In[10]:


# save summary of imputation results
imputation_summary = pd.DataFrame({
    "metric": [
        "patients",
        "promoter_CpGs",
        "promoter_CpGs_imputed",
        "KNN_neighbors",
        "missing_values_before",
        "missing_values_after",
        "overall_missing_fraction_before",
        "min_beta_after_imputation",
        "max_beta_after_imputation"
    ],
    "value": [
        meth.shape[0],
        meth.shape[1],
        meth_imputed.shape[1],
        k,
        int(meth.isna().sum().sum()),
        int(meth_imputed.isna().sum().sum()),
        meth.isna().mean().mean(),
        meth_imputed.min().min(),
        meth_imputed.max().max()
    ]
})

imputation_summary.to_csv(
    "../results/tables/knn_imputation_summary.csv",
    index=False
)

display(imputation_summary)


# In[ ]:




