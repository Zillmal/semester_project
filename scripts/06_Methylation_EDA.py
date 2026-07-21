#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

(Path(PROJECT_ROOT) / "results" / "figures").mkdir(parents=True, exist_ok=True)
(Path(PROJECT_ROOT) / "results" / "tables").mkdir(parents=True, exist_ok=True)

meth = pd.read_csv(Path(PROJECT_ROOT) / "data" / "processed" / "meth_pam50.csv", index_col=0)


# overall beta-vals distribtion 
all_beta = meth.to_numpy().ravel()
all_beta = all_beta[~pd.isna(all_beta)]

# Plot the distribution of all observed PAM50 promoter methylation beta values.

plt.figure(figsize=(8, 5))
plt.hist(all_beta, bins=50, edgecolor="black")
plt.xlabel("Methylation beta value")
plt.ylabel("Number of CpG measurements")
plt.title("Distribution of PAM50 Promoter Methylation Beta Values")
plt.tight_layout()

plt.savefig(
    Path(PROJECT_ROOT) / "results" / "figures" / "pam50_promoter_beta_distribution_before_imputation.png",
    dpi=300,
    bbox_inches="tight"
)

#plt.show()


# In[11]:


# Calculate the interquartile range (IQR) for each CpG.
summary = meth.describe().T

summary["iqr"] = meth.quantile(0.75) - meth.quantile(0.25)

summary.to_csv(
    Path(PROJECT_ROOT) / "results" / "tables" / "pam50_promoter_cpg_variation_summary.csv"
)

summary[["mean", "std", "min", "25%", "50%", "75%", "max", "iqr"]].sort_values(
    "iqr"
).head(15)


# In[12]:


low_variance_cpgs = summary[summary["iqr"] < 0.02]

print(f"Low-variation CpGs: {len(low_variance_cpgs)} / {meth.shape[1]}")


# In[13]:


# view high and low variation to understand whether CpGs show meaningful between patient variation
high_var_cpgs = summary.sort_values("iqr", ascending=False).head(6).index
low_var_cpgs = summary.sort_values("iqr").head(6).index

fig, axes = plt.subplots(2, 6, figsize=(18, 6))

for ax, cpg in zip(axes[0], high_var_cpgs):
    ax.hist(meth[cpg].dropna(), bins=20, edgecolor="black")
    ax.set_title(cpg)
    ax.set_xlabel("Beta")

for ax, cpg in zip(axes[1], low_var_cpgs):
    ax.hist(meth[cpg].dropna(), bins=20, edgecolor="black")
    ax.set_title(cpg)
    ax.set_xlabel("Beta")

plt.tight_layout()

plt.savefig(
    Path(PROJECT_ROOT) / "results" / "figures" / "high_vs_low_variance_cpg_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

#plt.show()


