#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("../data/processed")
RESULTS_DIR = Path("../results")
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# In[2]:


survival = pd.read_csv(DATA_DIR / "survival_luminal_clean.csv")

print(survival.shape)
survival.head()


# In[3]:


print(survival["BRCA_Subtype_PAM50"].value_counts())
print(survival["event"].value_counts())


# In[4]:


survival["stratum"] = (
    survival["BRCA_Subtype_PAM50"].astype(str)
    + "_event_"
    + survival["event"].astype(str)
)

survival["stratum"].value_counts()


# In[5]:


stratum_counts = survival["stratum"].value_counts()

print(stratum_counts)

if (stratum_counts < 5).any():
    print("Warning: at least one stratum has fewer than 5 patients.")
else:
    print("All strata have enough patients for 5-fold CV.")


# In[6]:


N_SPLITS = 5
RANDOM_STATE = 42

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

survival["fold"] = -1

for fold_number, (_, validation_idx) in enumerate(
    skf.split(survival, survival["stratum"]),
    start=1
):
    survival.loc[validation_idx, "fold"] = fold_number

survival["fold"].value_counts().sort_index()


# In[7]:


assert (survival["fold"] != -1).all()
assert survival["patient"].is_unique

print("Every patient has exactly one fold assignment.")


# In[8]:


fold_distribution = (
    survival
    .groupby(["fold", "BRCA_Subtype_PAM50", "event"])
    .size()
    .reset_index(name="n_patients")
    .sort_values(["fold", "BRCA_Subtype_PAM50", "event"])
)

fold_distribution


# In[10]:


fold_counts = (
    survival
    .groupby(["fold", "BRCA_Subtype_PAM50", "event"])
    .size()
    .reset_index(name="n_patients")
)

fold_totals = (
    survival
    .groupby("fold")
    .size()
    .reset_index(name="fold_total")
)

fold_proportions = fold_counts.merge(fold_totals, on="fold")
fold_proportions["proportion"] = (
    fold_proportions["n_patients"] / fold_proportions["fold_total"]
)

fold_proportions = fold_proportions.sort_values(
    ["fold", "BRCA_Subtype_PAM50", "event"]
)

fold_proportions


# In[12]:


cv_fold_assignments = survival[[
    "patient",
    "BRCA_Subtype_PAM50",
    "event",
    "time",
    "fold"
]].copy()

cv_fold_assignments.head()


# In[13]:


cv_fold_assignments.to_csv(
    DATA_DIR / "cv_fold_assignments.csv",
    index=False
)

fold_distribution.to_csv(
    TABLES_DIR / "cv_fold_distribution.csv",
    index=False
)

fold_proportions.to_csv(
    TABLES_DIR / "cv_fold_proportions.csv",
    index=False
)


# In[ ]:




