#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("../data/processed")
RESULTS_DIR = Path("../results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# In[ ]:


# Load CSV files

rna = pd.read_csv(DATA_DIR / "rna_pam50.csv")
meth = pd.read_csv(DATA_DIR / "meth_pam50.csv")
clinical = pd.read_csv(DATA_DIR / "clinical_luminal_brca.csv")
labels = pd.read_csv(DATA_DIR / "labels_luminal_brca.csv")


# In[ ]:


# check everything loaded correctly
print("RNA:", rna.shape)
print("Methylation:", meth.shape)
print("Clinical:", clinical.shape)
print("Labels:", labels.shape)

rna.head()


# In[ ]:


# verify that the patient IDs are aligned across all files
assert rna["patient"].equals(meth["patient"])
assert rna["patient"].equals(labels["patient"])
assert rna["patient"].equals(clinical["patient"])

print("All files are aligned.")


# In[ ]:


# check subtype counts
labels["subtype"].value_counts()


# In[8]:


ax = labels["subtype"].value_counts().loc[["LumA", "LumB"]].plot(kind="bar")

ax.set_title("Distribution of PAM50 Luminal Subtypes")
ax.set_xlabel("Subtype")
ax.set_ylabel("Number of patients")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "subtype_distribution.png", dpi=300)
plt.show()


# In[ ]:


# check rna and methylation feature counts
rna_features = [col for col in rna.columns if col != "patient"]
meth_features = [col for col in meth.columns if col != "patient"]

print("RNA features:", len(rna_features))
print("Methylation CpG features:", len(meth_features))


# In[ ]:


# check missingness
print("RNA missing fraction:", rna[rna_features].isna().mean().mean())
print("Methylation missing fraction:", meth[meth_features].isna().mean().mean())


# In[11]:


missing_per_patient = meth[meth_features].isna().mean(axis=1)

ax = missing_per_patient.plot(kind="hist", bins=30)

ax.set_title("Missing Methylation Fraction per Patient")
ax.set_xlabel("Fraction of promoter CpG values missing")
ax.set_ylabel("Number of patients")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "methylation_missingness_per_patient.png", dpi=300)
plt.show()


# In[12]:


missing_per_cpg = meth[meth_features].isna().mean(axis=0)

ax = missing_per_cpg.plot(kind="hist", bins=30)

ax.set_title("Missing Methylation Fraction per Promoter CpG")
ax.set_xlabel("Fraction of patients missing")
ax.set_ylabel("Number of CpGs")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "methylation_missingness_per_cpg.png", dpi=300)
plt.show()


# In[ ]:


# create survival status column
clinical.columns.tolist()


# In[14]:


[col for col in clinical.columns if any(
    word in col.lower()
    for word in ["survival", "vital", "death", "follow", "days"]
)]


# In[15]:


clinical[[
    "patient",
    "BRCA_Subtype_PAM50",
    "vital_status",
    "days_to_death",
    "days_to_last_follow_up"
]].head()


# In[ ]:


# create a survival dataframe with patient ID, subtype, event status, and time
survival = clinical[[
    "patient",
    "BRCA_Subtype_PAM50",
    "vital_status",
    "days_to_death",
    "days_to_last_follow_up"
]].copy()

survival["event"] = np.where(survival["vital_status"] == "Dead", 1, 0)

survival["days_to_death"] = pd.to_numeric(
    survival["days_to_death"],
    errors="coerce"
)

survival["days_to_last_follow_up"] = pd.to_numeric(
    survival["days_to_last_follow_up"],
    errors="coerce"
)

survival["time"] = np.where(
    survival["event"] == 1,
    survival["days_to_death"],
    survival["days_to_last_follow_up"]
)

survival.head()


# In[ ]:


# clean survival data by dropping rows with missing or invalid time/event values
survival_clean = survival.dropna(subset=["time", "event"]).copy()
survival_clean = survival_clean[survival_clean["time"] > 0].copy()

print("Original patients:", len(survival))
print("Patients with valid survival data:", len(survival_clean))
print("Events:")
print(survival_clean["event"].value_counts())


# In[ ]:


# save cleaned survival data for future use
survival_clean.to_csv(
    DATA_DIR / "survival_luminal_clean.csv",
    index=False
)


# In[ ]:


event_table = pd.crosstab(
    survival_clean["BRCA_Subtype_PAM50"],
    survival_clean["event"],
    normalize="index"
)

ax = event_table.plot(kind="bar", stacked=True)

ax.set_title("Overall Survival Event Proportion by Luminal Subtype")
ax.set_xlabel("PAM50 subtype")
ax.set_ylabel("Proportion of patients")
ax.legend(title="Event", labels=["Censored/alive", "Death/event"])

plt.tight_layout()
plt.savefig(FIGURES_DIR / "survival_event_by_subtype.png", dpi=300)
plt.show()


# In[ ]:


# create a summary table of key dataset characteristics
summary = pd.DataFrame({
    "metric": [
        "Patients with matched omics and clinical data",
        "Luminal A patients",
        "Luminal B patients",
        "PAM50 RNA features retained",
        "Promoter CpG methylation features retained",
        "Overall RNA missingness",
        "Overall methylation missingness",
        "Patients with valid survival outcome",
        "Deaths/events"
    ],
    "value": [
        len(labels),
        int((labels["subtype"] == "LumA").sum()),
        int((labels["subtype"] == "LumB").sum()),
        len(rna_features),
        len(meth_features),
        rna[rna_features].isna().mean().mean(),
        meth[meth_features].isna().mean().mean(),
        len(survival_clean),
        int(survival_clean["event"].sum())
    ]
})

summary 


# In[ ]:


# save summary table
summary.to_csv(
    TABLES_DIR / "week1_data_summary.csv",
    index=False
)


# In[ ]:




