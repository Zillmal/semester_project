#!/usr/bin/env python
# coding: utf-8

# 09b_Subtype_Correlation_Comparison.py
# Task 1 (pitch method 3): compare the promoter methylation-expression
# relationship between Luminal A and Luminal B. For each PAM50 gene the mean
# Spearman correlation is computed separately within each subtype, to see whether
# methylation-associated gene regulation differs between the two subtypes.
# Run from inside scripts/ (relative paths, like the other Python scripts).

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

Path("../results/figures").mkdir(parents=True, exist_ok=True)
Path("../results/tables").mkdir(parents=True, exist_ok=True)

# --- Load and align ----------------------------------------------------------
meth = pd.read_csv("../data/processed/meth_pam50_knn_imputed.csv", index_col=0)
rna = pd.read_csv("../data/processed/rna_pam50.csv").set_index("patient")
cpg_gene = pd.read_csv("../data/processed/cpg_gene_map.csv")
labels = pd.read_csv("../data/processed/labels_luminal_brca.csv").set_index("patient")

patients = meth.index.intersection(rna.index).intersection(labels.index)
meth, rna, labels = meth.loc[patients], rna.loc[patients], labels.loc[patients]
subtype = labels["subtype"]

pairs = cpg_gene[cpg_gene["cpg"].isin(meth.columns) & cpg_gene["gene"].isin(rna.columns)]


def per_gene_mean_rho(ids):
    """Mean Spearman rho (methylation vs expression) per gene within a patient set."""
    m, r = meth.loc[ids], rna.loc[ids]
    recs = [(gene, spearmanr(m[cpg], r[gene])[0])
            for cpg, gene in pairs[["cpg", "gene"]].itertuples(index=False)]
    return pd.DataFrame(recs, columns=["gene", "rho"]).groupby("gene")["rho"].mean()


luma_ids = subtype.index[subtype == "LumA"]
lumb_ids = subtype.index[subtype == "LumB"]
print(f"LumA n={len(luma_ids)} | LumB n={len(lumb_ids)}")

comp = pd.DataFrame({"rho_LumA": per_gene_mean_rho(luma_ids),
                     "rho_LumB": per_gene_mean_rho(lumb_ids)})
comp["difference"] = comp["rho_LumB"] - comp["rho_LumA"]   # +ve: weaker silencing in LumB
comp = comp.dropna().sort_values("rho_LumA")
comp.index.name = "gene"
comp.to_csv("../results/tables/correlation_by_subtype.csv")

print("\nGenes differing most between subtypes:")
print(comp.reindex(comp["difference"].abs().sort_values(ascending=False).index).head(8).round(3))

# --- Simple comparison plot: per-gene rho in LumA vs LumB ---------------------
# Points on the dashed y = x line behave identically in both subtypes;
# points off the line indicate subtype-specific methylation-expression coupling.
lims = [comp[["rho_LumA", "rho_LumB"]].min().min() - 0.05,
        comp[["rho_LumA", "rho_LumB"]].max().max() + 0.05]

plt.figure(figsize=(6.5, 6.5))
plt.plot(lims, lims, "--", color="grey", linewidth=1)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.scatter(comp["rho_LumA"], comp["rho_LumB"], s=28, alpha=0.7, edgecolor="black")
for gene in comp["difference"].abs().sort_values(ascending=False).head(6).index:
    plt.annotate(gene, (comp.loc[gene, "rho_LumA"], comp.loc[gene, "rho_LumB"]), fontsize=8)
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Mean Spearman rho (Luminal A)")
plt.ylabel("Mean Spearman rho (Luminal B)")
plt.title("Methylation-Expression Correlation per PAM50 Gene: LumA vs LumB")
plt.tight_layout()
plt.savefig("../results/figures/methylation_expression_correlation_LumA_vs_LumB.png", dpi=300)
plt.close()
print("\nSaved correlation_by_subtype.csv and methylation_expression_correlation_LumA_vs_LumB.png")
