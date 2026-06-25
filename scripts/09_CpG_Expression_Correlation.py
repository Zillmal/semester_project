#!/usr/bin/env python
# coding: utf-8

# 09_CpG_Expression_Correlation.py
# Task 1 (part 1): relationship between promoter methylation and gene expression.
# For each PAM50 gene, correlate (Spearman) its promoter CpGs (KNN-imputed beta)
# against its log2 mRNA. Negative correlation = epigenetic silencing signal.
# Run from inside scripts/ (paths are relative, like the other Python scripts).

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

Path("../results/figures").mkdir(parents=True, exist_ok=True)
Path("../results/tables").mkdir(parents=True, exist_ok=True)

# --- Load data ---------------------------------------------------------------
# Imputed methylation is used here for a complete (no-NaN) descriptive analysis;
# this is EDA, not cross-validated modelling, so global imputation is fine.
meth = pd.read_csv("../data/processed/meth_pam50_knn_imputed.csv", index_col=0)
rna = pd.read_csv("../data/processed/rna_pam50.csv").set_index("patient")
cpg_gene = pd.read_csv("../data/processed/cpg_gene_map.csv")

assert meth.isna().sum().sum() == 0, "Imputed methylation still contains NaNs."

# --- Align patients across both omics layers ---------------------------------
patients = meth.index.intersection(rna.index)
meth = meth.loc[patients]
rna = rna.loc[patients]
print(f"Patients used: {len(patients)} | CpGs: {meth.shape[1]} | genes: {rna.shape[1]}")

# Keep only mappable pairs whose CpG and gene are both present in the data.
cpg_gene = cpg_gene[cpg_gene["cpg"].isin(meth.columns) & cpg_gene["gene"].isin(rna.columns)]
print(f"CpG-gene pairs tested: {len(cpg_gene)}")


def bh_qvalues(pvals):
    """Benjamini-Hochberg FDR correction (no statsmodels dependency)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# --- Per-CpG Spearman correlation --------------------------------------------
rows = []
for _, (cpg, gene) in cpg_gene[["cpg", "gene"]].iterrows():
    rho, pval = spearmanr(meth[cpg], rna[gene])
    rows.append({"cpg": cpg, "gene": gene, "spearman_rho": rho, "p_value": pval})

cpg_corr = pd.DataFrame(rows)
cpg_corr["q_value"] = bh_qvalues(cpg_corr["p_value"])
cpg_corr = cpg_corr.sort_values("spearman_rho")  # most negative (silencing) first
cpg_corr.to_csv("../results/tables/cpg_expression_spearman.csv", index=False)

# --- Per-gene summary --------------------------------------------------------
# Per gene: mean rho over its promoter CpGs and its single most-negative CpG.
grp = cpg_corr.groupby("gene")
gene_corr = pd.DataFrame({
    "n_cpgs": grp.size(),
    "mean_rho": grp["spearman_rho"].mean(),
    "min_rho": grp["spearman_rho"].min(),
})
# Row index (per gene) of the most-negative CpG, used to pull its id and q-value.
min_idx = grp["spearman_rho"].idxmin().reindex(gene_corr.index)
gene_corr["min_rho_cpg"] = cpg_corr.loc[min_idx, "cpg"].values
gene_corr["min_rho_qvalue"] = cpg_corr.loc[min_idx, "q_value"].values
gene_corr = gene_corr.sort_values("mean_rho")
gene_corr.index.name = "gene"
gene_corr.to_csv("../results/tables/gene_methylation_expression_correlation_summary.csv")

# Gene with the strongest silencing signal (most negative mean rho); reused by
# the survival script (stage 10) to define methylation strata.
top_gene = gene_corr.index[0]
top_cpg = gene_corr.loc[top_gene, "min_rho_cpg"]
print(f"Strongest silencing gene: {top_gene} "
      f"(mean rho={gene_corr.loc[top_gene, 'mean_rho']:.2f}, "
      f"min rho={gene_corr.loc[top_gene, 'min_rho']:.2f} at {top_cpg})")

# --- Figure 1: per-gene mean Spearman rho ------------------------------------
colors = ["#D85A30" if r < 0 else "#378ADD" for r in gene_corr["mean_rho"]]
plt.figure(figsize=(8, 11))
plt.barh(gene_corr.index, gene_corr["mean_rho"], color=colors, edgecolor="black")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mean Spearman rho (promoter methylation vs mRNA)")
plt.title("Promoter Methylation - Expression Correlation per PAM50 Gene")
plt.tight_layout()
plt.savefig("../results/figures/methylation_expression_correlation_by_gene.png", dpi=300)
plt.close()

# --- Figure 2: scatter for the strongest-silencing CpG -----------------------
rho, _ = spearmanr(meth[top_cpg], rna[top_gene])
plt.figure(figsize=(6, 5))
plt.scatter(meth[top_cpg], rna[top_gene], s=14, alpha=0.5, edgecolor="none")
plt.xlabel(f"Promoter methylation beta ({top_cpg})")
plt.ylabel(f"log2 normalized expression ({top_gene})")
plt.title(f"{top_gene}: methylation vs expression (Spearman rho={rho:.2f})")
plt.tight_layout()
plt.savefig("../results/figures/top_silencing_gene_scatter.png", dpi=300)
plt.close()

print("Saved correlation tables and figures to ../results/.")
