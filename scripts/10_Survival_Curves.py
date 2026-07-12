#!/usr/bin/env python
# coding: utf-8

# 10_Survival_Curves.py
# Task 1 (part 2): Kaplan-Meier overall-survival visualizations.
#   (1) by PAM50 subtype (LumA vs LumB)
#   (2) by high/low promoter methylation of the strongest silencing gene (from 09)
# Run from inside scripts/ (relative paths, like the other Python scripts).

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

Path("../results/figures").mkdir(parents=True, exist_ok=True)
Path("../results/tables").mkdir(parents=True, exist_ok=True)

# --- Load --------------------------------------------------------------------
surv = pd.read_csv("../data/processed/survival_luminal_clean.csv").set_index("patient")
meth = pd.read_csv("../data/processed/meth_pam50_knn_imputed.csv", index_col=0)
cpg_gene = pd.read_csv("../data/processed/cpg_gene_map.csv")
gene_sum = pd.read_csv("../results/tables/gene_methylation_expression_correlation_summary.csv")

# Keep patients with a valid survival time; express time in years for readability.
surv = surv[surv["time"].notna() & (surv["time"] >= 0)].copy()
surv["time_years"] = surv["time"] / 365.25


def two_group_km(df, group_col, title, fname, order):
    TEXT_SIZE = 19
    """KM curves for two groups + log-rank test; returns a result row."""
    BLUE = "#0e5f60"
    PINK = "#9c224d"
    BG = "#fcfcfc"
    
    g0, g1 = order
    a, b = df[df[group_col] == g0], df[df[group_col] == g1]
    lr = logrank_test(a["time_years"], b["time_years"],
                      event_observed_A=a["event"], event_observed_B=b["event"])
    
    fig, ax = plt.subplots(
        figsize=(7, 5.5),
        facecolor=BG
    )
    ax.set_facecolor(BG)

    kmf = KaplanMeierFitter()

    group_settings = [
        (g0, a, "Luminal A", BLUE),
        (g1, b, "Luminal B", PINK)
    ]

    group_settings = [
        (g0, a, "Luminal A", BLUE),
        (g1, b, "Luminal B", PINK)
    ]

    for group, sub, display_name, color in group_settings:
        kmf.fit(
            durations=sub["time_years"],
            event_observed=sub["event"],
            label=f"{display_name} (n={len(sub)})"
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=color,
            linewidth=2.5,
            show_censors=False,
            ci_alpha=0.1
        )

    ax.set_title(
        title,
        fontsize=TEXT_SIZE,
        pad=12
    )

    ax.set_xlabel(
        "Time since diagnosis (years)",
        fontsize=TEXT_SIZE
    )

    ax.set_ylabel(
        "Survival probability",
        fontsize=TEXT_SIZE
    )

    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)

    # Log-rank p-value annotation
    if lr.p_value < 0.0001:
        p_text = r"Log-rank $p < 0.0001$"
    else:
        p_text = rf"Log-rank $p = {lr.p_value:.3g}$"

    ax.text(
        0.04,
        0.05,
        p_text,
        transform=ax.transAxes,
        fontsize=TEXT_SIZE*0.8,
        style="italic"
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=TEXT_SIZE*0.8,
        handlelength=1.5
    )

    # Minimal styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=TEXT_SIZE*0.8
    )

    plt.tight_layout()
    plt.savefig(f"../results/figures/{fname}", dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    return {"comparison": title, "group1": g0, "n1": len(a),
            "group2": g1, "n2": len(b),
            "logrank_chisq": lr.test_statistic, "p_value": lr.p_value}


results = []

# --- (1) Survival by PAM50 subtype -------------------------------------------
results.append(two_group_km(
    surv, "BRCA_Subtype_PAM50",
    "Overall survival by PAM50 subtype",
    "km_survival_by_subtype.png",
    order=["LumA", "LumB"]))

# --- (2) Survival by promoter methylation of the strongest silencing gene -----
# Gene chosen data-driven from stage 09 (most negative mean methylation-expression rho).
top_gene = gene_sum.sort_values("mean_rho").iloc[0]["gene"]
gene_cpgs = [c for c in cpg_gene.loc[cpg_gene["gene"] == top_gene, "cpg"] if c in meth.columns]
gene_meth = meth[gene_cpgs].mean(axis=1)  # mean promoter methylation per patient

common = surv.index.intersection(gene_meth.index)
d = surv.loc[common].copy()
median = gene_meth.loc[common].median()
# Median split into high vs low promoter methylation.
d["meth_group"] = np.where(gene_meth.loc[common] >= median,
                           "High methylation", "Low methylation")

results.append(two_group_km(
    d, "meth_group",
    f"Overall survival by {top_gene} promoter methylation",
    f"km_survival_by_{top_gene}_methylation.png",
    order=["Low methylation", "High methylation"]))

# --- Save log-rank results ---------------------------------------------------
res_df = pd.DataFrame(results)
res_df.to_csv("../results/tables/survival_logrank_tests.csv", index=False)
print(res_df.to_string(index=False))
print(f"\nMethylation strata based on gene: {top_gene} "
      f"({len(gene_cpgs)} promoter CpGs, median beta = {median:.3f})")
