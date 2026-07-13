#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
import umap

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

Path("../results/figures").mkdir(parents=True, exist_ok=True)
Path("../results/tables").mkdir(parents=True, exist_ok=True)


# In[2]:

rna      = pd.read_csv("../data/processed/rna_pam50.csv").set_index("patient")
meth     = pd.read_csv("../data/processed/meth_pam50_knn_imputed.csv", index_col=0)
meth_raw = pd.read_csv("../data/processed/meth_pam50.csv").set_index("patient")
labels   = pd.read_csv("../data/processed/labels_luminal_brca.csv").set_index("patient")
cpg_gene = pd.read_csv("../data/processed/cpg_gene_map.csv")
surv     = pd.read_csv("../data/processed/survival_luminal_clean.csv").set_index("patient")

patients = rna.index.intersection(meth.index).intersection(labels.index)
rna, meth, labels = rna.loc[patients], meth.loc[patients], labels.loc[patients]
subtype  = labels["subtype"]
meth_clean = meth.dropna(axis=1, how="all")
meth_scaled = StandardScaler().fit_transform(meth_clean)

COLORS = {"LumA": "#0e5f60", "LumB": "#9c224d"}
print(f"Patients: {len(patients)} | LumA: {(subtype=='LumA').sum()} | LumB: {(subtype=='LumB').sum()}")
print(f"RNA genes: {rna.shape[1]} | CpGs (imputed): {meth_clean.shape[1]}")


# In[3]:

surv_common = surv.loc[surv.index.intersection(patients)]
surv_common = surv_common[surv_common["time"].notna() & (surv_common["time"] > 0)].copy()
surv_common["time_years"] = surv_common["time"] / 365.25

print("Cohort summary:")
print(f"  Total patients:     {len(patients)}")
print(f"  LumA:               {(subtype=='LumA').sum()} ({(subtype=='LumA').mean()*100:.1f}%)")
print(f"  LumB:               {(subtype=='LumB').sum()} ({(subtype=='LumB').mean()*100:.1f}%)")
print(f"  Patients w/ valid survival: {len(surv_common)}")
print(f"  Deaths (events):    {int(surv_common['event'].sum())} ({surv_common['event'].mean()*100:.1f}%)")
print(f"  Median follow-up:   {surv_common['time_years'].median():.1f} years")

# subtype distribution plot
fig, ax = plt.subplots(figsize=(5, 4))
counts = subtype.value_counts().loc[["LumA", "LumB"]]
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[s] for s in counts.index], width=0.5, edgecolor="white")
for bar, (s, n) in zip(bars, counts.items()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
            f"{n}\n({n/len(patients)*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")
ax.set_title("PAM50 Luminal Subtype Distribution", fontsize=12)
ax.set_ylabel("Number of patients")
ax.set_ylim(0, counts.max() * 1.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/subtype_distribution.png", dpi=300)
plt.show()


# In[4]:

missing_cols = meth_raw.isna().any(axis=0).sum()
missing_frac_per_cpg = meth_raw.isna().mean(axis=0)
missing_frac_per_pat = meth_raw.isna().mean(axis=1)

print(f"RNA missing values: {rna.isna().sum().sum()}")
print(f"Methylation CpGs with any missing: {missing_cols} / {meth_raw.shape[1]}")
print(f"Methylation: mean missing fraction per CpG:     {missing_frac_per_cpg.mean():.3f}")
print(f"Methylation: mean missing fraction per patient: {missing_frac_per_pat.mean():.3f}")


# In[5]:

def beta_to_m(B):
    B = np.clip(B, 1e-4, 1 - 1e-4)
    return np.log2(B / (1 - B))

BG = "#fcfcfc"
TEXT_SIZE = 19
COL_BETA = "#9c224d"
COL_M    = "#0e5f60"

all_beta = meth_clean.values.ravel()
all_beta = all_beta[~np.isnan(all_beta)]
all_m = beta_to_m(all_beta)

high_var_cpgs = meth_clean.columns[meth_clean.var(axis=0) >= 0.0005]
low_var_cpgs  = meth_clean.columns[meth_clean.var(axis=0) <  0.0005]
all_beta_high = meth_clean[high_var_cpgs].values.ravel()
all_beta_high = all_beta_high[~np.isnan(all_beta_high)]
all_m_high    = beta_to_m(all_beta_high)

n_total = meth_clean.shape[1]
titles = [
    "All CpGs",
    f"Flat CpGs removed (var < 0.0005)\nn={len(low_var_cpgs)} ({len(low_var_cpgs)/n_total*100:.1f}%)",
    f"Retained CpGs (var ≥ 0.0005)\nn={len(high_var_cpgs)} ({len(high_var_cpgs)/n_total*100:.1f}%)"
]

fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey="row", facecolor=BG)
fig.patch.set_facecolor(BG)

for col, (vals_b, vals_m, title) in enumerate([
    (all_beta, all_m, titles[0]),
    (meth_clean[low_var_cpgs].values.ravel(),
     beta_to_m(meth_clean[low_var_cpgs].values.ravel()),
     titles[1]),
    (all_beta_high, all_m_high, titles[2])
]):
    vb = vals_b[~np.isnan(vals_b)]
    vm = vals_m[~np.isnan(vals_m)]

    for row_idx, (vals, color, xlabel) in enumerate([
        (vb, COL_BETA, "Beta value"),
        (vm, COL_M,    "M-value")
    ]):
        ax = axes[row_idx, col]
        ax.set_facecolor(BG)
        ax.hist(vals, bins=60, color=color, edgecolor="none", density=True, alpha=0.85)
        ax.set_title(title, fontsize=TEXT_SIZE * 0.75, pad=10, color="#263238")
        ax.set_xlabel(xlabel, fontsize=TEXT_SIZE * 0.75)
        ax.set_ylabel("Density", fontsize=TEXT_SIZE * 0.75)
        ax.tick_params(axis="both", labelsize=TEXT_SIZE * 0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1)

fig.suptitle(
    "PAM50 Promoter Methylation — Beta and M-value Distributions",
    fontsize=TEXT_SIZE,
    color="#9c224d",
    fontweight="bold",
    y=1.01
)
plt.tight_layout()
plt.savefig("../results/figures/beta_vs_mvalue_distribution.png", dpi=300,
            facecolor=BG, bbox_inches="tight")
plt.show()


# In[6]:

scaler = StandardScaler()
rna_scaled = scaler.fit_transform(rna)
pca_rna = PCA(n_components=5)
rna_pcs = pca_rna.fit_transform(rna_scaled)
var_rna = pca_rna.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (px, py) in zip(axes, [(0,1),(1,2)]):
    for st in ["LumA", "LumB"]:
        mask = (subtype == st).values
        ax.scatter(rna_pcs[mask, px], rna_pcs[mask, py], c=COLORS[st], label=st, s=18, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"PC{px+1} ({var_rna[px]:.1f}%)")
    ax.set_ylabel(f"PC{py+1} ({var_rna[py]:.1f}%)")
    ax.set_title(f"RNA PCA: PC{px+1} vs PC{py+1}")
    ax.legend(title="Subtype"); ax.spines[["top","right"]].set_visible(False)
fig.suptitle("PCA of PAM50 RNA Expression (50 genes)", fontsize=12)
plt.tight_layout()
plt.savefig("../results/figures/rna_pca_by_subtype.png", dpi=300)
plt.show()

reducer_rna = umap.UMAP(n_components=2, random_state=42)
rna_umap = reducer_rna.fit_transform(rna_scaled)
fig, ax = plt.subplots(figsize=(7, 5))
for st in ["LumA", "LumB"]:
    mask = (subtype == st).values
    ax.scatter(rna_umap[mask, 0], rna_umap[mask, 1], c=COLORS[st], label=st, s=18, alpha=0.6, edgecolors="none")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("UMAP of PAM50 RNA Expression (50 genes)")
ax.legend(title="Subtype"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/rna_umap_by_subtype.png", dpi=300)
plt.show()


# In[7]:

KEY_GENES = ["ESR1", "PGR", "MKI67", "BIRC5", "ERBB2", "GRB7", "KRT5", "KRT14"]
available = [g for g in KEY_GENES if g in rna.columns]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
for ax, gene in zip(axes, available):
    luma = rna.loc[subtype == "LumA", gene].dropna()
    lumb = rna.loc[subtype == "LumB", gene].dropna()
    parts = ax.violinplot([luma, lumb], positions=[0,1], showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], [COLORS["LumA"], COLORS["LumB"]]):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
    rng = np.random.default_rng(42)
    for pos, vals, color in zip([0,1],[luma,lumb],[COLORS["LumA"],COLORS["LumB"]]):
        idx = rng.choice(len(vals), size=min(80,len(vals)), replace=False)
        ax.scatter(pos + rng.uniform(-0.07,0.07,size=len(idx)), vals.iloc[idx], s=5, alpha=0.4, color=color)
    _, pval = stats.mannwhitneyu(luma, lumb, alternative="two-sided")
    sig = "***" if pval<0.001 else "**" if pval<0.01 else "*" if pval<0.05 else "ns"
    ax.set_title(f"{gene}  {sig}", fontsize=11)
    ax.set_xticks([0,1]); ax.set_xticklabels(["LumA","LumB"]); ax.set_ylabel("log2 expr.")
    ax.spines[["top","right"]].set_visible(False)
for ax in axes[len(available):]: ax.set_visible(False)
fig.suptitle("PAM50 Gene Expression by Luminal Subtype\n(Mann-Whitney U: * p<0.05, ** p<0.01, *** p<0.001)", fontsize=12)
plt.tight_layout()
plt.savefig("../results/figures/rna_violin_by_subtype.png", dpi=300)
plt.show()


# In[8]:

pca_meth = PCA(n_components=5, random_state=42)
meth_pcs = pca_meth.fit_transform(meth_scaled)
var_meth = pca_meth.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (px, py) in zip(axes, [(0,1),(1,2)]):
    for st in ["LumA","LumB"]:
        mask = (subtype==st).values
        ax.scatter(meth_pcs[mask,px], meth_pcs[mask,py], c=COLORS[st], label=st, s=18, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"PC{px+1} ({var_meth[px]:.1f}%)")
    ax.set_ylabel(f"PC{py+1} ({var_meth[py]:.1f}%)")
    ax.set_title(f"Methylation PCA: PC{px+1} vs PC{py+1}")
    ax.legend(title="Subtype"); ax.spines[["top","right"]].set_visible(False)
fig.suptitle("PCA of PAM50 Promoter Methylation (post-imputation)", fontsize=12)
plt.tight_layout()
plt.savefig("../results/figures/meth_pca_by_subtype.png", dpi=300)
plt.show()

reducer_meth = umap.UMAP(n_components=2, random_state=42)
meth_umap = reducer_meth.fit_transform(meth_scaled)

fig, ax = plt.subplots(figsize=(7, 5))
for st in ["LumA", "LumB"]:
    mask = (subtype == st).values
    ax.scatter(meth_umap[mask, 0], meth_umap[mask, 1],
               c=COLORS[st], label=st, s=18, alpha=0.6, edgecolors="none")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP of PAM50 Promoter Methylation (post-imputation)")
ax.legend(title="Subtype")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/meth_umap_by_subtype.png", dpi=300)
plt.show()


# In[9]:

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
for ax, gene in zip(axes, available):
    cpg_ids = [c for c in cpg_gene.loc[cpg_gene["gene"]==gene,"cpg"].tolist() if c in meth_clean.columns]
    if not cpg_ids: ax.set_visible(False); continue
    mean_beta = meth_clean[cpg_ids].mean(axis=1)
    luma = mean_beta.loc[subtype.index[subtype=="LumA"]].dropna()
    lumb = mean_beta.loc[subtype.index[subtype=="LumB"]].dropna()
    parts = ax.violinplot([luma,lumb], positions=[0,1], showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"],[COLORS["LumA"],COLORS["LumB"]]):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
    rng = np.random.default_rng(42)
    for pos, vals, color in zip([0,1],[luma,lumb],[COLORS["LumA"],COLORS["LumB"]]):
        idx = rng.choice(len(vals), size=min(80,len(vals)), replace=False)
        ax.scatter(pos+rng.uniform(-0.07,0.07,size=len(idx)), vals.iloc[idx], s=5, alpha=0.4, color=color)
    _, pval = stats.mannwhitneyu(luma, lumb, alternative="two-sided")
    sig = "***" if pval<0.001 else "**" if pval<0.01 else "*" if pval<0.05 else "ns"
    ax.set_title(f"{gene}  {sig}", fontsize=11)
    ax.set_xticks([0,1]); ax.set_xticklabels(["LumA","LumB"]); ax.set_ylabel("mean promoter beta")
    ax.spines[["top","right"]].set_visible(False)
for ax in axes[len(available):]: ax.set_visible(False)
fig.suptitle("PAM50 Promoter Methylation by Luminal Subtype\n(Mann-Whitney U: * p<0.05, ** p<0.01, *** p<0.001)", fontsize=12)
plt.tight_layout()
plt.savefig("../results/figures/meth_violin_by_subtype.png", dpi=300)
plt.show()


# In[10]:

SCATTER_GENES = ["ESR1","PGR","ERBB2","FOXA1","MKI67","KRT5","BCL2","MLPH"]
available_scatter = [g for g in SCATTER_GENES if g in rna.columns]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
n_plotted = 0
for ax, gene in zip(axes, available_scatter):
    cpg_ids = [c for c in cpg_gene.loc[cpg_gene["gene"]==gene,"cpg"].tolist() if c in meth_clean.columns]
    if not cpg_ids: ax.set_visible(False); continue
    mean_beta = meth_clean[cpg_ids].mean(axis=1)
    both = pd.DataFrame({"beta": mean_beta, "rna": rna[gene]}).dropna()
    rho, _ = stats.spearmanr(both["beta"], both["rna"])
    for st in ["LumA","LumB"]:
        mask = (subtype.loc[both.index]==st).values
        ax.scatter(both.loc[mask,"beta"], both.loc[mask,"rna"], c=COLORS[st], s=8, alpha=0.45, label=st, edgecolors="none")
    ax.set_xlabel("Mean promoter beta"); ax.set_ylabel("log2 expr.")
    ax.set_title(f"{gene}  rho={rho:.2f}"); ax.spines[["top","right"]].set_visible(False)
    n_plotted += 1
for ax in axes[n_plotted:]: ax.set_visible(False)
handles = [plt.scatter([],[],c=COLORS[s],label=s) for s in ["LumA","LumB"]]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Promoter Methylation vs Gene Expression (Spearman rho)\nnegative rho = epigenetic silencing", fontsize=12)
plt.tight_layout()
plt.savefig("../results/figures/meth_vs_rna_scatter_panel.png", dpi=300, bbox_inches="tight")
plt.show()


# In[11]:

pam50_axes = {
    "ESR1":"Hormone","PGR":"Hormone","FOXA1":"Hormone","MLPH":"Hormone","NAT1":"Hormone",
    "BAG1":"Hormone","BLVRA":"Hormone","SFRP1":"Hormone","BCL2":"Hormone","CXXC5":"Hormone",
    "SLC39A6":"Hormone","GPR160":"Hormone","ERBB2":"HER2","GRB7":"HER2","FGFR4":"HER2",
    "MKI67":"Prolif.","BIRC5":"Prolif.","CCNB1":"Prolif.","CCNE1":"Prolif.","CDC20":"Prolif.",
    "CDC6":"Prolif.","ANLN":"Prolif.","MELK":"Prolif.","MYBL2":"Prolif.","MMP11":"Prolif.",
    "UBE2C":"Prolif.","UBE2T":"Prolif.","PTTG1":"Prolif.","EXO1":"Prolif.","NUF2":"Prolif.",
    "NDC80":"Prolif.","KIF2C":"Prolif.","CENPF":"Prolif.","CEP55":"Prolif.","TYMS":"Prolif.",
    "RRM2":"Prolif.","ORC6":"Prolif.","MDM2":"Prolif.","MYC":"Prolif.",
    "KRT5":"Basal","KRT14":"Basal","KRT17":"Basal","FOXC1":"Basal","MIA":"Basal",
    "PHGDH":"Basal","EGFR":"Basal","CDH3":"Basal","ACTR3B":"Basal","MAPT":"Other","TMEM45B":"Other",
}

expr_rows = []
for gene in rna.columns:
    luma = rna.loc[subtype=="LumA",gene].dropna()
    lumb = rna.loc[subtype=="LumB",gene].dropna()
    _, pval = stats.mannwhitneyu(luma, lumb, alternative="two-sided")
    expr_rows.append({"gene":gene,"expr_median_LumA":round(luma.median(),3),
                      "expr_median_LumB":round(lumb.median(),3),
                      "expr_abs_diff":round(abs(luma.median()-lumb.median()),3),"expr_pval":round(pval,4)})
expr_df = pd.DataFrame(expr_rows)
expr_df["expr_qval"] = multipletests(expr_df["expr_pval"], method="fdr_bh")[1].round(4)
expr_df["expr_sig"] = expr_df["expr_qval"].apply(lambda q: "***" if q<0.001 else "**" if q<0.01 else "*" if q<0.05 else "ns")

meth_rows = []
for gene in rna.columns:
    cpg_ids = [c for c in cpg_gene.loc[cpg_gene["gene"]==gene,"cpg"].tolist() if c in meth_clean.columns]
    if not cpg_ids:
        meth_rows.append({"gene":gene,"n_cpgs":0,"meth_median_LumA":None,"meth_median_LumB":None,
                          "meth_abs_diff":None,"meth_pval":None,"cpg_sd":None}); continue
    mean_beta = meth_clean[cpg_ids].mean(axis=1)
    luma_m = mean_beta.loc[subtype.index[subtype=="LumA"]].dropna()
    lumb_m = mean_beta.loc[subtype.index[subtype=="LumB"]].dropna()
    _, pval = stats.mannwhitneyu(luma_m, lumb_m, alternative="two-sided")
    meth_rows.append({"gene":gene,"n_cpgs":len(cpg_ids),
                      "meth_median_LumA":round(luma_m.median(),3),"meth_median_LumB":round(lumb_m.median(),3),
                      "meth_abs_diff":round(abs(luma_m.median()-lumb_m.median()),3),"meth_pval":round(pval,4),
                      "cpg_sd":round(meth_clean[cpg_ids].mean(axis=0).std(),4)})
meth_df = pd.DataFrame(meth_rows)
valid_m = meth_df["meth_pval"].notna()
meth_df.loc[valid_m,"meth_qval"] = multipletests(meth_df.loc[valid_m,"meth_pval"],method="fdr_bh")[1].round(4)
meth_df["meth_sig"] = meth_df["meth_qval"].apply(lambda q: "***" if pd.notna(q) and q<0.001 else "**" if pd.notna(q) and q<0.01 else "*" if pd.notna(q) and q<0.05 else "ns" if pd.notna(q) else None)

corr_rows = []
for gene in rna.columns:
    cpg_ids = [c for c in cpg_gene.loc[cpg_gene["gene"]==gene,"cpg"].tolist() if c in meth_clean.columns]
    if not cpg_ids: corr_rows.append({"gene":gene,"spearman_rho":None,"corr_pval":None}); continue
    mean_beta = meth_clean[cpg_ids].mean(axis=1)
    both = pd.DataFrame({"beta":mean_beta,"rna":rna[gene]}).dropna()
    rho, pval = stats.spearmanr(both["beta"],both["rna"])
    corr_rows.append({"gene":gene,"spearman_rho":round(rho,3),"corr_pval":round(pval,4)})
corr_df = pd.DataFrame(corr_rows)
valid_c = corr_df["corr_pval"].notna()
corr_df.loc[valid_c,"corr_qval"] = multipletests(corr_df.loc[valid_c,"corr_pval"],method="fdr_bh")[1].round(4)
corr_df["corr_sig"] = corr_df["corr_qval"].apply(lambda q: "***" if pd.notna(q) and q<0.001 else "**" if pd.notna(q) and q<0.01 else "*" if pd.notna(q) and q<0.05 else "ns" if pd.notna(q) else None)

gene_table = expr_df.merge(meth_df,on="gene").merge(corr_df,on="gene").sort_values("gene").reset_index(drop=True)
gene_table.insert(1,"Axis",gene_table["gene"].map(pam50_axes).fillna("Other"))
gene_table.columns = ["Gene","Axis","Expr LumA","Expr LumB","Expr |diff|","Expr p","Expr q","Expr sig",
                       "N CpGs","Meth LumA","Meth LumB","Meth |diff|","Meth p","CpG SD","Meth q","Meth sig",
                       "Spearman rho","Corr p","Corr q","Corr sig"]
gene_table.to_csv("../results/tables/gene_summary_table.csv", index=False)
print(gene_table.to_string(index=False))


# In[12]:

silencing_genes = gene_table[
    (gene_table["Expr sig"] == "ns") &
    (gene_table["Meth sig"].isin(["*","**","***"])) &
    (gene_table["Corr sig"].isin(["*","**","***"])) &
    (gene_table["Spearman rho"] < 0)
].copy()

print(f"Genes with subtype-specific epigenetic silencing (Expr ns, Meth sig, rho<0): {len(silencing_genes)}")
print(silencing_genes[["Gene","Axis","Expr sig","Meth LumA","Meth LumB","Meth |diff|","Meth sig","Spearman rho","Corr sig"]].to_string(index=False))
silencing_genes.to_csv("../results/tables/subtype_specific_silencing_genes.csv", index=False)


# In[13]:

surv_common = surv.loc[surv.index.intersection(patients)]
surv_common = surv_common[surv_common["time"].notna() & (surv_common["time"]>0)].copy()
surv_common["time_years"] = surv_common["time"] / 365.25

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7, 5))
for st, color in [("LumA","#0e5f60"),("LumB","#9c224d")]:
    sub = surv_common[surv_common["BRCA_Subtype_PAM50"]==st]
    kmf.fit(sub["time_years"], sub["event"], label=f"{st} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)
lr = logrank_test(
    surv_common.loc[surv_common["BRCA_Subtype_PAM50"]=="LumA","time_years"],
    surv_common.loc[surv_common["BRCA_Subtype_PAM50"]=="LumB","time_years"],
    event_observed_A=surv_common.loc[surv_common["BRCA_Subtype_PAM50"]=="LumA","event"],
    event_observed_B=surv_common.loc[surv_common["BRCA_Subtype_PAM50"]=="LumB","event"])
ax.text(0.04,0.05,f"Log-rank p={lr.p_value:.4f}",transform=ax.transAxes,fontsize=11,style="italic")
ax.set_xlabel("Overall survival (years)"); ax.set_ylabel("Survival probability")
ax.set_title("LumA vs LumB Overall Survival"); ax.set_ylim(0,1.05)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/km_luma_vs_lumb_eda.png", dpi=300)
plt.show()


# In[14]:

death_summary = surv_common.groupby("BRCA_Subtype_PAM50").agg(
    n_patients=("event", "count"),
    n_deaths=("event", "sum"),
).reset_index()
death_summary["event_rate"] = (death_summary["n_deaths"] / death_summary["n_patients"] * 100).round(1)
death_summary.columns = ["Subtype", "N patients", "N deaths", "Event rate (%)"]
print(death_summary.to_string(index=False))


