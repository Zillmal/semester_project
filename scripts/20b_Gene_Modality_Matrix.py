# Gene × modality matrix (one score per gene)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TABLES = Path("../results/tables")
FIGURES = Path("../results/figures"); FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = {
    "LASSO_mRNA":       "feature_importance_lasso_mrna.csv",
    "LASSO_multiomics": "feature_importance_lasso_multiomics.csv",
    "NN_mRNA":          "feature_importance_nn_mrna.csv",
    "NN_integrated":    "feature_importance_nn_integrated.csv",
}
tables = {m: pd.read_csv(TABLES / f) for m, f in MODELS.items()}
ALL_GENES = sorted(set().union(*[set(df["gene"]) for df in tables.values()]))
print("Genes:", len(ALL_GENES))



# Build the 8 Columns

def modality_score(df, modality):
    sub = df[df["modality"] == modality]
    if sub.empty:
        return None                                   # e.g. methylation in an mRNA-only model
    g = sub.groupby("gene")["importance"].max().reindex(ALL_GENES).fillna(0.0)
    score = g.rank(pct=True)                           # rank among the 50 genes -> 0..1
    score[g <= 0] = 0.0                                # a gene never used scores 0
    return score

cols = {}
for m, df in tables.items():
    for mod, label in [("RNA", "expression"), ("METH", "methylation")]:
        s = modality_score(df, mod)
        cols[f"{m}__{label}"] = s if s is not None else pd.Series(np.nan, index=ALL_GENES)

matrix = pd.DataFrame(cols, index=ALL_GENES)
matrix.round(2).head()


# Modality means, combined score, rankings

expr_cols = [c for c in matrix.columns if c.endswith("expression")]
meth_cols = [c for c in matrix.columns if c.endswith("methylation")]

matrix["expression_mean"] = matrix[expr_cols].mean(axis=1)     # over 4 models
matrix["methylation_mean"] = matrix[meth_cols].mean(axis=1)    # over 2 models (empty cells ignored)
matrix["combined"] = matrix["expression_mean"] + matrix["methylation_mean"]

matrix = matrix.sort_values("combined", ascending=False)
matrix["rank"] = np.arange(1, len(matrix) + 1)

matrix.to_csv(TABLES / "gene_modality_matrix.csv")
matrix[["expression_mean", "methylation_mean", "combined", "rank"]].head(15).round(2)


# Figure: ## Figure: top 15 genes, split into the two modality contributions


d = matrix.head(50).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(d.index, d["expression_mean"], color="#4C72B0", label="expression (mean of 4 models)")
plt.barh(d.index, d["methylation_mean"], left=d["expression_mean"],
         color="#C44E52", label="methylation (mean of 2 models)")
plt.xlabel("combined score  (expression_mean + methylation_mean)")
plt.title("Top genes by combined importance")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIGURES / "gene_modality_matrix_top15.png", dpi=300)
#plt.show()