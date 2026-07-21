import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from pathlib import Path



# load data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

mRNA = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "lasso_cox_cv_results.csv")
multi = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "lasso_cox_multiomics_cv_results.csv")

mRNA["model"] = "mRNA-only"
multi["model"] = "Multi-omics"



# Wilcoxon Rank Test of C-Indexes

wilcoxon(mRNA["test_c_index"], multi["test_c_index"])



# Create Combined Data Frame

combined_df = pd.concat([mRNA, multi], ignore_index=True)


# Create short descriptive summary

summary = combined_df.groupby("model")["test_c_index"].agg(["mean", "std"]).reset_index()
print(summary)


# Visualize the C-Indexes


plt.figure(figsize=(6,4))
sns.boxplot(data=combined_df, x="model", y="test_c_index")
sns.stripplot(data=combined_df, x="model", y="test_c_index", color="black", alpha=0.6)

plt.title("C-index comparison (5-fold CV)")
plt.ylim(0.3, 0.8)
plt.tight_layout()

plt.savefig(PROJECT_ROOT / "results" / "figures" / "cindex_boxplot.png", dpi=300)
plt.close()


# Visualize the Mean and SDs

# Sort the factor levels of model for better comparability

summary["model"] = pd.Categorical(
    summary["model"],
    categories=["mRNA-only", "Multi-omics"],
    ordered=True
)

summary = summary.sort_values("model")


plt.figure(figsize=(6,4))

plt.errorbar(
    summary["model"],
    summary["mean"],
    yerr=summary["std"],
    fmt="o",
    capsize=5
)

plt.title("Mean C-index ± SD")
plt.ylim(0.3, 0.8)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "figures" / "cindex_mean_sd.png", dpi=300)
plt.close()


# Visualize Feature Selection Stability

plt.figure(figsize=(6,4))

sns.boxplot(data=combined_df, x="model", y="n_features_selected")
sns.stripplot(data=combined_df, x="model", y="n_features_selected",
              color="black", alpha=0.6, jitter=True)

plt.title("Feature selection stability")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "figures" / "feature_selection_comparison.png", dpi=300)
plt.close()
