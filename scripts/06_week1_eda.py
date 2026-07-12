#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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


# Colors
BG = "#fcfcfc"
BLUE = "#0e5f60"
PINK = "#9c224d"
TEXT_SIZE = 19

# Count samples
counts = labels["subtype"].value_counts().loc[["LumA", "LumB"]]
total = counts.sum()

fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=BG)
ax.set_facecolor(BG)

# Donut chart
wedges, _ = ax.pie(
    counts,
    startangle=90,
    colors=[BLUE, PINK],
    wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2)
)

# Center text
ax.text(
    0, 0,
    f"Total\n{total}\npatients",
    ha="center",
    va="center",
    fontsize=TEXT_SIZE
)

# Legend
legend_labels = [
    f"Luminal A\n{counts['LumA']} ({counts['LumA']/total*100:.0f}%)",
    f"Luminal B\n{counts['LumB']} ({counts['LumB']/total*100:.0f}%)"
]

ax.legend(
    wedges,
    legend_labels,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=TEXT_SIZE * 0.8,
    handlelength=1.2,
    labelspacing=1.4,
)

fig.suptitle(
    "Number of patients by PAM50 subtype",
    fontsize=16,
    y=0.85,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "subtype_distribution.png",
    dpi=300,
    facecolor=BG,
    bbox_inches="tight",
)
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


# Fraction missing for each patient
missing_per_patient = meth[meth_features].isna().mean(axis=1)

# Convert from proportions, such as 0.152, to percentages, such as 15.2
missing_percent = missing_per_patient * 100

# Main color
teal = "#0e5f60ff"
bg = "#fcfcfcff" 
text_size = 19

# Calculate median
median_missing = missing_percent.median()

fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor(bg)   # outside the plot
ax.set_facecolor(bg)          # inside the plot

# Histogram
ax.hist(
    missing_percent,
    bins=20,
    color=teal,
    alpha=0.78,
    edgecolor="white",
    linewidth=0.7
)

# Median line
ax.axvline(
    median_missing,
    color=teal,
    linestyle="--",
    linewidth=2,
    dashes=(5, 4)
)

# Median label
ax.text(
    median_missing + 1,
    ax.get_ylim()[1] * 0.88,
    f"Median: {median_missing:.1f}%",
    color=teal,
    fontsize=text_size * 0.8,
    fontweight="bold"
)

# Subtitle
ax.set_title(
    "Percentage of CpG sites with missing values for each patient",
    fontsize=text_size,
    pad=15
)

# Axis labels
ax.set_xlabel("Percent Missing CpG Sites", fontsize=text_size)
ax.set_ylabel("Number of Patients", fontsize=text_size)

# Tick parameters
ax.tick_params(axis="both", which="major", labelsize=text_size * 0.8)


# Show percent signs on the x-axis
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

# Optional: match the example's approximate x-axis range
xmin = missing_percent.min()
xmax = missing_percent.max()

padding = 0.5  # percent

ax.set_xlim(xmin - padding, xmax + padding)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Make remaining borders subtle
ax.spines["left"].set_color("#777777")
ax.spines["bottom"].set_color("#777777")

# Light horizontal grid
ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.6,
    alpha=0.18
)

# Keep the grid behind the bars
ax.set_axisbelow(True)

# Annotation box
q1 = missing_percent.quantile(0.25)
q3 = missing_percent.quantile(0.75)

ax.text(
    0.77,
    0.64,
    f"Middle 50% of patients\n have {q1:.1f}% - {q3:.1f}%\n missing CpG sites",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=text_size,
    color="#263238",
    linespacing=1.35,
    bbox=dict(
        boxstyle="round,pad=1.0",
        facecolor="#f1f8f7",
        edgecolor=teal,
        linewidth=1.2,
        alpha=0.9
    )
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "methylation_missingness_per_patient.png",
    dpi=300,
)

plt.show()


# In[12]:


missing_per_cpg = meth[meth_features].isna().mean(axis=0) * 100

teal = "#0e5f60ff"
background = "#fcfcfcff"
text_size = 19

# Separate CpGs with zero and nonzero missingness
zero_missing = (missing_per_cpg == 0).sum()
all_missing = (missing_per_cpg == 100).sum()
nonzero_missing = missing_per_cpg[(missing_per_cpg > 0) & (missing_per_cpg < 100)].sort_values()
some_missing = ((missing_per_cpg > 0) & (missing_per_cpg < 100)).sum()

pct_zero = zero_missing / len(missing_per_cpg) * 100
pct_all = all_missing / len(missing_per_cpg) * 100
pct_some = some_missing / len(missing_per_cpg) * 100
median_some = nonzero_missing.median()

x = np.arange(1, len(nonzero_missing) + 1)

fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor(background)
ax.set_facecolor(background)

ax.scatter(
    x,
    nonzero_missing,
    s=40,
    color=teal,
    alpha=0.55,
    edgecolors="none"
)

# Median among CpGs that actually have missing values
ax.axhline(
    median_some,
    color=teal,
    linestyle="--",
    linewidth=1.8,
    dashes=(5, 4)
)

ax.text(
    0.01,
    median_some + 2,
    f"Median among partially missing CpGs: {median_some:.1f}%",
    transform=ax.get_yaxis_transform(),
    color=teal,
    fontsize=text_size * 0.8,
    fontweight="bold"
)

fig.suptitle(
    "CpG sites with partial missingness only\n(0% and 100% missingness excluded)",
    fontsize=text_size,
    y=0.99
)

ax.set_xlabel("CpG Sites (ordered by missingness)", fontsize=text_size)
ax.set_ylabel("Percent of Patients Missing", fontsize=text_size)

ax.set_xlim(0, len(nonzero_missing) + 1)
ax.set_ylim(0, min(100, nonzero_missing.max() + 5))

ax.yaxis.set_major_formatter(
    mtick.PercentFormatter(xmax=100, decimals=0)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linewidth=0.6, alpha=0.18)
ax.set_axisbelow(True)

ax.tick_params(axis="both", which="major", labelsize=text_size * 0.8)

annotation = (
    f"{pct_zero:.1f}% of CpG sites\n"
    "have no missing values.\n\n"
    rf"$\bf{{{pct_some:.1f}\%\ of\ CpG\ sites}}$"
    "\n"
    r"$\bf{have\ some\ missingness.}$"
    "\n\n"
    f"{pct_all:.1f}% of CpG sites\n"
    "have all values missing."
)

# ax.text(
#     0.3,
#     0.6,
#     annotation,
#     transform=ax.transAxes,
#     ha="center",
#     va="center",
#     fontsize=text_size,
#     color="#263238",
#     linespacing=1.35,
#     bbox=dict(
#         boxstyle="round,pad=1.0",
#         facecolor="#f1f8f7",
#         edgecolor=teal,
#         linewidth=1.0
#     )
# )

ax.set_yscale("log")
ax.set_ylim(1e-1, 101)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "methylation_missingness_per_cpg.png",
    dpi=300,
    facecolor=background
)

plt.show()

# ------------------------------------------------------------------
# Calculate percent missing for each CpG site
# ------------------------------------------------------------------
missing_per_cpg = meth[meth_features].isna().mean(axis=0) * 100

# Exclude CpGs with 0% and 100% missingness
partial_missing = missing_per_cpg[
    (missing_per_cpg > 0) &
    (missing_per_cpg < 100)
]

# ------------------------------------------------------------------
# Plot styling
# ------------------------------------------------------------------
background = "#fcfcfc"
text_color = "#263238"
text_size = 19

# Sequential teal palette: light to dark
colors = [
    "#d9eceb",
    "#9fcac8",
    "#5d9997",
    "#0e5f60"
]

category_order = [
    "<1% Missingness",
    "1–10% Missingness",
    "10–50%\nMissingness",
    ">50% Missingness"
]

# ------------------------------------------------------------------
# Categorize partially missing CpG sites
# ------------------------------------------------------------------
missingness_categories = pd.Series(
    np.select(
        [
            partial_missing < 1,
            (partial_missing >= 1) & (partial_missing < 10),
            (partial_missing >= 10) & (partial_missing <= 50),
            partial_missing > 50
        ],
        category_order,
        default="Uncategorized"
    ),
    index=partial_missing.index,
    name="Missingness category"
)

# Count CpGs in each category
category_counts = (
    missingness_categories
    .value_counts()
    .reindex(category_order, fill_value=0)
)

total_cpgs = category_counts.sum()

# Optional summary table
summary = pd.DataFrame({
    "CpG sites": category_counts,
    "Percent": category_counts / total_cpgs * 100
})

print(summary)

# ------------------------------------------------------------------
# Create pie chart
# ------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(8, 6),
    facecolor=background
)

ax.set_facecolor(background)

wedges, _ = ax.pie(
    category_counts,
    colors=colors,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(
        edgecolor=background,
        linewidth=2
    )
)

# ------------------------------------------------------------------
# Add labels directly to slices
# ------------------------------------------------------------------
for wedge, category, count in zip(
    wedges,
    category_counts.index,
    category_counts.values
):
    pct = count / total_cpgs * 100

    # Midpoint angle of the wedge
    angle = (wedge.theta1 + wedge.theta2) / 2

    # Convert angle to x/y coordinates
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))

    label = f"{category}"

    # Place >50% outside with an arrow
    if category == ">50% Missingness":
        ax.annotate(
            label,
            xy=(x * 0.98, y * 0.98),
            xytext=(x * 0.8, 1.07),
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=text_size * 0.7,
            color=text_color,
            arrowprops=dict(
                arrowstyle="-",
                color=text_color,
                linewidth=1.2,
                connectionstyle="arc3,rad=0.1"
            )
        )

    else:
        ax.text(
            x * 0.60,
            y * 0.62,
            label,
            ha="center",
            va="center",
            fontsize=text_size * 0.7,
            color=text_color
        )

# ------------------------------------------------------------------
# Title and layout
# ------------------------------------------------------------------
fig.suptitle(
    "Distribution of Partial CpG Missingness",
    fontsize=text_size,
    y=0.95
)

ax.set_aspect("equal")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "methylation_partial_missingness_categories.png",
    dpi=300,
    facecolor=background,
    bbox_inches="tight"
)

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


# Colors
BG = "#fcfcfc"
BLUE = "#0e5f60"
PINK = "#9c224d"
TEXT_SIZE = 19

# Count samples
counts = survival_clean["BRCA_Subtype_PAM50"].value_counts().loc[["LumA", "LumB"]]
total = counts.sum()

fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=BG)
ax.set_facecolor(BG)

# Donut chart
wedges, _ = ax.pie(
    counts,
    startangle=90,
    colors=[BLUE, PINK],
    wedgeprops=dict(width=0.4, edgecolor=BG, linewidth=2)
)

# Center text
ax.text(
    0, 0,
    f"Total\n{total}\npatients",
    ha="center",
    va="center",
    fontsize=TEXT_SIZE
)

# Legend
legend_labels = [
    f"Luminal A\n{counts['LumA']} ({counts['LumA']/total*100:.0f}%)",
    f"Luminal B\n{counts['LumB']} ({counts['LumB']/total*100:.0f}%)"
]

ax.legend(
    wedges,
    legend_labels,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=TEXT_SIZE,
    handlelength=1.2,
    labelspacing=1.4,
)

fig.suptitle(
    "Number of patients by PAM50 subtype",
    fontsize=TEXT_SIZE,
    y=0.85,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "subtype_distribution_cleaned.png",
    dpi=300,
    facecolor=BG
)
plt.show()

# Colors
BG = "#fcfcfc"
BLUE = "#0e5f60"
PINK = "#9c224d"
TEXT_SIZE = 19

# Keep subtype order consistent
subtype_order = ["LumA", "LumB"]

# Number of patients in each subtype
counts = (
    survival_clean["BRCA_Subtype_PAM50"]
    .value_counts()
    .reindex(subtype_order)
)

# Proportion with an event/death
event_rates = (
    survival_clean
    .groupby("BRCA_Subtype_PAM50")["event"]
    .mean()
    .reindex(subtype_order)
)

fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=BG)
ax.set_facecolor(BG)

bars = ax.bar(
    ["Luminal A", "Luminal B"],
    event_rates,
    color=[BLUE, PINK],
    width=0.58
)

# Percentage labels above bars
for bar, rate in zip(bars, event_rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{rate:.1%}",
        ha="center",
        va="bottom",
        fontsize=TEXT_SIZE * 0.8
    )

# Add patient counts below subtype names
ax.set_xticks([0, 1])
ax.set_xticklabels([
    f"Luminal A\n(n = {counts['LumA']})",
    f"Luminal B\n(n = {counts['LumB']})"
], fontsize=TEXT_SIZE)

fig.suptitle(
    "Proportion of patients with events (deaths)",
    fontsize=TEXT_SIZE,
    y=0.90
)

ax.set_ylabel("Percent of patients", fontsize=TEXT_SIZE, labelpad=20)
ax.set_xlabel("")

# Percentage axis
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, max(event_rates) + 0.12)

# Horizontal dashed gridlines
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.45
)
ax.set_axisbelow(True)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.tick_params(axis="x", length=0)
ax.tick_params(axis="both", labelsize=TEXT_SIZE * 0.8)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "survival_event_by_subtype.png",
    dpi=300,
    facecolor=BG
)

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




