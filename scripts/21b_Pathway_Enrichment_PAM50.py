from pathlib import Path
from math import comb
import pandas as pd

TABLES = Path("../results/tables")
matrix = pd.read_csv(TABLES / "gene_modality_matrix.csv", index_col=0)

UNIVERSE = list(matrix.index)                 # the 50 PAM50 genes = background
TOP_N = 10
FOREGROUND = matrix.sort_values("combined", ascending=False).head(TOP_N).index.tolist()
print(f"Universe: {len(UNIVERSE)} genes | foreground: {len(FOREGROUND)} genes")
print("Foreground:", FOREGROUND)

try:
    import gseapy as gp
except Exception as e:
    raise SystemExit(f"gseapy not available - run 'pip install gseapy'. ({e})")

LIBRARIES = ["KEGG_2021_Human", "Reactome_2022"]
MIN_UNIVERSE = 3

universe_set = set(UNIVERSE)
gene_sets = {}
for lib in LIBRARIES:
    try:
        d = gp.get_library(name=lib)
    except Exception as e:
        print(f"  could not load {lib}: {e}"); continue
    for term, genes in d.items():
        members = sorted(universe_set & set(genes))       # restrict to PAM50
        if len(members) >= MIN_UNIVERSE:
            gene_sets[f"{lib.split('_')[0]} | {term}"] = members
print(f"Testable pathways (>= {MIN_UNIVERSE} PAM50 genes): {len(gene_sets)}")

def hypergeom_sf(k, M, n, N):
    """P(X >= k): drawing N from M, n successes in M, observed k."""
    return sum(comb(n, i) * comb(M - n, N - i) for i in range(k, min(n, N) + 1)) / comb(M, N)

M = len(UNIVERSE)
N = len(FOREGROUND)
fg = set(FOREGROUND)

rows = []
for term, members in gene_sets.items():
    n = len(members)
    hits = sorted(set(members) & fg)
    k = len(hits)
    p = hypergeom_sf(k, M, n, N) if k > 0 else 1.0
    rows.append({"pathway": term, "n_in_pam50": n, "overlap": k,
                 "overlap_genes": ", ".join(hits), "p_value": p})

res = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

# Benjamini-Hochberg FDR
m = len(res)
res["fdr"] = (res["p_value"] * m / (res.index + 1)).clip(upper=1.0)
res["fdr"] = res["fdr"][::-1].cummin()[::-1]

res.to_csv(TABLES / "pathway_enrichment_pam50.csv", index=False)
res[res["overlap"] > 0].head(15).round(4)