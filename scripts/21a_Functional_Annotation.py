from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TABLES = Path("../results/tables")
FIGURES = Path("../results/figures"); FIGURES.mkdir(parents=True, exist_ok=True)

matrix = pd.read_csv(TABLES / "gene_modality_matrix.csv", index_col=0)  
corr = pd.read_csv(TABLES / "gene_methylation_expression_correlation_summary.csv").set_index("gene")

TOP_N = 10
top = matrix.sort_values("combined", ascending=False).head(TOP_N)
print("Top genes:", list(top.index))

GROUP = {}
for g in ["ANLN","UBE2T","NDC80","BIRC5","ORC6","CDC6","MYBL2","CCNE1","CDC20","CENPF",
          "CCNB1","CEP55","KIF2C","NUF2","MKI67","PTTG1","MELK","RRM2","EXO1","UBE2C","TYMS","MYC"]:
    GROUP[g] = "Proliferation / cell cycle"
for g in ["ESR1","PGR","FOXA1","MLPH","BAG1","SLC39A6","NAT1","BCL2","MAPT","CXXC5","GPR160"]:
    GROUP[g] = "ER / luminal signaling"
for g in ["ERBB2","GRB7"]:
    GROUP[g] = "HER2"
for g in ["FOXC1","CDH3","KRT17","KRT5","KRT14","MIA","EGFR","SFRP1","PHGDH","FGFR4","ACTR3B"]:
    GROUP[g] = "Basal / myoepithelial"
for g in ["MMP11","BLVRA","MDM2","TMEM45B"]:
    GROUP[g] = "Invasion / other"

ROLE = {
 "ESR1":"Estrogen receptor - luminal identity", "PGR":"Progesterone receptor - hormone",
 "FOXA1":"Pioneer TF for ER", "MLPH":"Melanophilin - luminal marker",
 "NAT1":"N-acetyltransferase - ER-associated", "BCL2":"Anti-apoptosis - ER-associated",
 "MAPT":"Tau - ER-associated (tamoxifen)", "BAG1":"BCL2-associated - anti-apoptosis",
 "SLC39A6":"Zinc transporter - ER-associated", "CXXC5":"Wnt/methylation-linked TF",
 "GPR160":"Orphan GPCR - luminal-associated",
 "ERBB2":"HER2 receptor - 17q12 amplicon", "GRB7":"Adaptor - co-amplified with HER2",
 "FOXC1":"Basal-like TF", "CDH3":"P-cadherin - basal marker",
 "KRT5":"Basal keratin", "KRT14":"Basal keratin", "KRT17":"Basal keratin",
 "MIA":"Melanoma-associated - aggressive", "EGFR":"EGF receptor - basal/aggressive",
 "SFRP1":"Wnt antagonist - often methylation-silenced", "PHGDH":"Serine biosynthesis - metabolism",
 "FGFR4":"FGF receptor - growth signalling", "ACTR3B":"Actin-related - cytoskeleton",
 "ANLN":"Anillin - cytokinesis", "UBE2T":"Ubiquitin conjugation - proliferation",
 "NDC80":"Kinetochore - mitosis", "BIRC5":"Survivin - anti-apoptosis/proliferation",
 "ORC6":"Origin recognition - DNA replication", "CDC6":"Replication licensing",
 "MYBL2":"B-Myb - cell-cycle TF", "CCNE1":"Cyclin E1 - G1/S",
 "CDC20":"Anaphase-promoting - mitosis", "CENPF":"Centromere protein - mitosis",
 "CCNB1":"Cyclin B1 - G2/M", "CEP55":"Centrosome - cytokinesis",
 "KIF2C":"Kinesin - mitotic spindle", "NUF2":"Kinetochore - mitosis",
 "MKI67":"Ki-67 - proliferation marker", "PTTG1":"Securin - mitosis",
 "MELK":"Kinase - proliferation", "RRM2":"Ribonucleotide reductase - DNA synthesis",
 "EXO1":"Exonuclease - DNA repair/replication", "UBE2C":"Ubiquitin conjugation - mitosis",
 "TYMS":"Thymidylate synthase - DNA synthesis",
 "MMP11":"Stromelysin-3 - stroma/invasion", "BLVRA":"Biliverdin reductase - metabolism",
 "MDM2":"p53 negative regulator", "MYC":"MYC oncogene - proliferation",
 "TMEM45B":"Transmembrane - poorly characterised",
}

ann = pd.DataFrame(index=top.index)
ann["rank"] = top["rank"].astype(int)
ann["combined"] = top["combined"].round(2)
#ann["dominant_modality"] = np.where(top["methylation_mean"] >= top["expression_mean"],
#                                    "methylation", "expression")                         # removing this for now. 
ann["functional_group"] = [GROUP.get(g, "Other") for g in ann.index]
ann["mean_rho"] = corr["mean_rho"].reindex(ann.index).round(2)        
ann["min_rho"] = corr["min_rho"].reindex(ann.index).round(2)          
ann["min_rho_cpg"] = corr["min_rho_cpg"].reindex(ann.index)
ann["role"] = [ROLE.get(g, "") for g in ann.index]

ann = ann.reset_index().rename(columns={"index": "gene"})
ann.to_csv(TABLES / "functional_annotation_top10.csv", index=False)
ann