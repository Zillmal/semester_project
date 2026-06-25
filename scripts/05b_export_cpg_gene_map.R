# 05b_export_cpg_gene_map.R
# Export a CpG -> PAM50 gene map so the Python side can pair each gene's
# promoter CpGs with its mRNA. Mirrors the probe selection logic of stage 04.
# Run from the repository root (like the other R scripts).

rm(list = ls())

library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(minfi)

# PAM50 gene set (current HGNC symbols), identical to stage 04.
pam50_genes <- c(
  "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
  "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
  "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
  "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
  "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
  "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B",
  "TYMS", "UBE2C", "UBE2T"
)

# The hg19 450K annotation uses legacy symbols for three PAM50 genes. Map them
# back to current symbols so meth genes match the RNA symbols (CDCA1=NUF2,
# KNTC2=NDC80, ORC6L=ORC6).
alias <- c(CDCA1 = "NUF2", KNTC2 = "NDC80", ORC6L = "ORC6")
pam50_meth <- union(pam50_genes, names(alias))
promoter_regions <- c("TSS1500", "TSS200", "1stExon")

# Selected promoter CpGs are exactly the rows of the methylation matrix in the
# stage-04 feature object; use them so the map matches the analysis set.
pam50_features <- readRDS("data/multiomics/pam50_features_brca.rds")
selected_cpgs <- rownames(pam50_features$meth)

ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
ann <- ann[rownames(ann) %in% selected_cpgs, , drop = FALSE]

# Gene/Group fields are ';'-separated and positionally paired per probe.
genes  <- strsplit(ann$UCSC_RefGene_Name,  ";")
groups <- strsplit(ann$UCSC_RefGene_Group, ";")

# For each CpG keep the PAM50 genes paired with a promoter region, canonicalised.
pairs <- Map(function(g, grp) {
  keep <- g %in% pam50_meth & grp %in% promoter_regions
  if (!any(keep)) return(character(0))
  hits <- unique(g[keep])
  unname(ifelse(hits %in% names(alias), alias[hits], hits))
}, genes, groups)

names(pairs) <- rownames(ann)
pairs <- pairs[lengths(pairs) > 0]

# Long format: one row per CpG-gene promoter pair (a CpG may map to >1 gene).
map_df <- do.call(rbind, lapply(names(pairs), function(cpg) {
  data.frame(cpg = cpg, gene = pairs[[cpg]], stringsAsFactors = FALSE)
}))

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
write.csv(map_df, "data/processed/cpg_gene_map.csv", row.names = FALSE)

cat("CpGs mapped:", length(pairs), "| CpG-gene pairs:", nrow(map_df),
    "| genes covered:", length(unique(map_df$gene)), "\n")
