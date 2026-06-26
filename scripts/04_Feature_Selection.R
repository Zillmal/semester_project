# Clean Environment ------------------------------------------------------------

rm(list = ls())


# Purpose ----------------------------------------------------------------------
# Stage 04: restrict the multi-omics cohort to PAM50 features.
#   - RNA : keep PAM50 genes, normalize STAR counts (DESeq2, log2).
#   - Meth: keep CpGs in PAM50 promoter regions (TSS1500/TSS200/1stExon).
# Input : data/multiomics/multiomics_luminal_brca.rds (output of 02_Merge_Data.R),
#         whose beta/rna/clinical/y are already aligned to the same 563 patients.
# (CV folds are set up later, at the modeling stage, where they are used.)


# Install / Load Packages ------------------------------------------------------

if (!require("BiocManager"))
  install.packages("BiocManager")

BiocManager::install("DESeq2")
BiocManager::install("org.Hs.eg.db")
BiocManager::install("minfi")
BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19")

library(SummarizedExperiment)
library(DESeq2)
library(org.Hs.eg.db)
library(minfi)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)


# Load Multiomics Data ---------------------------------------------------------

multiomics <- readRDS("../data/multiomics/multiomics_luminal_brca.rds")

beta            <- multiomics$beta   # CpG x samples, raw beta values
rna_mat         <- multiomics$rna    # gene x samples, raw STAR counts
clinical_common <- multiomics$clinical
y               <- multiomics$y      # LumA / LumB factor, sample order of 02


# PAM50 Gene Set ---------------------------------------------------------------
# Current HGNC symbols; matches the official UNC PAM50 set (50 genes).

pam50_genes <- c(
  "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
  "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
  "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
  "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
  "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
  "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B",
  "TYMS", "UBE2C", "UBE2T"
)


# 01. RNA: subset to PAM50 genes and normalize ---------------------------------

# Map Ensembl IDs (drop version suffix) to gene symbols, keep PAM50 genes.
rownames(rna_mat) <- sub("\\..*", "", rownames(rna_mat))

gene_map <- AnnotationDbi::select(
  org.Hs.eg.db,
  keys    = rownames(rna_mat),
  columns = "SYMBOL",
  keytype = "ENSEMBL"
)

pam50_ensembl <- unique(gene_map$ENSEMBL[gene_map$SYMBOL %in% pam50_genes])
pam50_ensembl <- pam50_ensembl[pam50_ensembl %in% rownames(rna_mat)]

# Size factors are estimated on ALL genes (library composition); PAM50 rows are
# extracted afterwards. round() because DESeq2 expects integer counts.
dds <- DESeqDataSetFromMatrix(
  countData = round(rna_mat),
  colData   = DataFrame(y = y, row.names = colnames(rna_mat)),
  design    = ~ 1
)
dds <- estimateSizeFactors(dds)

rna_pam50 <- log2(counts(dds, normalized = TRUE)[pam50_ensembl, ] + 1)

# Use gene symbols as row names for interpretability.
rownames(rna_pam50) <- gene_map$SYMBOL[match(rownames(rna_pam50), gene_map$ENSEMBL)]

dim(rna_pam50)   # genes x 563


# 02. Methylation: subset to PAM50 promoter CpGs -------------------------------

# UCSC_RefGene_Name / _Group are ';'-separated and positionally paired per probe
# (a probe can be e.g. TSS200 for one gene and Body for another). Keep a probe
# only if a PAM50 gene is paired with a promoter region.
ann450k <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)

# hg19 450K annotation may use legacy symbols for three PAM50 genes; add the
# aliases so their promoter CpGs are not missed (CDCA1=NUF2, KNTC2=NDC80,
# ORC6L=ORC6). The RNA side uses current symbols via org.Hs.eg.db.
pam50_meth <- union(pam50_genes, c("CDCA1", "KNTC2", "ORC6L"))

genes  <- strsplit(ann450k$UCSC_RefGene_Name,  ";")
groups <- strsplit(ann450k$UCSC_RefGene_Group, ";")

is_promoter_pam50 <- mapply(
  function(g, grp) any(g %in% pam50_meth &
                       grp %in% c("TSS1500", "TSS200", "1stExon")),
  genes, groups
)

promoter_probes <- rownames(ann450k)[is_promoter_pam50]
meth_pam50 <- beta[rownames(beta) %in% promoter_probes, , drop = FALSE]

dim(meth_pam50)  # promoter CpGs x 563

cpg_annotation_summary <- ann450k[promoter_probes, c(
  "Name",
  "chr",
  "pos",
  "UCSC_RefGene_Name",
  "UCSC_RefGene_Group"
)]

write.csv(
  cpg_annotation_summary,
  "../data/multiomics/cpg_annotation_summary_pam50_promoters.csv",
  row.names = FALSE
)


# 03. Align samples and check missingness --------------------------------------

# 02 aligned all layers by patient; RNA/meth full barcodes differ by aliquot, so
# reduce column names to the 12-char patient ID and verify identical order.
colnames(rna_pam50)  <- substr(colnames(rna_pam50),  1, 12)
colnames(meth_pam50) <- substr(colnames(meth_pam50), 1, 12)
stopifnot(identical(colnames(rna_pam50), colnames(meth_pam50)))

# RNA has no missing values; methylation NAs are imputed later (KNN, stage 05).
cat("Methylation missing fraction per sample:\n")
print(summary(colMeans(is.na(meth_pam50))))
cat("Methylation missing fraction per CpG:\n")
print(summary(rowMeans(is.na(meth_pam50))))


# 04. Assemble and save feature object -----------------------------------------

pam50_features <- list(
  rna      = rna_pam50,
  meth     = meth_pam50,
  clinical = clinical_common,
  y        = y
)

saveRDS(pam50_features, "../data/multiomics/pam50_features_brca.rds")
