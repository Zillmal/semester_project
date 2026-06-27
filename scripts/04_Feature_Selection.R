# Clean Environment ------------------------------------------------------------

rm(list = ls())


# Purpose ----------------------------------------------------------------------
# 04_Feature_Selection removes all RNA and Methylation data that does not belong 
#        to the PAM50 features which serve to distinguish between LumA and LumB
#   - RNA : keep PAM50 genes, normalize STAR counts (DESeq2, log2).
#   - Meth: keep CpGs in PAM50 promoter regions (TSS1500/TSS200/1stExon).
# Input : data/multiomics/multiomics_luminal_brca.rds (output of 02_Merge_Data.R),


# Install / Load Packages ------------------------------------------------------

if (!require("BiocManager"))
  install.packages("BiocManager")

BiocManager::install("DESeq2")
BiocManager::install("org.Hs.eg.db")
BiocManager::install("minfi")
BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19")

library(SummarizedExperiment)   # base container class
library(DESeq2)     # for RNA-Normalization
library(org.Hs.eg.db)   # for annotation: maps Ensembl IDs <-> gene symbols
library(minfi)      # for Methylation-Array-Analysis
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)   # 450K probe annotation (hg19)


# Load Multiomics Data ---------------------------------------------------------
# multiomics: list of 4 components, all aligned to the same 563 patients (422 LumA / 141 LumB).
multiomics <- readRDS("../data/multiomics/multiomics_luminal_brca.rds")
# all in identical sample order
beta            <- multiomics$beta   # CpG x samples, raw beta values
rna_mat         <- multiomics$rna    # gene x samples, raw STAR counts
clinical_common <- multiomics$clinical  # dataframe with clinical variables
y               <- multiomics$y      # LumA / LumB factor


# PAM50 Gene Set ---------------------------------------------------------------
# The 50 official UNC PAM50 genes, written in current standard gene
# symbols so they match the symbols returned by org.Hs.eg.db below.
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

# drop version suffix from Ensembl IDs
rownames(rna_mat) <- sub("\\..*", "", rownames(rna_mat))

# translation map: Ensembl ID <-> standard gene symbol
gene_map <- AnnotationDbi::select(
  org.Hs.eg.db,
  keys    = rownames(rna_mat),
  columns = "SYMBOL",
  keytype = "ENSEMBL"
)

# subset to those defined in pam50_genes present in count-matrix
pam50_ensembl <- unique(gene_map$ENSEMBL[gene_map$SYMBOL %in% pam50_genes])
pam50_ensembl <- pam50_ensembl[pam50_ensembl %in% rownames(rna_mat)]

# dds bundles counts, sample info and the model formula 
dds <- DESeqDataSetFromMatrix(
  countData = round(rna_mat), # round() to ensure integer, count-matrix for each gene & sample
  colData   = DataFrame(y = y, row.names = colnames(rna_mat)), # sample info
  design    = ~ 1 # model formula for intercept only
)
# estimate size factor for each sample 
dds <- estimateSizeFactors(dds)

# PAM50 expression: size-factor-normalized, log2(+1)-transformed
rna_pam50 <- log2(counts(dds, normalized = TRUE)[pam50_ensembl, ] + 1)

# Use gene symbols as row names for interpretability.
rownames(rna_pam50) <- gene_map$SYMBOL[match(rownames(rna_pam50), gene_map$ENSEMBL)]

dim(rna_pam50)   # genes x 563


# 02. Methylation: subset to PAM50 promoter CpGs -------------------------------

# UCSC_RefGene_Name:  gene(s) the probe's CpG is annotated to (';'-separated)
# UCSC_RefGene_Group: matching region(s) in those genes (e.g. TSS200, Body)
# PAIRWISE checking: 
#     keep a probe only if a PAM50 gene sits at a position that is ALSO a promoter region

ann450k <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)

# add old gene names (CDCA1=NUF2, KNTC2=NDC80, ORC6L=ORC6) to account for naming mismatch 
# between the old Illumina 450K annotation and current symbols (org.Hs.eg.db)
pam50_meth <- union(pam50_genes, c("CDCA1", "KNTC2", "ORC6L"))

# split both vectors to position aligned lists (split at ;)
genes  <- strsplit(ann450k$UCSC_RefGene_Name,  ";")
groups <- strsplit(ann450k$UCSC_RefGene_Group, ";")

# MASK: returns True or False for each probe
# True if a PAM50 gene and a promoter region coincide
is_promoter_pam50 <- mapply(
  function(g, grp) any(g %in% pam50_meth &
                       grp %in% c("TSS1500", "TSS200", "1stExon")),
  genes, groups
)

# apply mask 
promoter_probes <- rownames(ann450k)[is_promoter_pam50]
meth_pam50 <- beta[rownames(beta) %in% promoter_probes, , drop = FALSE]

dim(meth_pam50)  # promoter CpGs x samples

# reference table: which CpGs were selected and which PAM50 gene/region they belong to
cpg_annotation_summary <- ann450k[promoter_probes, c(
  "Name",                # CpG probe ID (cg...)
  "chr",                 # chromosome
  "pos",                 # genomic position (bp, hg19)
  "UCSC_RefGene_Name",   # gene(s) the CpG is annotated to
  "UCSC_RefGene_Group"   # region(s) within those genes (e.g. TSS200)
)]

write.csv(
  cpg_annotation_summary,
  "../data/multiomics/cpg_annotation_summary_pam50_promoters.csv",
  row.names = FALSE
)


# 03. Align samples and check missingness --------------------------------------

# cut full barcodes to 12-char patient ID (RNA/meth use different sample portions)
# stopifnot: verify both layers have identical patient order
colnames(rna_pam50)  <- substr(colnames(rna_pam50),  1, 12)
colnames(meth_pam50) <- substr(colnames(meth_pam50), 1, 12)
stopifnot(identical(colnames(rna_pam50), colnames(meth_pam50)))

# RNA has no missing values; methylation NAs are imputed later with KNN.
cat("Methylation missing fraction per sample:\n")
print(summary(colMeans(is.na(meth_pam50))))
cat("Methylation missing fraction per CpG:\n")
print(summary(rowMeans(is.na(meth_pam50))))


# 04. Assemble and save feature object -----------------------------------------

pam50_features <- list(
  rna      = rna_pam50,   # matrix: genes x samples
  meth     = meth_pam50,  # matrix: CpGs x samples
  clinical = clinical_common, # df: samples x clinical variables
  y        = y  # factor: LumA vs LumB
)

saveRDS(pam50_features, "../data/multiomics/pam50_features_brca.rds")
