# Clean Environment ------------------------------------------------------------

rm(list = ls())


# Install Packages -------------------------------------------------------------

if (!require("BiocManager"))
  install.packages("BiocManager")

BiocManager::install("TCGAbiolinks")
BiocManager::install("SummarizedExperiment")
BiocManager::install("sesameData")
BiocManager::install("sesame")

# ------------------------------------------------------------------------------

# Load Packages

library(TCGAbiolinks)
library(sesame)
library(sesameData)
library(SummarizedExperiment)


# Create Directories -----------------------------------------------------------

dirs <- c(
  "data/raw",
  "data/multiomics")

for (d in dirs) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)
  }
}

# Methylation - Initial Data Download ------------------------------------------

meth_query <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "DNA Methylation",
  data.type = "Methylation Beta Value",
  platform = "Illumina Human Methylation 450"
)

GDCdownload(meth_query)

meth <- GDCprepare(meth_query)


# RNA - Initial Data Download -------------------------------------------------- 

rna_query <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)

GDCdownload(rna_query)

rna <- GDCprepare(rna_query)


# Clinical - Initial Data Download ---------------------------------------------

clinical <- GDCquery_clinic("TCGA-BRCA", type = "clinical")



# Save data to accelerate reloading data ---------------------------------------

saveRDS(meth, "data/raw/meth_tcga_brca.rds")
saveRDS(rna, "data/raw/rna_tcga_brca.rds")
saveRDS(clinical, "data/raw/clinical_brca.rds")
