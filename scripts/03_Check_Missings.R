# Clean Environment ------------------------------------------------------------

rm(list = ls())


# Load Multiomics Data ---------------------------------------------------------

multiomics <- readRDS("../data/multiomics/multiomics_luminal_brca.rds")

beta <- multiomics$beta

rna_mat <- multiomics$rna

clinical_common <- multiomics$clinical

y <- multiomics$y

# ------------------------------------------------------------------------------