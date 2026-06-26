# Clean Environment ------------------------------------------------------------

rm(list = ls())

# Load Packages

library(TCGAbiolinks)
library(sesame)
library(sesameData)
library(SummarizedExperiment)
library(glue)

# Load Prepared Data -----------------------------------------------------------

meth <- readRDS("../data/raw/meth_tcga_brca.rds")

rna <- readRDS("../data/raw/rna_tcga_brca.rds")

clinical <- readRDS("../data/raw/clinical_brca.rds")


# 01. Reduce the data to Luminal A and Lumnial B Subtypes ----------------------

subtype <- TCGAquery_subtype("BRCA")

table(subtype$BRCA_Subtype_PAM50, useNA = "ifany")

subtype$patient <- substr(subtype$patient, 1, 12)

luminal_patients <- subtype$patient[
  subtype$BRCA_Subtype_PAM50 %in% c("LumA", "LumB")
]

length(unique(luminal_patients))



meth_patients <- substr(colnames(meth), 1, 12)

rna_patients  <- substr(colnames(rna), 1, 12)

clinical$patient <- substr(clinical$submitter_id, 1, 12)



# Find the patients with lumA and lumB patients, for which all three data types
# are existent. 

common_luminal <- Reduce(
  intersect,
  list(
    unique(luminal_patients),
    unique(meth_patients),
    unique(rna_patients),
    unique(clinical$patient)
  )
)


glue("{length(common_luminal)} patiens have been identified, that have either LumA or LumB diagnosis and methylation data, rna data and clinical data is available.")




# Filter Datasets for these 563 Patients:

meth_common <- meth[
  ,
  meth_patients %in% common_luminal
]


rna_common <- rna[
  ,
  rna_patients %in% common_luminal
]


clinical_common <- clinical[
  clinical$patient %in% common_luminal,
]



# Add Pam50 label to clinical data.

pam50 <- subtype[
  subtype$patient %in% common_luminal,
  c("patient", "BRCA_Subtype_PAM50")
]

clinical_common <- merge(
  clinical_common,
  pam50,
  by = "patient"
)



# Synchronize the order of the data. -------------------------------------------

target_order <- sort(common_luminal)

meth_common <- meth_common[
  ,
  match(target_order,
        substr(colnames(meth_common), 1, 12))
]

rna_common <- rna_common[
  ,
  match(target_order,
        substr(colnames(rna_common), 1, 12))
]

clinical_common <- clinical_common[
  match(target_order,
        clinical_common$patient),
]



# Validate the synchronization process.

all(
  substr(colnames(meth_common),1,12) ==
    substr(colnames(rna_common),1,12)
)

all(
  clinical_common$patient ==
    substr(colnames(meth_common),1,12)
)



# Create final objects ---------------------------------------------------------

beta <- assay(meth_common)

rna_mat <- assay(rna_common)

y <- factor(
  clinical_common$BRCA_Subtype_PAM50,
  levels = c("LumA", "LumB")
)


# Check objects visually 

dim(beta)

# [1] 485577    563

dim(rna_mat)

# [1] 60660   563

table(y)

# LumA LumB 
# 422  141 


# Save as Multiomics dataset ---------------------------------------------------

multiomics <- list(
  beta = beta,
  rna = rna_mat,
  clinical = clinical_common,
  y = y
)

saveRDS(multiomics, "../data/multiomics/multiomics_luminal_brca.rds")

