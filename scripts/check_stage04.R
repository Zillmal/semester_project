# Sanity checks for stage 04 outputs -------------------------------------------
# Run from the repository root after 01 -> 02 -> 04. Verifies structure,
# sample alignment, value ranges and subtype counts of the PAM50 feature object.

f <- readRDS("../data/multiomics/pam50_features_brca.rds")

rna  <- f$rna
meth <- f$meth
clin <- f$clinical
y    <- f$y


# 1. Dimensions and sample count -----------------------------------------------

cat("RNA: ",  nrow(rna),  "genes x", ncol(rna),  "samples\n")
cat("Meth:",  nrow(meth), "CpGs  x", ncol(meth), "samples\n")
stopifnot(ncol(rna) == 563, ncol(meth) == 563, length(y) == 563)


# 2. Sample alignment across all layers ----------------------------------------

stopifnot(identical(colnames(rna), colnames(meth)))   # same patients, same order
stopifnot(identical(colnames(rna), clin$patient))     # clinical aligned too
cat("Sample alignment OK\n")


# 3. PAM50 gene coverage -------------------------------------------------------

pam50 <- c("ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1","CDC20",
           "CDC6","CDH3","CENPF","CEP55","CXXC5","EGFR","ERBB2","ESR1","EXO1",
           "FGFR4","FOXA1","FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17","KRT5",
           "MAPT","MDM2","MELK","MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1",
           "NDC80","NUF2","ORC6","PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6",
           "TMEM45B","TYMS","UBE2C","UBE2T")
cat("PAM50 genes present:", length(intersect(rownames(rna), pam50)), "/ 50\n")
missing <- setdiff(pam50, rownames(rna))
if (length(missing)) cat("Missing genes:", paste(missing, collapse = ", "), "\n")
stopifnot(nrow(meth) > 0)   # promoter-CpG selection must not be empty


# 4. Value sanity --------------------------------------------------------------

cat("RNA range (log2 norm):", paste(round(range(rna), 2), collapse = " .. "),
    "| NAs:", sum(is.na(rna)), "\n")
cat("Meth range (beta):", paste(round(range(meth, na.rm = TRUE), 3), collapse = " .. "),
    "| NA fraction:", round(mean(is.na(meth)), 3), "\n")
stopifnot(min(rna) >= 0)                                  # log2(x + 1) >= 0
stopifnot(all(is.na(meth) | (meth >= 0 & meth <= 1)))     # valid beta values


# 5. Subtype labels ------------------------------------------------------------

print(table(y))   # expect 422 LumA / 141 LumB

cat("\nAll checks passed.\n")
