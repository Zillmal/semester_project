# 09c_Differential_Methylation.R
# Do promoter methylation patterns associated with epigenetic
# silencing of PAM50 genes differ between Luminal A and Luminal B?

# Method: limma moderated t-test on M-values, LumA vs LumB, unadjusted for covariates


rm(list = ls())

if (!requireNamespace("limma", quietly = TRUE)) {
  stop("Package 'limma' not installed. Install via BiocManager::install('limma').")
}
library(limma)

dir.create("results/tables",  recursive = TRUE, showWarnings = FALSE)
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)

# load the data
# KNN-imputed beta matrix
# descriptive eda no leakage this is descriptive EDA
# with no train/test split, so global imputation cannot leak. 
# Patients in rows, CpGs in columns.
meth     <- read.csv("data/processed/meth_pam50_knn_imputed.csv",
                     row.names = 1, check.names = FALSE)
labels   <- read.csv("data/processed/labels_luminal_brca.csv",
                     stringsAsFactors = FALSE)
cpg_gene <- read.csv("data/processed/cpg_gene_map.csv",
                     stringsAsFactors = FALSE)
spearman <- read.csv("results/tables/cpg_expression_spearman.csv",
                     stringsAsFactors = FALSE)

#define silencing CpG set from 09
#"associated with epigenetic silencing" = significant negative methylation-
# expression correlation (FDR-controlled)
Q_SILENCING <- 0.05
silencing_cpgs <- spearman$cpg[spearman$spearman_rho < 0 &
                                 spearman$q_value < Q_SILENCING]
cat(sprintf("Silencing CpGs (rho<0 & q<%.2f): %d of %d tested\n",
            Q_SILENCING, length(silencing_cpgs), nrow(spearman)))

# align patients and subset CpGs
rownames(labels) <- labels$patient
common <- intersect(rownames(meth), rownames(labels))
common <- common[labels[common, "subtype"] %in% c("LumA", "LumB")]

cpgs    <- intersect(silencing_cpgs, colnames(meth))
beta    <- as.matrix(meth[common, cpgs])        # patients x CpGs
subtype <- factor(labels[common, "subtype"], levels = c("LumA", "LumB"))
cat(sprintf("Samples: LumA n=%d | LumB n=%d | CpGs tested: %d\n",
            sum(subtype == "LumA"), sum(subtype == "LumB"), ncol(beta)))

# Beta -> M-values M = log2(beta / (1 - beta)). 
EPS  <- 1e-3
beta <- pmin(pmax(beta, EPS), 1 - EPS)
M    <- log2(beta / (1 - beta))
M    <- t(M)                                    


design <- model.matrix(~ subtype)
fit    <- eBayes(lmFit(M, design))
res    <- topTable(fit, coef = "subtypeLumB", number = Inf, sort.by = "none")

# report beta differences for biological interpretability
# mean beta difference per CpG
mean_beta  <- function(mask) colMeans(beta[mask, , drop = FALSE])
delta_beta <- mean_beta(subtype == "LumB") - mean_beta(subtype == "LumA")

out <- data.frame(
  cpg        = rownames(res),
  logFC_M    = res$logFC,
  delta_beta = delta_beta[rownames(res)],
  t          = res$t,
  P          = res$P.Value,
  adj_P      = res$adj.P.Val,
  stringsAsFactors = FALSE
)
out <- merge(out, cpg_gene, by = "cpg", all.x = TRUE)   # attach gene

# Combined significance: FDR AND a biologically meaningful effect size
DB_THRESHOLD    <- 0.1
out$significant <- out$adj_P < 0.05 & abs(out$delta_beta) > DB_THRESHOLD
out <- out[order(out$adj_P, -abs(out$delta_beta)), ]

write.csv(out, "results/tables/differential_methylation_subtype.csv",
          row.names = FALSE)
cat(sprintf("Significant CpGs (adj.P<0.05 & |delta_beta|>%.1f): %d\n",
            DB_THRESHOLD, sum(out$significant)))

# Per-gene table
by_gene <- do.call(rbind, lapply(split(out, out$gene), function(g) data.frame(
  gene            = g$gene[1],
  n_cpgs          = nrow(g),
  n_significant   = sum(g$significant),
  mean_delta_beta = mean(g$delta_beta),
  min_adj_P       = min(g$adj_P),
  stringsAsFactors = FALSE
)))
by_gene <- by_gene[order(by_gene$min_adj_P), ]
write.csv(by_gene, "results/tables/differential_methylation_by_gene.csv",
          row.names = FALSE)

cat("Saved differential_methylation_subtype.csv and differential_methylation_by_gene.csv\n")
