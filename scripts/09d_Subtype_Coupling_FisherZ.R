# 09d_Subtype_Coupling_FisherZ.R
# test methylation~expression correlation differ between Luminal A and Luminal B
#
# statistically tests descriptive 09b using fisher z


rm(list = ls())

dir.create("results/tables",  recursive = TRUE, showWarnings = FALSE)
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)

#load KNN-imputed beta matrix
meth     <- read.csv("data/processed/meth_pam50_knn_imputed.csv",
                     row.names = 1, check.names = FALSE)
rna      <- read.csv("data/processed/rna_pam50.csv",
                     row.names = 1, check.names = FALSE)
labels   <- read.csv("data/processed/labels_luminal_brca.csv",
                     stringsAsFactors = FALSE)
spearman <- read.csv("results/tables/cpg_expression_spearman.csv",
                     stringsAsFactors = FALSE)


Q_SILENCING <- 0.05
sil <- spearman[spearman$spearman_rho < 0 & spearman$q_value < Q_SILENCING, ]
sil <- sil[sil$cpg %in% colnames(meth) & sil$gene %in% colnames(rna), ]
cat(sprintf("Silencing CpGs: %d across %d genes\n",
            nrow(sil), length(unique(sil$gene))))

# align
rownames(labels) <- labels$patient
common  <- Reduce(intersect, list(rownames(meth), rownames(rna), rownames(labels)))
common  <- common[labels[common, "subtype"] %in% c("LumA", "LumB")]
subtype <- factor(labels[common, "subtype"], levels = c("LumA", "LumB"))
cat(sprintf("Samples: LumA n=%d | LumB n=%d\n",
            sum(subtype == "LumA"), sum(subtype == "LumB")))

# Beta -> M-values 
EPS <- 1e-3
beta_to_M <- function(b) { b <- pmin(pmax(b, EPS), 1 - EPS); log2(b / (1 - b)) }

fisher_z <- function(r) atanh(r)              # 0.5 * log((1 + r) / (1 - r))


VAR_CONST <- 1.06

# Per-gene Fisher-z test

genes <- sort(unique(sil$gene))

rows <- lapply(genes, function(g) {
  g_cpgs <- sil$cpg[sil$gene == g]
  meth_g <- rowMeans(beta_to_M(as.matrix(meth[common, g_cpgs, drop = FALSE])))
  expr_g <- rna[common, g]

  a <- subtype == "LumA"; b <- subtype == "LumB"
  nA <- sum(a); nB <- sum(b)
  rA <- cor(meth_g[a], expr_g[a], method = "spearman")
  rB <- cor(meth_g[b], expr_g[b], method = "spearman")

  se <- sqrt(VAR_CONST / (nA - 3) + VAR_CONST / (nB - 3))
  z  <- (fisher_z(rA) - fisher_z(rB)) / se
  p  <- 2 * pnorm(-abs(z))

  data.frame(gene = g, n_cpgs = length(g_cpgs), n_LumA = nA, n_LumB = nB,
             rho_LumA = rA, rho_LumB = rB, rho_difference = rB - rA,
             fisher_z = z, P = p, stringsAsFactors = FALSE)
})

res <- do.call(rbind, rows)
res$adj_P       <- p.adjust(res$P, method = "BH")   # FDR across GENES
res$significant <- res$adj_P < 0.05
res <- res[order(res$adj_P, -abs(res$rho_difference)), ]

write.csv(res, "results/tables/subtype_coupling_fisherz.csv", row.names = FALSE)
cat(sprintf("Genes with significantly different coupling (adj.P<0.05): %d of %d\n",
            sum(res$significant), nrow(res)))


cat("Saved subtype_coupling_fisherz.csv")
