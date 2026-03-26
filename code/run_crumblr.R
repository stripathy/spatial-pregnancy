#!/usr/bin/env Rscript
#
# crumblr compositional analysis: Pregnancy conditions
#
# Tests for cell type proportion changes across:
#   CTRL (nulliparous) vs PREG (pregnant E18) vs POSTPART (postpartum day 20)
#
# Formula: ~ condition  (no covariates needed: all female, same age)
# Contrasts: PREG vs CTRL, POSTPART vs CTRL, POSTPART vs PREG
#
# Input:  output/crumblr/crumblr_input_{dataset}_{method}[_{stratum}].csv
# Output: output/crumblr/crumblr_results_{dataset}_{method}[_{stratum}].csv

library(crumblr)
library(dreamlet)
library(variancePartition)
library(limma)

cat("Libraries loaded\n")

# ── Paths ──────────────────────────────────────────────────────────
base_dir <- path.expand("~/Github/spatial pregnancy")
in_dir <- file.path(base_dir, "output", "crumblr")
out_dir <- in_dir

# ── Discover input files ───────────────────────────────────────────
input_files <- Sys.glob(file.path(in_dir, "crumblr_input_*.csv"))
cat(sprintf("Found %d input files:\n", length(input_files)))
for (f in input_files) cat(sprintf("  %s\n", basename(f)))

all_results <- list()
idx <- 1

for (fpath in input_files) {
  # Parse level from filename
  fname <- tools::file_path_sans_ext(basename(fpath))
  level <- sub("crumblr_input_", "", fname)
  cat(sprintf("\n══ Processing: %s ══\n", level))

  # ── Load data ──────────────────────────────────────────────────
  df <- read.csv(fpath, stringsAsFactors = FALSE)
  cat(sprintf("  %d rows, %d donors, %d cell types\n",
              nrow(df), length(unique(df$donor)), length(unique(df$celltype))))

  # Check conditions
  cat(sprintf("  Conditions: %s\n", paste(sort(unique(df$condition)), collapse=", ")))

  # ── Pivot to wide count matrix (donors × cell types) ──────────
  count_wide <- reshape(df[, c("donor", "celltype", "count")],
                        idvar = "donor", timevar = "celltype",
                        direction = "wide")
  rownames(count_wide) <- count_wide$donor
  count_wide$donor <- NULL
  colnames(count_wide) <- sub("^count\\.", "", colnames(count_wide))
  count_wide[is.na(count_wide)] <- 0

  # ── Filter: cell types present in ≥50% of samples ─────────────
  presence <- colMeans(count_wide > 0)
  keep <- presence >= 0.5
  cat(sprintf("  Keeping %d / %d types (≥50%% presence)\n",
              sum(keep), length(keep)))
  count_wide <- count_wide[, keep, drop = FALSE]

  if (ncol(count_wide) < 2) {
    cat("  Skipping: fewer than 2 cell types\n")
    next
  }

  # ── Build metadata ─────────────────────────────────────────────
  meta <- unique(df[, c("donor", "condition")])
  rownames(meta) <- meta$donor
  meta <- meta[rownames(count_wide), , drop = FALSE]
  meta$condition <- factor(meta$condition, levels = c("CTRL", "PREG", "POSTPART"))

  cat(sprintf("  %d CTRL, %d PREG, %d POSTPART\n",
              sum(meta$condition == "CTRL"),
              sum(meta$condition == "PREG"),
              sum(meta$condition == "POSTPART")))

  # ── Run crumblr ────────────────────────────────────────────────
  count_mat <- as.matrix(count_wide)

  cobj <- tryCatch(
    crumblr(count_mat),
    error = function(e) {
      cat(sprintf("  crumblr error: %s\n", e$message))
      return(NULL)
    }
  )
  if (is.null(cobj)) next

  # ── Fit dream model ────────────────────────────────────────────
  form <- ~ condition

  fit <- tryCatch({
    f <- dream(cobj, form, meta)
    eBayes(f)
  }, error = function(e) {
    cat(sprintf("  dream error: %s\n", e$message))
    return(NULL)
  })
  if (is.null(fit)) next

  # ── Extract results for each contrast ──────────────────────────
  contrasts <- c("conditionPREG", "conditionPOSTPART")
  contrast_labels <- c("PREG_vs_CTRL", "POSTPART_vs_CTRL")

  for (ci in 1:length(contrasts)) {
    coef_name <- contrasts[ci]
    contrast_label <- contrast_labels[ci]

    # Check if coefficient exists
    if (!(coef_name %in% colnames(fit$coefficients))) {
      cat(sprintf("  Coefficient %s not found, skipping\n", coef_name))
      next
    }

    res <- topTable(fit, coef = coef_name, number = Inf, sort.by = "none")
    res$celltype <- rownames(res)
    res$level <- level
    res$contrast <- contrast_label

    # Standard error
    res$SE <- res$logFC / res$t

    # Per-level FDR
    res$FDR <- p.adjust(res$P.Value, method = "BH")

    # Save individual results
    res_sorted <- res[order(res$P.Value), ]
    out_file <- file.path(out_dir, sprintf("crumblr_results_%s_%s.csv",
                                            level, contrast_label))
    write.csv(res_sorted, out_file, row.names = FALSE)
    cat(sprintf("  %s: saved %d types\n", contrast_label, nrow(res)))

    all_results[[idx]] <- res
    idx <- idx + 1

    # Print summary
    n_fdr05 <- sum(res$FDR < 0.05)
    n_fdr10 <- sum(res$FDR < 0.10)
    n_nom05 <- sum(res$P.Value < 0.05)
    cat(sprintf("    FDR<0.05: %d | FDR<0.10: %d | nom p<0.05: %d\n",
                n_fdr05, n_fdr10, n_nom05))

    # Show top hits
    sig <- res[res$P.Value < 0.05, ]
    sig <- sig[order(sig$P.Value), ]
    if (nrow(sig) > 0) {
      cat("    Nominal p < 0.05:\n")
      for (i in 1:min(nrow(sig), 10)) {
        r <- sig[i, ]
        d <- ifelse(r$logFC > 0, "↑", "↓")
        cat(sprintf("      %-35s %s logFC=%+.4f p=%.5f FDR=%.4f\n",
                    r$celltype, d, r$logFC, r$P.Value, r$FDR))
      }
    }
  }

  # ── POSTPART vs PREG contrast (manual) ───────────────────────────
  # Create contrast matrix for POSTPART - PREG
  if ("conditionPOSTPART" %in% colnames(fit$coefficients) &&
      "conditionPREG" %in% colnames(fit$coefficients)) {

    con <- makeContrasts(conditionPOSTPART - conditionPREG, levels = fit$design)
    fit2 <- contrasts.fit(fit, con)
    fit2 <- eBayes(fit2)

    res <- topTable(fit2, number = Inf, sort.by = "none")
    res$celltype <- rownames(res)
    res$level <- level
    res$contrast <- "POSTPART_vs_PREG"
    res$SE <- res$logFC / res$t
    res$FDR <- p.adjust(res$P.Value, method = "BH")

    res_sorted <- res[order(res$P.Value), ]
    out_file <- file.path(out_dir, sprintf("crumblr_results_%s_POSTPART_vs_PREG.csv", level))
    write.csv(res_sorted, out_file, row.names = FALSE)

    n_fdr05 <- sum(res$FDR < 0.05)
    n_nom05 <- sum(res$P.Value < 0.05)
    cat(sprintf("  POSTPART_vs_PREG: FDR<0.05: %d | nom p<0.05: %d\n", n_fdr05, n_nom05))

    sig <- res[res$P.Value < 0.05, ]
    sig <- sig[order(sig$P.Value), ]
    if (nrow(sig) > 0) {
      cat("    Nominal p < 0.05:\n")
      for (i in 1:min(nrow(sig), 10)) {
        r <- sig[i, ]
        d <- ifelse(r$logFC > 0, "↑", "↓")
        cat(sprintf("      %-35s %s logFC=%+.4f p=%.5f FDR=%.4f\n",
                    r$celltype, d, r$logFC, r$P.Value, r$FDR))
      }
    }

    all_results[[idx]] <- res
    idx <- idx + 1
  }
}

# ── Combine all results ────────────────────────────────────────────
if (length(all_results) > 0) {
  combined <- do.call(rbind, all_results)
  combined <- combined[order(combined$P.Value), ]
  out_file <- file.path(out_dir, "crumblr_results_all.csv")
  write.csv(combined, out_file, row.names = FALSE)
  cat(sprintf("\nSaved combined: %s (%d total rows)\n",
              basename(out_file), nrow(combined)))
} else {
  cat("\nNo results to combine!\n")
}

cat("\nDone!\n")
