library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)
library(patchwork)

# ── Load data ─────────────────────────────────────────────────────────────
df <- read.csv("output/crumblr/meta_analysis_preg_vs_ctrl.csv")

# Clean up cell type names: strip numeric prefix for labels
df$label <- gsub("^\\d+\\s+", "", df$celltype)

# Flag significance
df$sig <- ifelse(df$fdr_combined < 0.05, "FDR < 0.05",
           ifelse(df$fdr_combined < 0.20, "FDR < 0.20", "NS"))
df$sig <- factor(df$sig, levels = c("FDR < 0.05", "FDR < 0.20", "NS"))

# Only label concordant + FDR < 0.20, or top 5 by p_combined
df$show_label <- (df$fdr_combined < 0.20) | (rank(df$p_combined) <= 5)

# ── Helper: make one scatterplot ──────────────────────────────────────────
make_scatter <- function(df, xcol, ycol, xlab, ylab) {
  # Correlations
  r_pearson <- cor(df[[xcol]], df[[ycol]], method = "pearson")
  r_spearman <- cor(df[[xcol]], df[[ycol]], method = "spearman")
  p_pearson <- cor.test(df[[xcol]], df[[ycol]], method = "pearson")$p.value
  p_spearman <- cor.test(df[[xcol]], df[[ycol]], method = "spearman")$p.value

  subtitle <- sprintf("Pearson r = %.3f (p = %.2e)  |  Spearman ρ = %.3f (p = %.2e)",
                       r_pearson, p_pearson, r_spearman, p_spearman)

  p <- ggplot(df, aes(x = .data[[xcol]], y = .data[[ycol]])) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", se = TRUE, color = "grey40", linewidth = 0.5, alpha = 0.15) +
    geom_point(aes(color = sig, size = sig), alpha = 0.7) +
    scale_color_manual(values = c("FDR < 0.05" = "#E63946", "FDR < 0.20" = "#F4A261", "NS" = "#A8DADC"),
                       name = "Meta-analysis\nFDR") +
    scale_size_manual(values = c("FDR < 0.05" = 4, "FDR < 0.20" = 3, "NS" = 1.5),
                      name = "Meta-analysis\nFDR") +
    geom_text_repel(
      data = df %>% filter(show_label),
      aes(label = label),
      size = 3, max.overlaps = 20, segment.alpha = 0.4,
      box.padding = 0.4, point.padding = 0.2,
      min.segment.length = 0.1
    ) +
    labs(x = paste0(xlab, " logFC"), y = paste0(ylab, " logFC"),
         subtitle = subtitle) +
    theme_minimal(base_size = 13) +
    theme(
      plot.subtitle = element_text(size = 10, color = "grey30"),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    ) +
    coord_cartesian(clip = "off")

  return(p)
}

# ── Make 3 pairwise plots ─────────────────────────────────────────────────
p1 <- make_scatter(df, "logFC_merfish", "logFC_slidetags", "MERFISH", "Slide-tags")
p2 <- make_scatter(df, "logFC_merfish", "logFC_xenium", "MERFISH", "Xenium 5k")
p3 <- make_scatter(df, "logFC_slidetags", "logFC_xenium", "Slide-tags", "Xenium 5k")

combined <- p1 + p2 + p3 +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Cross-platform concordance of compositional changes in pregnancy (PREG vs CTRL)",
    subtitle = "Subclass-level logFC from crumblr | Colored by meta-analysis FDR",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12, color = "grey40"),
      legend.position = "bottom"
    )
  )

ggsave("output/crumblr/figures/cross_platform_concordance_ggplot.png",
       combined, width = 20, height = 7, dpi = 250, bg = "white")
cat("Saved cross_platform_concordance_ggplot.png\n")

# ── Also make individual larger panels ────────────────────────────────────
for (info in list(
  list(x = "logFC_merfish", y = "logFC_slidetags", xn = "MERFISH", yn = "Slide-tags", fn = "merfish_vs_slidetags"),
  list(x = "logFC_merfish", y = "logFC_xenium", xn = "MERFISH", yn = "Xenium 5k", fn = "merfish_vs_xenium"),
  list(x = "logFC_slidetags", y = "logFC_xenium", xn = "Slide-tags", yn = "Xenium 5k", fn = "slidetags_vs_xenium")
)) {
  p <- make_scatter(df, info$x, info$y, info$xn, info$yn) +
    ggtitle(paste0(info$xn, " vs ", info$yn, " — PREG vs CTRL"))

  ggsave(paste0("output/crumblr/figures/concordance_", info$fn, ".png"),
         p, width = 8, height = 7, dpi = 250, bg = "white")
  cat(paste0("Saved concordance_", info$fn, ".png\n"))
}

# ── Print correlation summary ─────────────────────────────────────────────
cat("\n=== Correlation Summary ===\n")
pairs <- list(
  c("logFC_merfish", "logFC_slidetags", "MERFISH vs Slide-tags"),
  c("logFC_merfish", "logFC_xenium", "MERFISH vs Xenium 5k"),
  c("logFC_slidetags", "logFC_xenium", "Slide-tags vs Xenium 5k")
)
for (pair in pairs) {
  r_p <- cor.test(df[[pair[1]]], df[[pair[2]]], method = "pearson")
  r_s <- cor.test(df[[pair[1]]], df[[pair[2]]], method = "spearman")
  cat(sprintf("  %s: Pearson r=%.3f (p=%.2e), Spearman rho=%.3f (p=%.2e)\n",
              pair[3], r_p$estimate, r_p$p.value, r_s$estimate, r_s$p.value))
}
