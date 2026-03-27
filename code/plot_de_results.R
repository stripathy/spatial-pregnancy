#!/usr/bin/env Rscript
#
# DE result visualizations:
#   1. Meta-analysis volcano plot (gene-level)
#   2. Per-platform volcano plots
#   3. Slide-tags vs Xenium 5k concordance scatter
#

library(ggplot2)
library(ggrepel)
library(dplyr)

library(here)
base_dir <- here::here()
de_dir   <- file.path(base_dir, "output", "de")
fig_dir  <- file.path(de_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Meta-analysis volcano ──────────────────────────────────────────────
cat("Plotting meta-analysis volcano...\n")
meta <- read.csv(file.path(de_dir, "de_meta_PREG_vs_CTRL.csv"))

meta$neg_log10_p <- -log10(meta$p_combined)
meta$fdr_cat <- case_when(
  meta$fdr_combined < 0.05 ~ "FDR < 0.05",
  meta$fdr_combined < 0.10 ~ "FDR < 0.10",
  meta$fdr_combined < 0.20 ~ "FDR < 0.20",
  TRUE ~ "NS"
)
meta$fdr_cat <- factor(meta$fdr_cat, levels = c("FDR < 0.05", "FDR < 0.10", "FDR < 0.20", "NS"))

# Short label: gene (celltype short)
meta$ct_short <- sub("^(\\d+ \\S+ \\S+).*", "\\1", meta$celltype)
meta$label <- paste0(meta$gene, "\n(", meta$ct_short, ")")

# Label FDR<0.10
meta$show_label <- meta$fdr_combined < 0.10

# Number of platforms
meta$n_plat_f <- factor(meta$n_platforms, levels = c(2, 3))

p_meta <- ggplot(meta, aes(x = mean_logFC, y = neg_log10_p)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
  geom_point(aes(color = fdr_cat, shape = n_plat_f, size = fdr_cat), alpha = 0.7) +
  scale_color_manual(
    values = c("FDR < 0.05" = "#E63946", "FDR < 0.10" = "#F4A261",
               "FDR < 0.20" = "#E9C46A", "NS" = "#B0C4DE"),
    name = "Meta-analysis FDR"
  ) +
  scale_size_manual(
    values = c("FDR < 0.05" = 4, "FDR < 0.10" = 3.5, "FDR < 0.20" = 2.5, "NS" = 1.2),
    name = "Meta-analysis FDR"
  ) +
  scale_shape_manual(
    values = c("2" = 17, "3" = 16),
    name = "Platforms",
    labels = c("2 platforms", "3 platforms")
  ) +
  geom_text_repel(
    data = meta %>% filter(show_label),
    aes(label = label),
    size = 3.2, max.overlaps = 30, segment.alpha = 0.5,
    box.padding = 0.5, point.padding = 0.3,
    min.segment.length = 0.1, force = 2, lineheight = 0.85
  ) +
  annotate("text", x = max(meta$mean_logFC, na.rm=TRUE) * 0.85,
           y = -log10(0.05) + 0.15, label = "FDR = 0.05",
           color = "grey40", size = 3, hjust = 1) +
  labs(
    x = "Meta-analysis mean logFC (PREG vs CTRL)",
    y = expression(-log[10](p[combined])),
    title = "Gene-level DE meta-analysis: PREG vs CTRL",
    subtitle = sprintf(
      "Stouffer's method | MERFISH (n=3), Slide-tags (n=3), Xenium 5k (n=3) | %d FDR<0.05, %d FDR<0.10",
      sum(meta$fdr_combined < 0.05), sum(meta$fdr_combined < 0.10)
    )
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "de_meta_volcano.png"), p_meta,
       width = 13, height = 8, dpi = 250, bg = "white")
cat("  Saved de_meta_volcano.png\n")

# ── 2. Per-platform volcano plots ────────────────────────────────────────
cat("\nPlotting per-platform volcanoes...\n")

platform_files <- list(
  MERFISH = file.path(de_dir, "de_merfish_PREG_vs_CTRL.csv"),
  `Slide-tags` = file.path(de_dir, "de_slidetags_PREG_vs_CTRL.csv"),
  `Xenium 5k` = file.path(de_dir, "de_xenium5k_PREG_vs_CTRL.csv")
)

platform_n <- c(MERFISH = 3, `Slide-tags` = 3, `Xenium 5k` = 3)

plots <- list()
for (nm in names(platform_files)) {
  f <- platform_files[[nm]]
  if (!file.exists(f)) next
  df <- read.csv(f)
  df$neg_log10_fdr <- -log10(pmax(df$FDR, 1e-30))
  df$neg_log10_p   <- -log10(pmax(df$PValue, 1e-30))
  df$fdr_cat <- case_when(
    df$FDR < 0.05 ~ "FDR < 0.05",
    df$FDR < 0.10 ~ "FDR < 0.10",
    df$FDR < 0.20 ~ "FDR < 0.20",
    TRUE ~ "NS"
  )
  df$fdr_cat <- factor(df$fdr_cat, levels = c("FDR < 0.05", "FDR < 0.10", "FDR < 0.20", "NS"))
  df$ct_short <- sub("^(\\d+ \\S+ \\S+).*", "\\1", df$celltype)
  df$label <- paste0(df$gene, " (", df$ct_short, ")")
  df$show_label <- df$FDR < 0.05

  n_sig <- sum(df$FDR < 0.05)
  n_cells <- platform_n[nm]

  # Hex-binned density plot (fast for large datasets)
  p <- ggplot(df, aes(x = logFC, y = neg_log10_p)) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50", linewidth = 0.4) +
    geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_point(aes(color = fdr_cat, size = fdr_cat), alpha = 0.4) +
    scale_color_manual(
      values = c("FDR < 0.05" = "#E63946", "FDR < 0.10" = "#F4A261",
                 "FDR < 0.20" = "#E9C46A", "NS" = "#B0C4DE"),
      name = "FDR"
    ) +
    scale_size_manual(
      values = c("FDR < 0.05" = 2.5, "FDR < 0.10" = 2, "FDR < 0.20" = 1.5, "NS" = 0.8),
      name = "FDR"
    ) +
    geom_text_repel(
      data = df %>% filter(show_label) %>% slice_min(FDR, n = 20),
      aes(label = label),
      size = 2.8, max.overlaps = 15, segment.alpha = 0.4,
      box.padding = 0.3, point.padding = 0.2
    ) +
    labs(
      x = "logFC (PREG vs CTRL)",
      y = expression(-log[10](p)),
      title = nm,
      subtitle = sprintf("n=%d animals per group | %d genes FDR<0.05 | %d cell types",
                         n_cells, n_sig, length(unique(df$celltype)))
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(size = 10, color = "grey30"),
      legend.position = "right",
      panel.grid.minor = element_blank()
    )

  out <- file.path(fig_dir, paste0("de_volcano_", gsub("[^a-zA-Z0-9]", "_", nm), ".png"))
  ggsave(out, p, width = 12, height = 8, dpi = 250, bg = "white")
  cat(sprintf("  Saved %s\n", basename(out)))
}

# ── 3. Slide-tags vs Xenium 5k concordance ───────────────────────────────
cat("\nPlotting Slide-tags vs Xenium 5k concordance...\n")

st  <- read.csv(file.path(de_dir, "de_slidetags_PREG_vs_CTRL.csv"))
xe  <- read.csv(file.path(de_dir, "de_xenium5k_PREG_vs_CTRL.csv"))

# Join on gene + celltype
shared <- inner_join(st,  xe,  by = c("gene", "celltype"), suffix = c("_st", "_xe"))
cat(sprintf("  Shared (gene, celltype) pairs: %d\n", nrow(shared)))

# Overall correlation
r_all  <- cor(shared$logFC_st, shared$logFC_xe, method = "pearson")
rho_all <- cor(shared$logFC_st, shared$logFC_xe, method = "spearman")

# Color by significance in either platform
shared$sig_either <- shared$FDR_st < 0.05 | shared$FDR_xe < 0.05
shared$sig_both   <- shared$FDR_st < 0.05 & shared$FDR_xe < 0.05

shared$sig_cat <- case_when(
  shared$sig_both   ~ "FDR<0.05 both",
  shared$sig_either ~ "FDR<0.05 either",
  TRUE              ~ "NS"
)
shared$sig_cat <- factor(shared$sig_cat, levels = c("FDR<0.05 both", "FDR<0.05 either", "NS"))

shared$label <- ifelse(shared$sig_both, paste0(shared$gene, "\n(", sub("^(\\d+ \\S+).*","\\1",shared$celltype), ")"), NA)

p_conc <- ggplot(shared, aes(x = logFC_st, y = logFC_xe)) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
  geom_point(aes(color = sig_cat, size = sig_cat), alpha = 0.3) +
  scale_color_manual(
    values = c("FDR<0.05 both" = "#E63946", "FDR<0.05 either" = "#F4A261", "NS" = "#B0C4DE"),
    name = "Significance"
  ) +
  scale_size_manual(
    values = c("FDR<0.05 both" = 3, "FDR<0.05 either" = 2, "NS" = 0.5),
    name = "Significance"
  ) +
  geom_text_repel(
    data = shared %>% filter(!is.na(label)),
    aes(label = label),
    size = 2.8, max.overlaps = 20, segment.alpha = 0.5,
    box.padding = 0.4, lineheight = 0.85
  ) +
  geom_smooth(method = "lm", color = "#2196F3", linewidth = 0.8, se = TRUE, alpha = 0.15) +
  labs(
    x = "logFC Slide-tags (PREG vs CTRL)",
    y = "logFC Xenium 5k (PREG vs CTRL)",
    title = "Slide-tags vs Xenium 5k DE concordance",
    subtitle = sprintf("n=%d (gene, celltype) pairs | Pearson r = %.3f | Spearman ρ = %.3f",
                       nrow(shared), r_all, rho_all)
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "de_concordance_st_vs_xenium.png"), p_conc,
       width = 10, height = 9, dpi = 250, bg = "white")
cat("  Saved de_concordance_st_vs_xenium.png\n")

# ── 4. Per-cell-type concordance bar chart (top 15) ─────────────────────
cat("\nPlotting per-cell-type concordance...\n")
per_ct <- read.csv(file.path(de_dir, "de_concordance_slidetags_xenium_per_ct.csv"))

top <- per_ct %>%
  filter(n_genes >= 50) %>%
  arrange(desc(pearson_r)) %>%
  slice(1:20)

top$ct_short <- sub("^(\\d+ \\S+ \\S+).*", "\\1", top$celltype)
top$ct_short <- factor(top$ct_short, levels = rev(top$ct_short))

p_bar <- ggplot(top, aes(x = pearson_r, y = ct_short)) +
  geom_col(aes(fill = pearson_r > 0), width = 0.7) +
  geom_vline(xintercept = 0, color = "grey40") +
  scale_fill_manual(values = c("TRUE" = "#E63946", "FALSE" = "#4A90D9"), guide = "none") +
  geom_text(aes(label = sprintf("r=%.2f (n=%d)", pearson_r, n_genes),
                x = ifelse(pearson_r >= 0, pearson_r + 0.005, pearson_r - 0.005),
                hjust = ifelse(pearson_r >= 0, 0, 1)),
            size = 3.5, color = "grey20") +
  labs(
    x = "Pearson r (Slide-tags vs Xenium 5k logFC)",
    y = NULL,
    title = "Per-cell-type DE concordance: Slide-tags vs Xenium 5k",
    subtitle = "Top 20 by Pearson r (≥50 shared genes) | PREG vs CTRL"
  ) +
  xlim(NA, max(top$pearson_r) + 0.08) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 10, color = "grey30"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(fig_dir, "de_concordance_per_celltype_bar.png"), p_bar,
       width = 12, height = 8, dpi = 250, bg = "white")
cat("  Saved de_concordance_per_celltype_bar.png\n")

cat("\nDone!\n")
