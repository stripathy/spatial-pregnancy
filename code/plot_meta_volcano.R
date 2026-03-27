library(ggplot2)
library(ggrepel)
library(dplyr)

library(here)
df <- read.csv(here::here("output/crumblr/meta_analysis_preg_vs_ctrl.csv"))

# Clean labels
df$label <- gsub("^\\d+\\s+", "", df$celltype)
df$neg_log10_fdr <- -log10(df$fdr_combined)
df$neg_log10_p <- -log10(df$p_combined)

# Significance categories
df$sig <- case_when(
  df$fdr_combined < 0.05 ~ "FDR < 0.05",
  df$fdr_combined < 0.10 ~ "FDR < 0.10",
  df$fdr_combined < 0.20 ~ "FDR < 0.20",
  TRUE ~ "NS"
)
df$sig <- factor(df$sig, levels = c("FDR < 0.05", "FDR < 0.10", "FDR < 0.20", "NS"))

# Label FDR < 0.20 or top 10
df$show_label <- (df$fdr_combined < 0.20) | (rank(df$p_combined) <= 10)

# Direction annotation
df$direction <- ifelse(df$mean_logFC > 0, "Increased in PREG", "Decreased in PREG")

# Concordance annotation
df$concordance_label <- ifelse(df$concordant, "Concordant (3/3)",
                               paste0("Discordant (", df$n_agree, "/3)"))

p <- ggplot(df, aes(x = mean_logFC, y = neg_log10_fdr)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_hline(yintercept = -log10(0.20), linetype = "dotted", color = "grey60", linewidth = 0.3) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
  geom_point(aes(color = sig, shape = concordance_label, size = sig), alpha = 0.8) +
  scale_color_manual(
    values = c("FDR < 0.05" = "#E63946", "FDR < 0.10" = "#F4A261",
               "FDR < 0.20" = "#E9C46A", "NS" = "#A8DADC"),
    name = "Meta-analysis FDR"
  ) +
  scale_size_manual(
    values = c("FDR < 0.05" = 5, "FDR < 0.10" = 4, "FDR < 0.20" = 3.5, "NS" = 2),
    name = "Meta-analysis FDR"
  ) +
  scale_shape_manual(
    values = c("Concordant (3/3)" = 16, "Discordant (2/3)" = 17, "Discordant (1/3)" = 4),
    name = "Cross-platform\nconcordance"
  ) +
  geom_text_repel(
    data = df %>% filter(show_label),
    aes(label = label),
    size = 3.5, max.overlaps = 25, segment.alpha = 0.4,
    box.padding = 0.5, point.padding = 0.3,
    min.segment.length = 0.1, force = 2
  ) +
  annotate("text", x = max(df$mean_logFC) * 0.8, y = -log10(0.05) + 0.15,
           label = "FDR = 0.05", color = "grey40", size = 3, hjust = 1) +
  annotate("text", x = max(df$mean_logFC) * 0.8, y = -log10(0.20) + 0.15,
           label = "FDR = 0.20", color = "grey50", size = 3, hjust = 1) +
  labs(
    x = "Meta-analysis mean logFC (PREG vs CTRL)",
    y = expression(-log[10](FDR)),
    title = "Meta-analysis of cell type proportion changes in pregnancy",
    subtitle = "Stouffer's method combining MERFISH (n=3), Slide-tags (n=3), and Xenium 5k (n=6) | Subclass level"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

ggsave("output/crumblr/figures/meta_volcano.png", p,
       width = 12, height = 8, dpi = 250, bg = "white")
cat("Saved meta_volcano.png\n")
