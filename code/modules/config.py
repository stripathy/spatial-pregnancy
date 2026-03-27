"""Shared constants for the spatial pregnancy analysis pipeline."""

# ── Pseudobulk / DE ───────────────────────────────────────────────────────
MIN_CELLS_PER_PB: int = 10
MIN_SAMPLES_PER_GROUP: int = 2
MIN_GENES_AFTER_FILTER: int = 10

# ── Correlation classifier ────────────────────────────────────────────────
TOP_N_EXEMPLARS: int = 200
MIN_CELLS_PASS2: int = 20
CORR_CHUNK_SIZE: int = 10000

# ── Compositional analysis ────────────────────────────────────────────────
PRESENCE_THRESHOLD: float = 0.5

# ── Expression normalization ──────────────────────────────────────────────
RANDOM_SEED: int = 42
NORMALIZE_TARGET_SUM: float = 1e4

# ── Cell type biology ─────────────────────────────────────────────────────
NEURONAL_KEYWORDS: tuple = ('Glut', 'GABA', 'Dopa', 'Sero', 'Gnrh1')
