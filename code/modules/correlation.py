"""
Shared correlation functions for the spatial pregnancy analysis pipeline.

Provides:
  correlate()                  — chunked Pearson correlation, cells vs centroids
  assign_labels()              — argmax assignment with confidence + margin
  build_centroids_from_labels() — mean log-normed centroids from labeled cells
  normalize_expr()             — CPM 10k + log1p normalization
  spatial_coherence()          — k-NN label purity (per-cell array)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scanpy as sc
from scipy.spatial import KDTree

from .config import CORR_CHUNK_SIZE, NORMALIZE_TARGET_SUM, RANDOM_SEED, TOP_N_EXEMPLARS


def correlate(query_expr, centroids, chunk_size=CORR_CHUNK_SIZE, verbose=True):
    """Chunked Pearson correlation between cells and centroids.

    Z-scores each row, then dot product / n_genes.

    Parameters
    ----------
    query_expr : np.ndarray, shape (n_cells, n_genes)
        Normalized expression (already log-normed or rank-transformed).
    centroids : pd.DataFrame, shape (n_types, n_genes)
        Centroid expression; index = type names.
    chunk_size : int
        Number of cells to process at once (controls memory).
    verbose : bool
        Print progress every 50k cells.

    Returns
    -------
    corr_matrix : np.ndarray, shape (n_cells, n_types), float32
    type_names : list of str
    """
    type_names = list(centroids.index)
    centroid_arr = centroids.values.astype(np.float64)
    centroid_arr = np.nan_to_num(centroid_arr, nan=0.0)

    n_cells, n_genes = query_expr.shape

    c_mean = centroid_arr.mean(axis=1, keepdims=True)
    c_std = centroid_arr.std(axis=1, keepdims=True, ddof=0)
    c_std[c_std == 0] = 1.0
    c_norm = (centroid_arr - c_mean) / c_std

    corr_matrix = np.zeros((n_cells, len(type_names)), dtype=np.float32)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = query_expr[start:end].astype(np.float64)

        q_mean = chunk.mean(axis=1, keepdims=True)
        q_std = chunk.std(axis=1, keepdims=True, ddof=0)
        q_std[q_std == 0] = 1.0
        q_norm = (chunk - q_mean) / q_std
        q_norm = np.nan_to_num(q_norm, nan=0.0)

        corr_matrix[start:end] = ((q_norm @ c_norm.T) / n_genes).astype(np.float32)

        if verbose and (end % 50000 == 0 or end == n_cells):
            print(f"    Correlated {end:,}/{n_cells:,} cells", flush=True)

    return corr_matrix, type_names


def assign_labels(corr_matrix, type_names):
    """Assign best-match labels from correlation matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray, shape (n_cells, n_types)
    type_names : list of str, length n_types

    Returns
    -------
    labels : np.ndarray of str, shape (n_cells,)
    best_corr : np.ndarray, float32, shape (n_cells,)
    margin : np.ndarray, float32, shape (n_cells,) — best minus second-best correlation
    """
    sorted_corr = np.sort(corr_matrix, axis=1)
    best_idx = np.argmax(corr_matrix, axis=1)
    labels = np.array([type_names[i] for i in best_idx])
    best_corr = sorted_corr[:, -1]
    second_corr = (sorted_corr[:, -2] if corr_matrix.shape[1] >= 2
                   else np.zeros_like(best_corr))
    margin = best_corr - second_corr
    return labels, best_corr.astype(np.float32), margin.astype(np.float32)


def build_centroids_from_labels(adata, subclass_col='subclass', confidence_col=None,
                                 top_n=TOP_N_EXEMPLARS, random_seed=RANDOM_SEED):
    """Build per-subclass mean expression centroids from labeled cells.

    Selects up to top_n exemplar cells per subclass (by confidence if available,
    otherwise random), normalizes to CPM 10k + log1p, and computes mean expression.

    Parameters
    ----------
    adata : AnnData
        Raw counts in .X.
    subclass_col : str
        Column in adata.obs with subclass labels.
    confidence_col : str or None
        Column with confidence scores; selects highest-confidence cells if provided.
    top_n : int
        Maximum cells per subclass to use for centroid computation.
    random_seed : int
        Random seed for reproducible cell sampling when confidence_col is None.

    Returns
    -------
    centroids : pd.DataFrame, shape (n_subclasses, n_genes)
    cell_counts : dict mapping subclass name -> number of exemplar cells
    gene_names : list of str
    """
    labels = adata.obs[subclass_col].astype(str).values
    unique_labels = sorted(set(labels))
    rng = np.random.RandomState(random_seed)

    exemplar_indices = []
    cell_counts = {}

    for lab in unique_labels:
        lab_mask = np.where(labels == lab)[0]
        n_available = len(lab_mask)
        if n_available == 0:
            continue

        n_use = min(top_n, n_available)
        if confidence_col and confidence_col in adata.obs.columns:
            conf = adata.obs[confidence_col].values[lab_mask].astype(float)
            top_idx = lab_mask[np.argsort(conf)[-n_use:]]
        else:
            top_idx = rng.choice(lab_mask, n_use, replace=False)

        exemplar_indices.append(top_idx)
        cell_counts[lab] = n_use

    all_idx = np.concatenate(exemplar_indices)
    adata_ex = adata[all_idx].copy()

    sc.pp.normalize_total(adata_ex, target_sum=NORMALIZE_TARGET_SUM)
    sc.pp.log1p(adata_ex)

    X = adata_ex.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    gene_names = list(adata_ex.var_names)
    ex_labels = adata_ex.obs[subclass_col].astype(str).values

    centroids_dict = {}
    for lab in sorted(cell_counts.keys()):
        lab_mask = ex_labels == lab
        centroids_dict[lab] = X[lab_mask].mean(axis=0)

    centroids = pd.DataFrame(centroids_dict, index=gene_names).T
    print(f"  Built {len(centroids)} subclass centroids (top-{top_n} exemplars each)")
    return centroids, cell_counts, gene_names


def normalize_expr(adata, target_sum=NORMALIZE_TARGET_SUM):
    """CPM normalization + log1p; returns dense float32 array.

    Parameters
    ----------
    adata : AnnData
        Raw counts in .X. A copy is made internally; adata is not modified.
    target_sum : float
        Normalization target (default 1e4 = CPM 10k).

    Returns
    -------
    X : np.ndarray, shape (n_cells, n_genes), float32
    """
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    return X.astype(np.float32)


def spatial_coherence(labels, coords, k=20):
    """Per-cell spatial coherence: fraction of k nearest neighbours sharing the same label.

    Parameters
    ----------
    labels : array-like, shape (n_cells,)
        Cell type labels.
    coords : array-like, shape (n_cells, 2)
        Spatial coordinates (x, y).
    k : int
        Number of neighbours to query (excluding the cell itself).

    Returns
    -------
    coherence : np.ndarray, float, shape (n_cells,)
        Per-cell coherence scores in [0, 1]. Call `.mean()` for an overall scalar.
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    kdt = KDTree(coords)
    _, nn_idx = kdt.query(coords, k=k + 1)
    nn_idx = nn_idx[:, 1:]
    nn_labels = labels[nn_idx]
    return (nn_labels == labels[:, None]).mean(axis=1)
