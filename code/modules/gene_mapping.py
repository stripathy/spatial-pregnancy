"""
Gene symbol to Ensembl ID mapping utilities for the spatial pregnancy pipeline.

Provides:
  load_gene_mapping()  — build symbol -> Ensembl dict from CSV + JSON sources
  get_gene_indices()   — find query genes in reference by Ensembl lookup
"""

import json
import os
import pandas as pd


def load_gene_mapping(merfish_mapping_path=None, symbol_to_ensembl_path=None):
    """Build gene symbol -> Ensembl ID mapping.

    Loads a general symbol->ensembl JSON first, then overlays a MERFISH-panel-
    specific CSV if provided (the panel CSV achieves higher coverage for targeted
    panel genes because probes have pre-assigned Ensembl IDs).

    Parameters
    ----------
    merfish_mapping_path : str or None
        Path to MERFISH panel mapping CSV with 'Gene Symbol' and 'Gene ID ' columns.
        Overrides entries from the general JSON when both are present.
    symbol_to_ensembl_path : str or None
        Path to gene_symbol_to_ensembl_mouse.json (symbol -> Ensembl string).

    Returns
    -------
    mapping : dict[str, str]
        Gene symbol -> Ensembl ID (e.g. {'Actb': 'ENSMUSG00000029580'}).
    """
    mapping = {}

    if symbol_to_ensembl_path and os.path.exists(symbol_to_ensembl_path):
        with open(symbol_to_ensembl_path) as f:
            mapping.update(json.load(f))

    if merfish_mapping_path and os.path.exists(merfish_mapping_path):
        df = pd.read_csv(merfish_mapping_path)
        for _, row in df.iterrows():
            gene_sym = row['Gene Symbol']
            gene_id = str(row['Gene ID ']).strip()
            if gene_sym and gene_id:
                mapping[gene_sym] = gene_id

    return mapping


def get_gene_indices(query_genes, ref_gene_ensembl, gene_mapping):
    """Find indices of query genes in the reference gene list via Ensembl ID lookup.

    Parameters
    ----------
    query_genes : list of str
        Gene symbols in the query dataset (adata.var_names).
    ref_gene_ensembl : list of str
        Ensembl IDs for each column of the reference matrix.
    gene_mapping : dict[str, str]
        Gene symbol -> Ensembl ID (from load_gene_mapping).

    Returns
    -------
    gene_order : list of str
        Query gene symbols that were matched in the reference (in match order).
    ref_indices : list of int
        Corresponding column indices in ref_gene_ensembl / reference matrix.
    """
    ens2idx = {g: i for i, g in enumerate(ref_gene_ensembl)}

    gene_order = []
    ref_indices = []
    for gene in query_genes:
        ens = gene_mapping.get(gene)
        if ens and ens in ens2idx:
            gene_order.append(gene)
            ref_indices.append(ens2idx[ens])

    return gene_order, ref_indices
