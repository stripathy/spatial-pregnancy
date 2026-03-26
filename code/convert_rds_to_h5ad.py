#!/usr/bin/env python3
"""
Convert Seurat RDS object (Xenium 5k data) to h5ad format.

Usage on HPC login node:
    # Load R and Python modules (adjust for your HPC)
    module load r/4.3.0 python/3.10

    # Install R dependencies (one-time)
    Rscript -e "install.packages('Matrix', repos='https://cran.r-project.org')"

    # Install Python dependencies (one-time)
    pip install --user anndata scipy pandas numpy h5py

    # Run conversion
    python3 convert_rds_to_h5ad.py \
        --input /path/to/objf_raw.rds \
        --output /path/to/xenium5k_raw.h5ad

This script:
1. Uses R (via subprocess) to extract components from the Seurat object
   into intermediate flat files (counts as MatrixMarket, metadata as CSV)
2. Reassembles them into an h5ad in Python

This avoids loading the full object in Python (no rpy2 needed) and
keeps peak memory usage to ~1x the object size (R only).
"""

import argparse
import subprocess
import tempfile
import os
import sys
import time


def run_r_extraction(rds_path, tmp_dir):
    """Run R script to extract Seurat components to flat files."""

    r_script = f"""
library(Matrix)

cat("Loading RDS file...\\n")
t0 <- Sys.time()
obj <- readRDS("{rds_path}")
cat(sprintf("Loaded in %.1f minutes\\n", difftime(Sys.time(), t0, units="mins")))

cat(sprintf("Object class: %s\\n", class(obj)))
cat(sprintf("Dimensions: %d cells x %d genes\\n", ncol(obj), nrow(obj)))

# ── Extract counts matrix ─────────────────────────────────────────────
cat("Extracting counts matrix...\\n")

# Try Seurat v5 (layers) first, fall back to v4
counts <- tryCatch({{
    # Seurat v5: counts may be in layers
    if (inherits(obj, "Seurat")) {{
        assay <- DefaultAssay(obj)
        cat(sprintf("Default assay: %s\\n", assay))

        # Try GetAssayData (works for both v4 and v5)
        mat <- GetAssayData(obj, slot = "counts")
        if (ncol(mat) == 0 || nrow(mat) == 0) {{
            # Try data slot if counts is empty
            cat("Counts slot empty, trying data slot...\\n")
            mat <- GetAssayData(obj, slot = "data")
        }}
        mat
    }} else {{
        stop("Not a Seurat object")
    }}
}}, error = function(e) {{
    cat(sprintf("Seurat extraction failed: %s\\n", e$message))
    cat("Trying direct slot access...\\n")

    # Direct slot access for various object types
    if ("assays" %in% slotNames(obj)) {{
        assay_name <- names(obj@assays)[1]
        cat(sprintf("Using assay: %s\\n", assay_name))
        assay_obj <- obj@assays[[assay_name]]

        # Seurat v5 assay
        if ("layers" %in% slotNames(assay_obj)) {{
            layer_names <- names(assay_obj@layers)
            cat(sprintf("Layers: %s\\n", paste(layer_names, collapse=", ")))
            if ("counts" %in% layer_names) {{
                return(assay_obj@layers[["counts"]])
            }} else if ("data" %in% layer_names) {{
                return(assay_obj@layers[["data"]])
            }}
        }}

        # Seurat v4 assay
        if ("counts" %in% slotNames(assay_obj)) {{
            mat <- assay_obj@counts
            if (ncol(mat) > 0) return(mat)
        }}
        if ("data" %in% slotNames(assay_obj)) {{
            return(assay_obj@data)
        }}
    }}
    stop("Could not extract expression matrix")
}})

cat(sprintf("Matrix: %d genes x %d cells, class: %s\\n",
            nrow(counts), ncol(counts), class(counts)))

# Ensure sparse
if (!inherits(counts, "dgCMatrix")) {{
    cat("Converting to sparse matrix...\\n")
    counts <- as(counts, "dgCMatrix")
}}

# Write MatrixMarket (transposed: cells x genes for anndata)
cat("Writing counts matrix (MatrixMarket)...\\n")
# WriteMM writes in gene x cell orientation; we'll transpose in Python
writeMM(counts, file = "{tmp_dir}/counts.mtx")

# Write gene names
cat("Writing gene names...\\n")
writeLines(rownames(counts), "{tmp_dir}/genes.txt")

# Write cell barcodes
cat("Writing cell barcodes...\\n")
writeLines(colnames(counts), "{tmp_dir}/barcodes.txt")

# ── Extract metadata ──────────────────────────────────────────────────
cat("Extracting metadata...\\n")
meta <- obj@meta.data
write.csv(meta, "{tmp_dir}/metadata.csv", row.names = TRUE)
cat(sprintf("Metadata: %d cells x %d columns\\n", nrow(meta), ncol(meta)))

# ── Extract spatial coordinates if available ──────────────────────────
cat("Extracting spatial coordinates...\\n")
has_spatial <- FALSE

# Check images slot (Xenium/Visium)
if ("images" %in% slotNames(obj) && length(obj@images) > 0) {{
    img_name <- names(obj@images)[1]
    cat(sprintf("Found image: %s\\n", img_name))
    coords <- GetTissueCoordinates(obj, image = img_name)
    if (!is.null(coords) && nrow(coords) > 0) {{
        write.csv(coords, "{tmp_dir}/spatial_coords.csv", row.names = TRUE)
        has_spatial <- TRUE
        cat(sprintf("Spatial coords: %d cells\\n", nrow(coords)))
    }}
}}

# Also check for x/y in metadata
if (!has_spatial) {{
    xy_cols <- intersect(c("x_centroid", "y_centroid", "x", "y",
                           "spatial_x", "spatial_y", "center_x", "center_y"),
                         colnames(meta))
    if (length(xy_cols) >= 2) {{
        cat(sprintf("Spatial coords found in metadata: %s\\n",
                    paste(xy_cols, collapse=", ")))
    }}
}}

# ── Extract embeddings if available ───────────────────────────────────
cat("Checking for embeddings...\\n")
if ("reductions" %in% slotNames(obj)) {{
    for (red_name in names(obj@reductions)) {{
        emb <- Embeddings(obj, reduction = red_name)
        if (!is.null(emb) && nrow(emb) > 0) {{
            write.csv(emb, sprintf("{tmp_dir}/embedding_%s.csv", red_name),
                      row.names = TRUE)
            cat(sprintf("Saved embedding: %s (%d x %d)\\n",
                        red_name, nrow(emb), ncol(emb)))
        }}
    }}
}}

# ── Extract variable features if available ────────────────────────────
tryCatch({{
    vf <- VariableFeatures(obj)
    if (length(vf) > 0) {{
        writeLines(vf, "{tmp_dir}/variable_features.txt")
        cat(sprintf("Variable features: %d\\n", length(vf)))
    }}
}}, error = function(e) cat("No variable features found\\n"))

cat("\\nR extraction complete!\\n")
cat(sprintf("Peak memory: %.1f GB\\n",
            as.numeric(gc()[2,6]) / 1024))
"""

    # Write R script to file
    r_script_path = os.path.join(tmp_dir, "extract.R")
    with open(r_script_path, 'w') as f:
        f.write(r_script)

    print("Running R extraction (this may take a while for large files)...")
    t0 = time.time()

    result = subprocess.run(
        ["Rscript", "--vanilla", r_script_path],
        capture_output=True, text=True,
        env={**os.environ, "R_MAX_VSIZE": "200Gb"}
    )

    elapsed = time.time() - t0
    print(f"R extraction took {elapsed/60:.1f} minutes")

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("R STDERR:", result.stderr, file=sys.stderr)
        raise RuntimeError(f"R extraction failed with exit code {result.returncode}")

    return True


def assemble_h5ad(tmp_dir, output_path):
    """Assemble extracted components into h5ad."""
    import anndata as ad
    import scipy.io
    import scipy.sparse as sparse
    import pandas as pd
    import numpy as np

    print("Assembling h5ad from extracted components...")

    # ── Load counts ───────────────────────────────────────────────────
    print("  Loading counts matrix...")
    # MatrixMarket is genes x cells; need to transpose for anndata (cells x genes)
    counts = scipy.io.mmread(os.path.join(tmp_dir, "counts.mtx"))
    counts = sparse.csc_matrix(counts).T.tocsr()  # transpose to cells x genes
    print(f"  Counts: {counts.shape[0]:,} cells x {counts.shape[1]:,} genes")

    # ── Load gene names and barcodes ──────────────────────────────────
    with open(os.path.join(tmp_dir, "genes.txt")) as f:
        genes = [line.strip() for line in f]
    with open(os.path.join(tmp_dir, "barcodes.txt")) as f:
        barcodes = [line.strip() for line in f]

    print(f"  Genes: {len(genes)}, Barcodes: {len(barcodes)}")
    assert counts.shape == (len(barcodes), len(genes)), \
        f"Shape mismatch: counts={counts.shape}, barcodes={len(barcodes)}, genes={len(genes)}"

    # ── Load metadata ─────────────────────────────────────────────────
    print("  Loading metadata...")
    meta = pd.read_csv(os.path.join(tmp_dir, "metadata.csv"), index_col=0)
    # Reindex to match barcode order (in case of mismatch)
    if not all(meta.index == barcodes):
        print("  Reindexing metadata to match barcode order...")
        meta = meta.reindex(barcodes)

    # ── Create AnnData ────────────────────────────────────────────────
    print("  Creating AnnData object...")
    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=meta,
        var=pd.DataFrame(index=genes)
    )
    adata.var['gene_symbol'] = adata.var.index

    # ── Add spatial coordinates ───────────────────────────────────────
    spatial_path = os.path.join(tmp_dir, "spatial_coords.csv")
    if os.path.exists(spatial_path):
        print("  Loading spatial coordinates...")
        coords = pd.read_csv(spatial_path, index_col=0)
        coords = coords.reindex(adata.obs.index)
        adata.obsm['spatial'] = coords.values
        # Also add as obs columns
        for col in coords.columns:
            if col not in adata.obs.columns:
                adata.obs[col] = coords[col].values

    # ── Add embeddings ────────────────────────────────────────────────
    for fname in os.listdir(tmp_dir):
        if fname.startswith("embedding_") and fname.endswith(".csv"):
            red_name = fname.replace("embedding_", "").replace(".csv", "")
            print(f"  Loading embedding: {red_name}")
            emb = pd.read_csv(os.path.join(tmp_dir, fname), index_col=0)
            emb = emb.reindex(adata.obs.index)
            adata.obsm[f'X_{red_name}'] = emb.values.astype(np.float32)

    # ── Mark variable features ────────────────────────────────────────
    vf_path = os.path.join(tmp_dir, "variable_features.txt")
    if os.path.exists(vf_path):
        with open(vf_path) as f:
            vf = set(line.strip() for line in f)
        adata.var['highly_variable'] = adata.var.index.isin(vf)
        print(f"  Marked {sum(adata.var['highly_variable'])} variable features")

    # ── Save ──────────────────────────────────────────────────────────
    print(f"  Saving to {output_path}...")
    adata.write(output_path)

    file_size = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes ({file_size:.1f} GB)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  === Summary ===")
    print(f"  Cells: {adata.shape[0]:,}")
    print(f"  Genes: {adata.shape[1]:,}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    if adata.obsm:
        print(f"  obsm: {list(adata.obsm.keys())}")

    # Sample/condition info
    for col in ['sample', 'orig.ident', 'condition', 'group']:
        if col in adata.obs.columns:
            print(f"\n  {col}:")
            for v, n in adata.obs[col].value_counts().items():
                print(f"    {v}: {n:,}")

    return adata


def main():
    parser = argparse.ArgumentParser(
        description="Convert Seurat RDS to h5ad via R extraction + Python assembly")
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input .rds file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output .h5ad file')
    parser.add_argument('--tmp-dir', default=None,
                        help='Temporary directory for intermediate files '
                             '(default: auto-created, cleaned up after)')
    parser.add_argument('--keep-tmp', action='store_true',
                        help='Keep temporary files after conversion')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check R is available
    try:
        r_version = subprocess.run(
            ["Rscript", "--version"], capture_output=True, text=True)
        print(f"R: {r_version.stderr.strip()}")
    except FileNotFoundError:
        print("Error: Rscript not found. Load R module first.", file=sys.stderr)
        sys.exit(1)

    # Create temp directory
    if args.tmp_dir:
        tmp_dir = args.tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="rds_convert_")

    print(f"Temp directory: {tmp_dir}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    t0 = time.time()

    try:
        # Step 1: R extraction
        run_r_extraction(args.input, tmp_dir)

        # Step 2: Python assembly
        assemble_h5ad(tmp_dir, args.output)

        elapsed = time.time() - t0
        print(f"\nTotal conversion time: {elapsed/60:.1f} minutes")

    finally:
        if not args.keep_tmp and not args.tmp_dir:
            import shutil
            print(f"Cleaning up {tmp_dir}...")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif args.keep_tmp:
            print(f"Intermediate files kept in: {tmp_dir}")


if __name__ == "__main__":
    main()
