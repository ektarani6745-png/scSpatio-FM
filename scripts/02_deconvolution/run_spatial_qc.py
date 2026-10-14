#!/usr/bin/env python3
"""
Step 2: Spatial QC, Normalization & Feature Engineering
Input: data/raw/ (H5 matrix + spatial image folder)
Outputs:
  - results/qc/qc_violin_metrics.png
  - results/qc/spatial_counts_and_genes.png
  - data/processed/lymph_node_processed.h5ad
"""

import os
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

RAW_DIR = "../../data/raw"
PROCESSED_DIR = "../../data/processed"
QC_RESULTS_DIR = "../../results/qc"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(QC_RESULTS_DIR, exist_ok=True)

print("[INFO] Loading 10x Visium dataset from raw directory...")
adata = sq.read.visium(
    path=RAW_DIR,
    counts_file="V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5"
)
adata.var_names_make_unique()

print(f"[INFO] Raw dataset loaded: {adata.n_obs} spots across {adata.n_vars} genes.")

# 1. Mitochondrial QC Metrics
adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# 2. QC Violin Plots
print("[INFO] Generating QC metric violin plots...")
plt.figure(figsize=(10, 4))
sc.pl.violin(
    adata,
    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    jitter=0.4,
    multi_panel=True,
    show=False
)
plt.savefig(os.path.join(QC_RESULTS_DIR, "qc_violin_metrics.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Spatial Feature Maps
print("[INFO] Rendering spatial distribution maps on tissue...")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sq.pl.spatial_scatter(adata, color="total_counts", ax=axs[0], title="Total UMI Counts", cmap="viridis")
sq.pl.spatial_scatter(adata, color="n_genes_by_counts", ax=axs[1], title="Detected Genes", cmap="plasma")
plt.tight_layout()
plt.savefig(os.path.join(QC_RESULTS_DIR, "spatial_counts_and_genes.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Quality Filtering
print("[INFO] Performing quality filtering...")
sc.pp.filter_cells(adata, min_counts=500)
sc.pp.filter_genes(adata, min_cells=10)

# 5. Normalization, Log-transform & HVG Selection (standard Seurat flavor)
print("[INFO] Normalizing library sizes and detecting HVGs...")
adata.raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# 6. Save Processed AnnData
output_path = os.path.join(PROCESSED_DIR, "lymph_node_processed.h5ad")
print(f"[INFO] Saving processed AnnData to {output_path}...")
adata.write_h5ad(output_path)
print("[SUCCESS] Step 2 QC and preprocessing complete.")
