#!/usr/bin/env python3
"""
Step 5: GNN-based Spatial Clustering and Visualization
Input: data/processed/lymph_node_processed_gnn.h5ad
Output: results/spatial_plots/gnn_leiden_clusters.png
"""
import os
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

PROCESSED_DIR = "../../data/processed"
PLOT_DIR = "../../results/spatial_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

print("[INFO] Loading GNN-smoothed spatial data...")
adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "lymph_node_processed_gnn.h5ad"))

print("[INFO] Computing neighborhood graph from GNN latent space...")
sc.pp.neighbors(adata, use_rep="X_gnn", n_neighbors=15)

print("[INFO] Performing Leiden community detection (igraph backend)...")
# Using the modern, faster igraph flavor to avoid warnings
sc.tl.leiden(adata, resolution=0.5, key_added="gnn_clusters", flavor="igraph", n_iterations=2, directed=False)

print("[INFO] Computing UMAP...")
sc.tl.umap(adata)

print("[INFO] Rendering publication-grade spatial plots...")
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(adata, color="gnn_clusters", ax=axs[0], title="UMAP of GNN Embeddings", show=False, palette="tab20")
sq.pl.spatial_scatter(
    adata, 
    color="gnn_clusters", 
    ax=axs[1], 
    title="scSpatio-FM: Tissue Microenvironments", 
    palette="tab20", 
    size=1.2
)

plt.tight_layout()
output_file = os.path.join(PLOT_DIR, "gnn_leiden_clusters.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"[SUCCESS] Pipeline complete! Final visualizations saved to {output_file}")
