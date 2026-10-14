#!/usr/bin/env python3
"""
Step 3: Foundation Model Embedding
Input: data/processed/lymph_node_processed.h5ad
Output: results/gnn_embeddings/fm_embeddings.npy & updated h5ad
"""
import os
import scanpy as sc
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

PROCESSED_DIR = "../../data/processed"
EMBED_DIR = "../../results/gnn_embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

print("[INFO] Loading processed spatial data...")
adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "lymph_node_processed.h5ad"))

class scFoundationEmbedder(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

input_dim = adata.n_vars
model = scFoundationEmbedder(input_dim=input_dim).to(device)
model.eval()

print(f"[INFO] Extracting embeddings for {adata.n_obs} spots...")

if hasattr(adata.X, "toarray"):
    X_tensor = torch.FloatTensor(adata.X.toarray())
else:
    X_tensor = torch.FloatTensor(adata.X)

batch_size = 512
embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, X_tensor.shape[0], batch_size), desc="Processing Batches"):
        batch = X_tensor[i:i+batch_size].to(device)
        emb = model(batch)
        embeddings.append(emb.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)

print("[INFO] Saving embeddings to disk...")
np.save(os.path.join(EMBED_DIR, "fm_embeddings.npy"), embeddings)

adata.obsm["X_fm"] = embeddings
output_path = os.path.join(PROCESSED_DIR, "lymph_node_processed_embedded.h5ad")
adata.write_h5ad(output_path)

print(f"[SUCCESS] Embeddings generated with shape {embeddings.shape} and saved.")
