#!/usr/bin/env python3
"""
Step 4: Spatial GNN with C++ Accelerated Edge Construction
Input: data/processed/lymph_node_processed_embedded.h5ad
Output: results/gnn_embeddings/gnn_smoothed_embeddings.npy
"""
import os
import scanpy as sc
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.utils.cpp_extension import load_inline
import numpy as np

PROCESSED_DIR = "../../data/processed"
EMBED_DIR = "../../results/gnn_embeddings"

print("[INFO] Loading embedded spatial data...")
adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "lymph_node_processed_embedded.h5ad"))
coords = adata.obsm["spatial"]
fm_embeddings = adata.obsm["X_fm"]

# ==========================================
# 1. THE C++ KERNEL (Fast Spatial Adjacency)
# ==========================================
print("[INFO] Compiling C++ Spatial Edge Kernel (this may take 10-20 seconds the first time)...")

cpp_source = """
#include <torch/extension.h>
#include <vector>
#include <cmath>

// Computes edges based on Euclidean distance threshold
std::vector<torch::Tensor> compute_spatial_edges(torch::Tensor coords, float threshold) {
    auto coords_acc = coords.accessor<float, 2>();
    int num_nodes = coords.size(0);
    
    std::vector<int64_t> src, dst;
    
    for (int i = 0; i < num_nodes; i++) {
        for (int j = i + 1; j < num_nodes; j++) {
            float dx = coords_acc[i][0] - coords_acc[j][0];
            float dy = coords_acc[i][1] - coords_acc[j][1];
            float dist = std::sqrt(dx*dx + dy*dy);
            
            if (dist < threshold) {
                src.push_back(i); dst.push_back(j);
                src.push_back(j); dst.push_back(i); // Undirected graph
            }
        }
    }
    
    auto edge_index = torch::empty({2, static_cast<int64_t>(src.size())}, torch::kInt64);
    for (size_t k = 0; k < src.size(); k++) {
        edge_index[0][k] = src[k];
        edge_index[1][k] = dst[k];
    }
    return {edge_index};
}
"""

# Compile the C++ code dynamically
spatial_cpp = load_inline(
    name='spatial_cpp',
    cpp_sources=cpp_source,
    functions=['compute_spatial_edges'],
    verbose=False
)

# Execute C++ function
coords_tensor = torch.FloatTensor(coords)
# Distance threshold for Visium spots (usually physical adjacent spots are ~150-200 units apart in pixel space)
edge_index = spatial_cpp.compute_spatial_edges(coords_tensor, 250.0)[0]
print(f"[INFO] C++ Kernel generated {edge_index.shape[1]} physical edges.")

# ==========================================
# 2. THE PYTORCH GEOMETRIC GRAPH NETWORK
# ==========================================
class SpatialGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device("cpu")
model = SpatialGNN(in_channels=256, hidden_channels=128, out_channels=64).to(device)
x_tensor = torch.FloatTensor(fm_embeddings).to(device)
edge_index = edge_index.to(device)

print("[INFO] Passing Foundation Embeddings through Spatial GNN...")
model.eval()
with torch.no_grad():
    gnn_embeddings = model(x_tensor, edge_index).numpy()

print("[INFO] Saving final GNN smoothed embeddings...")
np.save(os.path.join(EMBED_DIR, "gnn_smoothed_embeddings.npy"), gnn_embeddings)

adata.obsm["X_gnn"] = gnn_embeddings
adata.write_h5ad(os.path.join(PROCESSED_DIR, "lymph_node_processed_gnn.h5ad"))

print(f"[SUCCESS] Spatial-GNN complete. Final latent shape: {gnn_embeddings.shape}")
