# scSpatio-FM 🧬💻
**Multimodal Spatial Transcriptomics Imputation & Cell-Cell Communication via Foundation Models**

![Nextflow](https://img.shields.io/badge/Orchestration-Nextflow-23aa62.svg)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C.svg)
![C++](https://img.shields.io/badge/Acceleration-C++-00599C.svg)
![Scanpy](https://img.shields.io/badge/Spatial-Scanpy%2FSquidpy-blue.svg)

## 📌 Abstract
**scSpatio-FM** is a state-of-the-art computational biology pipeline designed to process, embed, and impute spatial transcriptomics data. Standard Graph Neural Networks (GNNs) often bottleneck when processing whole-tissue spatial graphs. This architecture solves this by leveraging **Single-Cell Foundation Model Embeddings** as node features, coupled with a **Custom C++ PyBind11 Kernel** for blazing-fast spatial adjacency computations.

## 🏗️ Algorithmic Architecture
The pipeline is fully automated and segmented into 5 highly optimized modules:
1. **Orchestration (`main.nf`)**: Nextflow-driven automated fetching and unpacking of 10x Genomics Visium high-resolution tissue images and feature matrices.
2. **Spatial QC (`run_spatial_qc.py`)**: Sparse gene filtering, log1p normalization, and mitochondrial QC using `scanpy` and `squidpy`.
3. **Foundation Model Projection (`run_foundation_embedding.py`)**: High-dimensional Highly Variable Genes (HVGs) are compressed into a 256-dimensional semantic latent space using a PyTorch neural encoder.
4. **C++ Accelerated Spatial GNN (`run_spatial_gnn.py`)**: 
   - Dynamically compiles a native `C++` extension to compute Euclidean distance edges across thousands of physical tissue spots in milliseconds.
   - Passes the foundation embeddings through a `GraphSAGE` message-passing network to physically smooth the latent representations.
5. **Tissue Microenvironment Clustering (`run_visualization.py`)**: Employs the `igraph` backend for high-speed Leiden community detection, projecting the GNN clusters directly onto the H&E tissue histology.

## 🚀 Execution & Reproducibility
The environment relies on a strict Python 3.12+ `.venv` to prevent PEP 668 system conflicts.

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/scSpatio-FM.git](https://github.com/YOUR_USERNAME/scSpatio-FM.git)
cd scSpatio-FM

# 2. Initialize Environment
python3 -m venv .venv
source .venv/bin/activate
pip install torch torch_geometric pybind11 scanpy squidpy igraph leidenalg

# 3. Trigger Orchestration
cd scripts/01_orchestration
./download_data.sh
📊 Results & Output

The network successfully captures complex spatial microenvironments, effectively mapping the deep-learning latent space back onto the physical tissue architecture.
Pipeline Metrics:

    Raw Data: 4,026 physical spots across 33,538 genes.

    Foundation Space: Compressed to 256 dimensions using a PyTorch GELU encoder.

    Graph Topology: 46,316 physical tissue edges generated via C++ PyBind11 kernel in < 1 second.

    GNN Latent Space: Smoothed 64-dimensional graph embeddings.

Spatial Tissue Clustering

Below is the result of mapping the GNN-derived Leiden clusters directly onto the 10x Genomics Visium H&E tissue image, revealing distinct biological microenvironments based on their foundation embeddings:

