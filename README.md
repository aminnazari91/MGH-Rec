# MGH-Rec

# Multi-Task Heterogeneous Graph Learning with Cross-Attention Fusion for Robust Recommendation

This repository contains the official implementation of the paper:
“A Multi-Task Heterogeneous Graph Learning with Cross-Attention Fusion for Robust Recommendation”

# 🧠 Overview

This work introduces a multi-task heterogeneous graph learning framework for recommendation systems that jointly addresses:

- Data sparsity
- Cold-start problem
- Negative transfer between tasks
- Noisy interactions in user–item graphs

We propose a dual-branch architecture:

- GCN-based branch for rating prediction (recommendation task)
- HGT-based branch for node classification (semantic understanding)
- Cross-attention module for adaptive fusion of representations

Additionally, we enrich MovieLens interaction data using IMDb metadata to construct a heterogeneous graph with semantic entities (actors, directors, relations).

# ⚙️ Key Features
- Multi-task learning with independent graph encoders
- Heterogeneous graph construction (MovieLens + IMDb)
- GCN for interaction modeling
- HGT for semantic representation learning
- Cross-attention fusion mechanism
- Support for 100K and 1M MovieLens datasets

# Architecture
The proposed framework operates in three main stages:
- Graph Construction
- User–item interaction graph from MovieLens
- Semantic enrichment using IMDb metadata (actors, directors, relations)
- Dual-Branch Representation Learning
- GCN-based recommender branch
- HGT-based classification branch
- Cross-Attention Fusion
- Adaptive feature integration

# Datasets
- MovieLens (rating data)
- IMDb (knowledge graph/metadata)

# 🚀 How to Run

1. Install dependencies
      pip install -r requirements 
2. Run main experiment (100K dataset)
      Run_100k_Proposed.ipynb
بخق 
# Citation
If you use this code, please cite:

@article{nazari2026multi,
  title={A multi-task heterogeneous graph learning with cross-attention fusion for robust recommendation},
  author={Nazari, Amin and Mansoorizadeh, Muharram and Khotanlou, Hassan},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
