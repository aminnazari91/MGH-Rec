# MGH-Rec

# Multi-Task Heterogeneous Graph Learning with Cross-Attention Fusion for Robust Recommendation

This repository contains the official implementation of the paper:
# “A Multi-Task Heterogeneous Graph Learning with Cross-Attention Fusion for Robust Recommendation”

# Overview

This work proposes a multi-task heterogeneous graph learning framework for recommender systems that addresses key challenges such as data sparsity, cold-start problems, and task interference. The model integrates interaction data from MovieLens with semantic knowledge from IMDb to construct a rich heterogeneous graph and learns robust representations through a dual-branch architecture.

# The framework consists of:
A GCN-based branch for recommendation (rating prediction)

A Heterogeneous Graph Transformer (HGT)-based branch for node classification

A cross-attention mechanism for adaptive fusion of task-specific representations

An end-to-end fine-tuning stage for joint optimization

# Key Features
Multi-task learning with task-specific graph encoders
Heterogeneous graph construction using MovieLens + IMDb data
Independent pre-training to reduce negative task interference
Bidirectional cross-attention for feature fusion
Improved robustness against sparsity and noisy interactions

# Architecture
The proposed framework operates in three main stages:
Graph Construction
User–item interaction graph from MovieLens
Semantic enrichment using IMDb metadata (actors, directors, relations)
Dual-Branch Representation Learning
GCN-based recommender branch
HGT-based classification branch
Cross-Attention Fusion
Adaptive feature integration

# Datasets
MovieLens (rating data)
IMDb (knowledge graph/metadata)

# Citation
If you use this code, please cite:
@article{nazari2026multi,
  title={A multi-task heterogeneous graph learning with cross-attention fusion for robust recommendation},
  author={Nazari, Amin and Mansoorizadeh, Muharram and Khotanlou, Hassan},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
