# Lesson 8: Advanced Topics & Applications - Complete Index

## Quick Navigation

### Textbook Sections (lesson_08_textbook.md)

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| 1.1-1.6 | Heterogeneous Graphs | Node types, edge types, meta-paths, HAN, HGT |
| 2.1-2.5 | Temporal GNNs | Dynamic graphs, snapshots, Neural ODE, TGN |
| 3.1-3.4 | Graph Generation | GraphRNN, Graph VAE, Diffusion, Evaluation |
| 4.1-4.5 | Knowledge Graphs | Triples, TransE, DistMult, Link prediction |
| 5.1-5.3 | Real-World Apps | Molecules, Social networks, Recommendations |
| 6.1-6.6 | Research Trends | Transformers, Equivariance, Scalability, Robustness |
| 7.1-7.7 | Best Practices | Data prep, Training, Tuning, Evaluation |

### Notebook Projects (lesson_08_notebook.ipynb)

| Project | Implementation | Classes | Scale |
|---------|----------------|---------|-------|
| 1 | Heterogeneous Graphs | HeterogeneousGraph, SimpleHeterogeneousGNN | 11 nodes, 3 types |
| 2 | Temporal Graphs | TemporalGraph, TemporalGraphForecaster | 5 timesteps, 4 nodes |
| 3 | Knowledge Graphs | KnowledgeGraph, TransEModel | 13 facts, 6 entities |
| 4 | Molecular Graphs | MolecularGNN | 20 molecules, 5-9 atoms each |
| 5 | Recommendations | RecommendationGNN | 20 users, 30 items, 150 interactions |
| 6 | Link Prediction | LinkPredictionGNN | 50 papers, ~120 citations |

### Key Implementations

#### Classes Defined
```
- HeterogeneousGraph
- SimpleHeterogeneousGNN
- TemporalGraph
- TemporalGraphForecaster
- KnowledgeGraph
- TransEModel
- MolecularGNN
- RecommendationGNN
- LinkPredictionGNN
```

#### Algorithms
```
- Meta-path discovery (DFS)
- Temporal memory updates (RNN-like)
- Embedding-based scoring
- Message passing aggregation
- Collaborative filtering
- Link probability prediction
```

#### Evaluation Metrics
```
- Embedding convergence tracking
- MAE/RMSE (recommendations)
- AUC-ROC (link prediction)
- Average Precision (ranking)
- Coverage (recommendations)
- Sparsity metrics
```

---

## Learning Paths

### Path 1: Beginner (8-10 hours)
```
Day 1:
  □ Read textbook Parts 1-2 (2h)
  □ Run notebook Projects 1-2 (1.5h)
  
Day 2:
  □ Complete Exercise 1-2 (2.5h)
  □ Review and experiment (2h)
```

### Path 2: Intermediate (12-15 hours)
```
Week 1:
  □ Read textbook Parts 1-5 (4h)
  □ Run notebook Projects 1-5 (3h)
  
Week 2:
  □ Complete Exercises 1-5 (5h)
  □ Deep dive into one project (3h)
```

### Path 3: Advanced (15-20 hours)
```
Week 1-2:
  □ Read all textbook parts (5h)
  □ Run all projects (4h)
  
Week 3:
  □ Complete all exercises (6h)
  □ Extend projects with PyTorch Geometric (5h)
```

---

## Content by Topic

### Heterogeneous Networks
- **Textbook**: Sections 1.1-1.6 (p. 1-10)
- **Notebook**: Project 1 (cells 1-15)
- **Exercise**: Exercise 1

### Temporal Dynamics
- **Textbook**: Sections 2.1-2.5 (p. 11-17)
- **Notebook**: Project 2 (cells 16-25)
- **Exercise**: Exercise 2

### Knowledge Base Completion
- **Textbook**: Sections 4.1-4.5 (p. 26-32)
- **Notebook**: Project 3 (cells 26-40)
- **Exercise**: Exercise 3

### Molecular Design
- **Textbook**: Section 5.1 (p. 40-44)
- **Notebook**: Project 4 (cells 41-55)
- **Exercise**: Exercise 4

### Recommendation Systems
- **Textbook**: Section 5.3 (p. 48-53)
- **Notebook**: Project 5 (cells 56-75)
- **Exercise**: Exercise 5

### Link Prediction
- **Textbook**: Section 4.1-4.5 (p. 26-32)
- **Notebook**: Project 6 (cells 76-95)
- **Baseline Methods**: Common neighbors, Jaccard similarity

---

## Key Equations & Formulas

### TransE Scoring
```
Score(h, r, t) = ||h + r - t||₂
```

### HAN Attention
```
a_ij^P = exp(LeakyReLU(W_att^T [h_i || h_j])) / Σ_k exp(...)
```

### Message Passing
```
aggregated = Mean({h_neighbor for neighbor in N(v)})
h_v_new = tanh(h_v + W @ aggregated)
```

### Collaborative Filtering
```
rating_pred = 2.5 + 2.5 * tanh(u_emb · i_emb)
```

---

## Real-World Applications Covered

### Molecular Graphs
- **Task**: Property prediction (solubility, toxicity)
- **Dataset**: QM9 (mentioned), synthetic examples
- **Applications**: Drug discovery, material design

### Social Networks
- **Task**: Link prediction, friend recommendation, detection
- **Scale**: Facebook (3B users), Twitter, LinkedIn
- **Applications**: Network analysis, influence propagation

### Recommendation Systems
- **Task**: Top-k recommendation, cold-start handling
- **Scale**: Netflix (200M+ users), Amazon, Spotify
- **Applications**: E-commerce, content discovery

### Knowledge Bases
- **Task**: Entity-relation completion
- **Datasets**: Freebase (45M entities), YAGO, Wikidata, DBpedia
- **Applications**: Search, reasoning, QA systems

---

## Code Organization

### Import Structure
```python
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt, seaborn as sns
# Optional: torch, torch_geometric (for extensions)
```

### Class Hierarchy
```
HeterogeneousGraph (container)
  └─ SimpleHeterogeneousGNN (inference)

TemporalGraph (container)
  └─ TemporalGraphForecaster (prediction)

KnowledgeGraph (container)
  └─ TransEModel (embedding & learning)

MolecularGNN (standalone)

RecommendationGNN (standalone)

LinkPredictionGNN (standalone)
```

---

## Hyperparameter Reference

### Model Parameters
| Model | Embedding Dim | Num Layers | Learning Rate |
|-------|---|---|---|
| HeterogeneousGNN | 8-16 | 2 | - |
| TemporalForecaster | 4-16 | - | - |
| TransE | 10-20 | - | 0.01 |
| MolecularGNN | 8-16 | 2 | - |
| RecommendationGNN | 8-16 | 2 | - |
| LinkPredictionGNN | 16 | 3 | - |

### Training Parameters
| Parameter | Range | Default |
|---|---|---|
| Epochs | 10-100 | 50 |
| Batch Size | 32-512 | Full batch |
| Dropout | 0.0-0.8 | 0.5 |
| Weight Decay | 1e-6 to 1e-2 | None |
| Learning Rate | 1e-4 to 1e-1 | 0.01 |

---

## Dataset Statistics

### Project Datasets
| Project | Nodes | Edges | Features | Labels |
|---------|-------|-------|----------|--------|
| 1 | 11 | 12 | Implicit | - |
| 2 | 4 | ~8/timestep | Implicit | - |
| 3 | 6 | 13 | - | - |
| 4 | 100 (atoms) | 150 (bonds) | Atomic | Synthetic |
| 5 | 50 | 150 | Implicit | Ratings |
| 6 | 50 | ~120 | - | Binary |

---

## Exercise Solutions Overview

### Exercise 1: Heterogeneous Graph Classification
**Hint**: Extract features from neighborhoods using meta-paths
**Key Step**: Create node representations from structural role

### Exercise 2: Temporal Graph Analysis
**Hint**: Track density, centrality, and edge patterns over time
**Key Metric**: Network diameter, clustering coefficient evolution

### Exercise 3: Knowledge Graph Completion
**Hint**: Implement DistMult as alternative to TransE
**Key Formula**: Score = <h, r, t> element-wise multiplication

### Exercise 4: Molecular Property Optimization
**Hint**: Modify molecules (add/remove atoms/bonds) and score
**Objective**: Maximize solubility, minimize toxicity

### Exercise 5: Recommendation Improvements
**Hint**: Calculate NDCG, Hits@K, implement cross-validation
**Baseline**: Compare to simple matrix factorization

### Exercise 6: Custom GNN Project
**Options**: 
  - Protein interaction networks
  - Flight route optimization
  - Code clone detection
  - Supply chain networks
  - Disease-gene interactions

---

## Debugging Tips

| Issue | Solution |
|-------|----------|
| Memory Error | Reduce dataset size, use sampling |
| Low Accuracy | Increase model capacity, more layers |
| Overfitting | Add dropout, L2 regularization |
| Slow Training | Reduce network size, use mini-batches |
| NaN Embeddings | Check normalization, reduce learning rate |

---

## Further Resources

### Papers Referenced
1. HAN: Xiao Wang et al., 2019
2. HGT: Ziniu Hu et al., 2020
3. TGN: Emanuele Rossi et al., 2020
4. GraphRNN: Jiaxuan You et al., 2018
5. TransE: Bordes et al., 2013
6. NGCF: Xiang Wang et al., 2019

### Datasets
- Open Graph Benchmark (OGB) - large-scale graphs
- PubMed - citation network
- QM9 - molecular properties
- MovieLens - recommendations
- Freebase/YAGO - knowledge graphs

### Libraries for Extension
- PyTorch Geometric (PyG)
- Deep Graph Library (DGL)
- NetworkX (analysis)
- RDKit (molecules)

---

## Completion Checklist

- [ ] Read Textbook Parts 1-2
- [ ] Run Notebook Projects 1-2
- [ ] Complete Exercise 1-2
- [ ] Read Textbook Parts 3-5
- [ ] Run Notebook Projects 3-5
- [ ] Complete Exercise 3-5
- [ ] Read Textbook Parts 6-7
- [ ] Run Notebook Project 6
- [ ] Attempt custom project
- [ ] Extend with PyTorch Geometric
- [ ] Apply to real datasets

---

**Last Updated**: December 28, 2025
**Version**: 1.0
