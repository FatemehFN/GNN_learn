# Lesson 8: Advanced Topics & Applications

This comprehensive lesson covers cutting-edge topics in Graph Neural Networks and their practical applications.

## Files Included

### 1. lesson_08_textbook.md (22 KB, 753 lines)
**Comprehensive markdown textbook covering:**

- **Part 1: Heterogeneous Graphs (Sections 1.1-1.6)**
  - Concept and motivation for heterogeneous networks
  - Real-world examples (academic networks, e-commerce, bibliographic)
  - Challenges with heterogeneous graphs
  - HAN and HGT architectures
  - Meta-path selection and implementation

- **Part 2: Temporal GNNs (Sections 2.1-2.5)**
  - Discrete-time and continuous-time temporal graphs
  - Snapshot-based, continuous-time, and recurrent approaches
  - Temporal Point Processes and Neural ODEs
  - Temporal Graph Networks (TGN)
  - Applications (traffic, social networks, recommendations)

- **Part 3: Graph Generation (Sections 3.1-3.4)**
  - Autoregressive methods (GraphRNN)
  - One-shot generation (Graph VAE)
  - Score-based generative models
  - Evaluation metrics (validity, novelty, diversity, realism)

- **Part 4: Knowledge Graphs (Sections 4.1-4.5)**
  - Definition and structure of KGs
  - Notable KGs (Freebase, YAGO, Wikidata, DBpedia)
  - Knowledge graph completion task
  - Translation-based (TransE) and semantic matching models
  - Relation types and reasoning

- **Part 5: Real-World Applications (Sections 5.1-5.3)**
  - Molecular graphs and drug discovery
  - Social network applications (link prediction, community detection)
  - Recommendation systems (collaborative filtering, NGCF)
  - Challenges and real platforms

- **Part 6: Current Research Directions (Sections 6.1-6.6)**
  - Graph Transformers
  - Equivariant Neural Networks (SE(3) equivariance)
  - Large-scale GNNs (sampling, partitioning)
  - Explainability and interpretability
  - Robustness and adversarial attacks
  - Few-shot learning on graphs

- **Part 7: Best Practices (Sections 7.1-7.7)**
  - Data preparation and normalization
  - Model selection strategies
  - Training tips and optimization
  - Hyperparameter tuning
  - Evaluation metrics by task
  - Common pitfalls to avoid
  - Reproducibility guidelines

## 2. lesson_08_notebook.ipynb (67 KB, 1514 lines)
**Interactive Jupyter notebook with 6 complete projects:**

### Project 1: Heterogeneous Graphs
- `HeterogeneousGraph` class for multi-type nodes and edges
- Academic network example (Authors, Papers, Venues)
- `SimpleHeterogeneousGNN` implementation with type-aware aggregation
- Meta-path discovery and analysis

### Project 2: Temporal Graphs
- `TemporalGraph` class for time-evolving networks
- Social network simulation over 5 time steps
- `TemporalGraphForecaster` for predicting future edges
- Temporal neighborhood analysis

### Project 3: Knowledge Graphs & Link Prediction
- `KnowledgeGraph` class for entity-relation-entity triples
- Knowledge graph about famous scientists
- `TransEModel` implementation:
  - Embedding-based scoring: h + r ≈ t
  - Margin ranking loss
  - Link prediction via scoring
- Prediction of missing facts

### Project 4: Molecular Graphs
- `create_molecular_graph()` for synthetic molecules
- Atom and bond representations
- `MolecularGNN` for property prediction
- Feature extraction and message passing
- Molecular property visualization

### Project 5: Recommendation Systems
- User-item bipartite graph creation
- `RecommendationGNN` with collaborative filtering
- Neighborhood aggregation for embeddings
- Top-k recommendation generation
- Evaluation with MAE, RMSE, coverage metrics

### Project 6: Citation Network Link Prediction
- Citation graph construction (50 papers, temporal)
- Baseline methods:
  - Common Neighbors
  - Jaccard Similarity
- `LinkPredictionGNN` with message passing
- Comprehensive evaluation:
  - Train/test split
  - Negative sampling
  - AUC-ROC and Average Precision
  - Method comparison

## 6 Hands-On Exercises

1. **Heterogeneous Graph Classification**: Classify venues using meta-paths
2. **Temporal Graph Analysis**: Analyze network evolution patterns
3. **Knowledge Graph Completion**: Extend TransE model with advanced reasoning
4. **Molecular Property Optimization**: Design molecules with desired properties
5. **Recommendation Improvements**: Enhanced model with better evaluation
6. **Custom GNN Project**: Build application for domain of choice

## Key Implementation Highlights

### Code Features
- Clean, well-documented Python implementations
- NumPy-based core algorithms (no framework dependencies for educational clarity)
- Real-world datasets and synthetic data generation
- Comprehensive evaluation methodologies
- Visualization of results with matplotlib

### Topics Covered
- Heterogeneous node/edge types
- Temporal dynamics and sequences
- Embedding-based scoring functions
- Message passing aggregation
- Bipartite graph analysis
- Link prediction evaluation
- Meta-path based reasoning
- Graph-based property prediction

### Practical Applications
- Academic network analysis
- Temporal social network forecasting
- Knowledge base completion
- Drug discovery and molecular design
- E-commerce recommendations
- Citation analysis and impact prediction

## How to Use

1. **Read the textbook first** (lesson_08_textbook.md)
   - Builds understanding from concepts to applications
   - Provides theoretical foundation

2. **Work through the notebook** (lesson_08_notebook.ipynb)
   - Run each section sequentially
   - Understand implementations
   - Modify code to experiment

3. **Complete the exercises**
   - Apply concepts to new problems
   - Build intuition through implementation
   - Create your own applications

## Dependencies

```python
# Core
numpy
pandas
matplotlib
seaborn
networkx

# Optional (for advanced use)
torch  # PyTorch
torch_geometric  # PyTorch Geometric
scikit-learn  # ML utilities
rdkit  # Molecular analysis (for real molecules)
```

## Learning Path

1. **Beginner**: Read textbook Parts 1-2, run notebook Projects 1-2
2. **Intermediate**: Read Parts 3-5, run Projects 3-5, attempt Exercises 1-3
3. **Advanced**: Read Part 6, run Project 6, complete all exercises, design custom project

## What's Next?

- Implement with PyTorch Geometric for production
- Apply to your own domain-specific graphs
- Explore recent papers in GNN literature
- Contribute to open-source GNN projects
- Study graph transformers and equivariant networks

---

**Created**: December 28, 2025
**Total Content**: 22 KB textbook + 67 KB notebook with 6 complete projects and 6 exercises
**Estimated Study Time**: 15-20 hours for complete understanding
