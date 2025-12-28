# Lesson 8: Advanced Topics & Applications

## Introduction

This lesson covers cutting-edge topics in Graph Neural Networks and their practical applications to real-world problems. We'll explore heterogeneous graphs with multiple node and edge types, temporal dynamics in graphs, graph generation, knowledge graphs, and domain-specific applications from molecules to social networks.

---

## Part 1: Heterogeneous Graphs

### 1.1 Concept and Motivation

A **heterogeneous graph** (or heterogeneous network) contains multiple types of nodes and/or multiple types of edges. Unlike simple graphs where all nodes and edges are identical, heterogeneous graphs model complex real-world systems where entities play different roles.

**Formal Definition**:
- **Heterogeneous Graph**: $G = (V, E, A, R)$
  - $V$: set of nodes
  - $E$: set of edges
  - $A$: node type set
  - $R$: edge type set
  - Type mapping: $\tau: V \rightarrow A$ and $\phi: E \rightarrow R$

### 1.2 Examples in Real-World

**Academic Network (ACM)**:
- Node types: Authors, Papers, Venues
- Edge types: Writes (Author→Paper), Publishes (Paper→Venue), Collaborates (Author→Author)

**E-commerce Network**:
- Node types: Users, Products, Categories, Shops
- Edge types: Purchases, Reviews, Belongs-to, Sells

**Bibliographic Network**:
- Node types: Authors, Papers, Keywords, Institutions
- Edge types: Authored, Published-in, Contains, Affiliated-with

### 1.3 Challenges with Heterogeneous Graphs

1. **Information Heterogeneity**: Different node types have different features and structural roles
2. **Type-Aware Message Passing**: Need to consider edge types when aggregating information
3. **Meta-paths**: Multiple relationship types create complex dependency structures
4. **Imbalanced Data**: Some node types may be much more common than others

### 1.4 Heterogeneous Graph Neural Networks

**HAN (Hierarchical Attention Networks)**:
- Uses attention mechanisms to weight different meta-paths
- Meta-path: A sequence of node types and relation types
- Example meta-path: Author → Paper → Venue → Paper → Author

**Key Components**:
1. **Meta-path Attention**: Learn importance weights for different meta-paths
2. **Node-level Attention**: Learn importance weights for neighbors along each meta-path
3. **Semantic-level Aggregation**: Aggregate features from different meta-paths

**Mathematical Formulation**:

For each meta-path P and target node i:

$$a_{ij}^P = \frac{\exp(\text{LeakyReLU}(W_{\text{att}}^T [h_i \, || \, h_j]))}{\sum_k \exp(\text{LeakyReLU}(W_{\text{att}}^T [h_i \, || \, h_k]))}$$

$$h_i^P = \sum_j a_{ij}^P W_{\text{in}}^P h_j$$

$$h_i = \sum_P \beta_P h_i^P$$

Where:
- $P$ represents a meta-path
- $\beta_P$ is the importance weight of meta-path P
- $h_i^P$ is the aggregated representation via meta-path P

### 1.5 Implementation Considerations

1. **Meta-path Selection**:
   - Domain knowledge is crucial
   - Can include direct and indirect relationships
   - Different meta-paths capture different semantic meanings

2. **Feature Heterogeneity**:
   - Different node types may have different feature dimensions
   - Use type-specific transformations: $W_i^{(l)}$ for node type i
   - Project features to common embedding space

3. **Scalability**:
   - Heterogeneous graphs can be extremely large
   - Need efficient meta-path sampling
   - Consider mini-batch learning strategies

### 1.6 Variants and Extensions

**HGT (Heterogeneous Graph Transformer)**:
- Attention-based temporal heterogeneous graph learning
- Handles both node and edge type heterogeneity
- Efficient for large-scale graphs

**R-GCN (Relational GCN)**:
- Extends GCN to handle multiple relation types
- Separate weight matrices for each relation type
- Introduces basis functions to reduce parameters

---

## Part 2: Temporal GNNs (Dynamic Graphs)

### 2.1 Introduction to Temporal Graphs

Real-world graphs are not static; they evolve over time. **Temporal graphs** (also called dynamic graphs or evolving graphs) capture this temporal dimension.

**Types of Temporal Dynamics**:
1. **Discrete time**: $G(t_0), G(t_1), G(t_2), \ldots$ sequence of snapshots
2. **Continuous time**: $G(t)$ for any time $t \in [0, \infty)$
3. **Event-based**: Graph state changes with discrete events

### 2.2 Challenges

1. **Long-Range Dependencies**: Information from distant past may be relevant
2. **Efficient Computation**: Computing over all historical data is expensive
3. **Missing Data**: Some time periods may have sparse information
4. **Non-stationary Distribution**: Graph structure and features change over time

### 2.3 Approaches to Temporal GNNs

#### A. Snapshot-Based Methods

Treat temporal graph as sequence of static graphs and apply GNN at each timestamp.

**Architecture**:
$$G(t_0) \rightarrow \text{GNN} \rightarrow h(t_0)$$
$$G(t_1) \rightarrow \text{GNN} \rightarrow h(t_1)$$
$$G(t_2) \rightarrow \text{GNN} \rightarrow h(t_2)$$
$$\downarrow$$
$$\text{RNN (LSTM/GRU)} \rightarrow \text{Future Prediction}$$

**Pros**: Simple, can leverage existing GNN architectures
**Cons**: May miss fine-grained temporal dynamics

#### B. Continuous-Time Methods

Use neural networks to model continuous evolution.

**Temporal Point Process**:
- Model events as a point process
- Intensity function: λ(t) = base rate + history influence
- Can handle irregular sampling

**Neural Ordinary Differential Equations (Neural ODEs)**:
- Model node embeddings as continuous functions of time
- $\frac{dh_i}{dt} = f_\theta(h_i(t), t)$
- Can interpolate embeddings between observed times

#### C. Recurrent Approaches

Use RNNs or Transformers on graph structure.

**Temporal Graph Networks (TGN)**:
- Maintains temporal memory for each node
- Updates memory with new interactions
- Uses memory to generate node embeddings

**Architecture**:
$$\text{Event: } (u, v, t, m)$$
$$\downarrow$$
$$\text{Update Memory: } m_u(t) = \text{RNN}(m_u(t^-), (v, t, m))$$
$$\downarrow$$
$$\text{Generate Embedding: } h_u(t) = \text{Aggregate}(m_u(t), \text{neighbors})$$
$$\downarrow$$
$$\text{Prediction}$$

### 2.4 Mathematical Framework

**Discrete-Time Formulation**:

For temporal graphs with snapshots $G(t)$, $t = 0, 1, \ldots, T$:

$$h_i^{(t)} = \text{AGGREGATE}(\{(h_j^{(t-\tau)}, e_{ij}^{(t-\tau)}) \text{ for } j \in N_i\})$$

where $\tau \in [0, T_{\text{agg}}]$ represents historical context window

**Continuous-Time Formulation**:

Using Neural ODE:
$$\frac{dh_i}{dt} = f(h_i(t), e_i(t), t)$$

$$h_i(t_1) = h_i(t_0) + \int_{t_0}^{t_1} f(h_i(\tau), e_i(\tau), \tau) d\tau$$

### 2.5 Applications

1. **Traffic Prediction**: Road networks change with time and congestion
2. **Social Network Evolution**: Friendships form and dissolve
3. **Recommendation Dynamics**: User-item interactions change over time
4. **Knowledge Graph Completion**: Facts emerge and become obsolete

---

## Part 3: Graph Generation

### 3.1 Problem Statement

**Graph Generation**: Given a training set of graphs, learn to generate new graphs with similar properties.

**Applications**:
- **Drug Discovery**: Generate novel molecules with desired properties
- **Design**: Discover new circuit designs or material structures
- **Anomaly Detection**: Generate normal graphs and detect deviations

### 3.2 Generative Models

#### A. Autoregressive Methods

Generate graph one node/edge at a time, conditioned on previous choices.

**GraphRNN**:
- Generate node sequence: $n_1, n_2, \ldots, n_N$
- At each step, determine connections to previous nodes
- Output sequence of edge adjacency vectors

**Architecture**:
$$s_1 \rightarrow \text{RNN} \rightarrow p(\text{edges to node 2})$$
$$s_1,s_2 \rightarrow \text{RNN} \rightarrow p(\text{edges to node 3})$$
$$s_1,s_2,s_3 \rightarrow \text{RNN} \rightarrow p(\text{edges to node 4})$$
$$\ldots$$

**Order Matters**: Different node orderings produce different generation sequences

#### B. One-Shot Generation

Generate entire graph at once (or edge matrix)

**Graph VAE (Variational Autoencoder)**:
- Encoder: Graph → Latent vector z
- Decoder: Latent vector z → Graph
- Learn continuous latent space of graphs

**Architecture**:
```
Input Graph
    ↓
Graph Encoder (GNN) → μ, σ
    ↓
Sample: z ~ N(μ, σ²)
    ↓
Graph Decoder (GNN) → Adjacency Matrix
    ↓
Generated Graph
```

#### C. Score-Based Generative Models

Use diffusion process: start from noise, gradually denoise to generate graphs

**Reverse SDE Approach**:
```
Forward (Diffusion): G(T) is pure noise
Reverse (Diffusion): G(0) → G(1) → ... → G(T) [learned reverse]
```

### 3.3 Challenges

1. **Validity**: Generated graphs must satisfy chemical/structural constraints
2. **Uniqueness**: Should not memorize training examples
3. **Scalability**: Exponential growth of possible graphs
4. **Evaluation**: Hard to quantify generation quality (no ground truth)

### 3.4 Evaluation Metrics

- **Validity**: % of generated graphs satisfying constraints
- **Novelty**: % of unique graphs not in training set
- **Diversity**: Variety of generated graphs
- **Realism**: Statistical similarity to training data (MM-distances, Weisfeiler-Lehman kernels)
- **Efficiency**: Graph property optimization (maximize reward function)

---

## Part 4: Knowledge Graphs

### 4.1 Definition and Structure

A **knowledge graph** (KG) represents facts about the world as a directed graph where nodes are entities and edges are relations.

**Example Facts**:
- (Einstein, born_in, Ulm)
- (Ulm, country, Germany)
- (Einstein, discovered, Theory_of_Relativity)

**Representation**: Triple (head_entity, relation, tail_entity) or (h, r, t)

### 4.2 Notable Knowledge Graphs

- **Freebase**: 45M entities, 1.2B facts
- **YAGO**: 10M entities from Wikipedia
- **Wikidata**: Community-driven, 100M+ entities
- **DBpedia**: Extracted from Wikipedia

### 4.3 Knowledge Graph Completion

Real KGs are incomplete. We need to predict missing facts.

**Link Prediction Problem**:
- Given incomplete KG with some facts unknown
- Predict missing facts: Which entities are likely to be related?

**Methods**:

**A. Translation-Based Models (TransE)**

Represent entities and relations as embeddings:
$$h + r \approx t \text{ (in ideal case)}$$

$$\text{Score}(h, r, t) = ||h + r - t||_2$$

Key idea: Relation is a translation from head to tail entity

**B. Semantic Matching Models (DistMult)**

Learn multiple semantic aspects:
$$\text{Score}(h, r, t) = \langle h, r, t \rangle \text{ (multilinear form)}$$
$$= \sum_i h_i \cdot r_i \cdot t_i$$

**C. Graph Neural Network Approaches**

Use GNNs to learn entity embeddings:
$$h_{\text{entity}} = \text{GNN}(\text{neighborhood features, relation features})$$
$$\text{Score}(h, r, t) = \text{Neural network}(h_{\text{entity}}, r, t_{\text{entity}})$$

### 4.4 Relation Types

- **One-to-One**: spouse (usually one spouse)
- **One-to-Many**: parent (one person has multiple children)
- **Many-to-One**: child_of (multiple people have one parent)
- **Many-to-Many**: knows (person knows many, known by many)

### 4.5 Reasoning and Inference

**Path-Based Reasoning**:
$$(X, \text{borrows\_from}, Y) \leftarrow (X, \text{employed\_by}, Z) \land (Z, \text{partner\_of}, W) \land (W, \text{owns}, Y)$$

**Logical Reasoning**:
- Rule mining from KG
- Neural symbolic approaches combining neural networks with logic

---

## Part 5: Real-World Applications

### 5.1 Molecular Graphs

**Molecular Representation**:
- Atoms as nodes (C, N, O, H, etc.)
- Chemical bonds as edges (single, double, aromatic)
- Atom features: atomic number, charge, hybridization

**Applications**:
1. **Molecular Property Prediction**
   - Predict: solubility, toxicity, HOMO-LUMO gap
   - Use: GCN, GraphSAGE, or GATv2
   - Real dataset: QM9 (134k molecules with 13 properties)

2. **Drug Discovery**
   - Design molecules with specific properties
   - Combine with reinforcement learning
   - Example: Generating molecules with high binding affinity

3. **Reaction Prediction**
   - Graph transformation: reactants → products
   - Learn transformation rules
   - Used by pharmaceutical companies for synthesis planning

**Challenges**:
- Molecular graphs are small (5-100 atoms) but highly complex
- Features (aromaticity, chirality) important but hard to capture
- Need equivariance to atom permutations and 3D rotations

### 5.2 Social Networks

**Graph Structure**:
- Users as nodes
- Friendships/followers as edges
- Attributes: profile information, interests

**Applications**:

1. **Link Prediction** (Friend Recommendation)
   - Predict future friendships
   - Common neighbors, resource allocation index
   - Neural approach: GNN + node pair representation

2. **Community Detection**
   - Find groups of closely connected users
   - Louvain method, spectral clustering
   - GNN-based: Learn node embeddings, cluster in latent space

3. **Influence Maximization**
   - Which k users to target for maximum spread?
   - Combinatorial optimization
   - Use GNNs for efficient propagation modeling

4. **Fake Account Detection**
   - Identify bot networks and fraud
   - Anomalous graph patterns
   - Graph autoencoders for novelty detection

**Real Platforms**:
- Facebook: 3B users, trillion-scale graphs
- Twitter: 500M users, retweet/mention graphs
- LinkedIn: 900M users, professional network

### 5.3 Recommendation Systems

**Graph Formulation**:
- Bipartite graph: Users ↔ Items
- Can include: Categories, Tags, Sellers

**Approaches**:

1. **Collaborative Filtering (Graph-based)**
   - User-User: Find similar users
   - Item-Item: Find similar items
   - GNN computes embeddings respecting graph structure

2. **Knowledge Graph Embeddings**
   - Knowledge graph includes user properties, item features, attributes
   - Joint embedding space: users and items
   - Prediction: (user, likes, item)?

3. **Graph Convolutional Networks for Recommendations (NGCF)**
   - Architecture:
   ```
   User/Item embeddings
       ↓
   GCN layers (multi-hop neighborhoods)
       ↓
   Interaction prediction
   ```
   - Captures collaborative signals through graph structure

4. **Session-based Recommendation (RNNs + Graphs)**
   - Model user session as directed graph
   - Item-to-item edges from sequential clicks
   - GNN captures both sequential and structural patterns

**Real Systems**:
- Netflix: 200M+ users
- Amazon: Billions of user-item pairs
- Spotify: Music recommendation graph

**Challenges**:
- Data sparsity: Most users rate few items
- Cold start: New users/items with no history
- Scalability: Billion-scale user-item matrices
- Long-term engagement vs. immediate relevance

---

## Part 6: Current Research Directions

### 6.1 Graph Transformers

Extend Transformers to graphs by modifying attention.

**Key Ideas**:
- Query and Key both from graph structure
- Attention weights depend on graph distance
- Can handle arbitrary graphs (unlike CNN)

**Challenges**:
- Computing attention between all pairs is O(n²)
- Need spatial and structural biases

### 6.2 Equivariant Neural Networks

Design networks that respect symmetries.

**Example: 3D Molecular Graphs**
- Permutation equivariance: Order of atoms shouldn't matter
- SE(3) equivariance: Rotation/translation shouldn't affect predictions
- Equivariant networks: Output transforms with input transforms

**Key Architectures**:
- Tensor field networks
- SE(3)-Transformers
- NequIP (NEquivariant QUantum Interacting Particle)

### 6.3 Large-Scale GNNs

Scaling to billion-node graphs.

**Techniques**:
1. **Node Sampling**: Random walk, neighbor sampling
2. **Graph Partitioning**: Distributed computation
3. **Approximate Aggregation**: Use sketches/summaries
4. **Learnable Sparsification**: Learn which edges to keep

### 6.4 Explainability

Understanding which parts of graph drive predictions.

**Methods**:
1. **GNNExplainer**: Find important edges by optimization
2. **Attention-based**: Attention weights as importance
3. **Perturbation**: Remove nodes/edges and measure impact
4. **Concept-based**: Learn high-level concepts

### 6.5 Robustness and Adversarial Attacks

GNNs can be fooled by adversarial perturbations.

**Adversarial Attack Example**:
```
Original node classification: Correct
Add 1-2 edges: Misclassification

Perturbation is small but effective
```

**Defense Strategies**:
- Adversarial training
- Robust graph normalization
- Certified defenses

### 6.6 Few-Shot Learning on Graphs

Learning from few examples using graph structure.

**Approach**:
- Use graph structure to propagate limited labels
- Meta-learning: Learn to learn from few examples
- Transfer learning: Leverage pre-trained graph encoders

---

## Part 7: Best Practices and Guidelines

### 7.1 Data Preparation

1. **Feature Engineering**
   - Use domain knowledge
   - Normalize continuous features
   - One-hot encode categorical features
   - Handle missing values appropriately

2. **Graph Normalization**
   - Symmetric normalization: D^(-1/2) A D^(-1/2)
   - Helps with convergence and stability

3. **Train-Test Split**
   - For node classification: Random node split
   - For link prediction: Random edge removal
   - For graph classification: Random graph split
   - Ensure no information leakage

### 7.2 Model Selection

1. **Start Simple**
   - Begin with GCN or GraphSAGE
   - Add complexity only if needed
   - Simpler models generalize better with limited data

2. **Architecture Choices**
   - Node classification: 2-3 GNN layers usually sufficient
   - Graph classification: More layers often needed
   - Heterogeneous graphs: Use specialized models (HAN, HGT)

3. **Aggregation Functions**
   - Mean: Simple, robust
   - Sum: Captures more info but needs normalization
   - Attention: More flexible but can overfit
   - Max: Good for graph classification

### 7.3 Training Tips

1. **Optimization**
   - Use Adam optimizer (default good starting point)
   - Learning rate: 0.01 is typical starting point
   - Weight decay: L2 regularization helps generalization

2. **Batch Size**
   - Mini-batch learning: Memory efficient
   - Mini-batch size 32-512 usually good
   - Full-batch: If graph fits in memory

3. **Early Stopping**
   - Monitor validation loss
   - Stop if no improvement for 50-100 epochs
   - Prevents overfitting

4. **Regularization**
   - Dropout: 0.5-0.8 commonly used
   - Layer normalization: Stabilizes training
   - Weight decay: L2 penalty (1e-5 to 1e-3 typical)

### 7.4 Hyperparameter Tuning

**Key Hyperparameters**:
1. Number of layers: 2-4 typical
2. Hidden dimension: 32-512
3. Learning rate: 1e-4 to 1e-1
4. Dropout rate: 0.0-0.8
5. Weight decay: 1e-6 to 1e-2

**Tuning Strategy**:
1. Grid search for rough tuning
2. Random search for fine tuning
3. Bayesian optimization for final tuning

### 7.5 Evaluation Metrics

**Node Classification**:
- Accuracy, Macro F1, Micro F1
- Per-class metrics for imbalanced data

**Link Prediction**:
- AUC-ROC, Hits@k, MRR
- Common evaluation: Remove 10% of edges, predict

**Graph Classification**:
- Accuracy
- Cross-validation important (small datasets)

**Graph Generation**:
- Validity, Uniqueness, Novelty
- Statistical similarity (MMD, kernel methods)

### 7.6 Common Pitfalls

1. **Label Leakage**
   - Don't use target information during aggregation
   - Careful with train-test split on graphs

2. **Over-smoothing**
   - Deep GNNs produce similar node embeddings
   - Use residual connections, skip connections
   - Limit to 2-4 layers unless specialized architecture

3. **Scalability Issues**
   - Full adjacency matrix: O(n²) memory
   - Use sparse formats (COO, CSR)
   - Implement sampling for large graphs

4. **Feature Imbalance**
   - Different features may have different scales
   - Normalize across all features
   - Use batch normalization

5. **Graph Heterogeneity**
   - Don't ignore node/edge types
   - Use type-specific parameters
   - Learn type-aware attention weights

### 7.7 Reproducibility

1. **Set Random Seeds**
   ```python
   np.random.seed(42)
   torch.manual_seed(42)
   torch.cuda.manual_seed(42)
   ```

2. **Document Hyperparameters**
   - Save all settings used
   - Include data preprocessing steps
   - Record random seeds

3. **Version Control**
   - Use git for code
   - Track dependencies (requirements.txt)
   - Save trained models

---

## Summary

In this lesson, we covered:

1. **Heterogeneous Graphs**: Multiple node/edge types require specialized architectures (HAN, HGT)
2. **Temporal GNNs**: Dynamic graphs captured via snapshots, continuous-time, or RNNs
3. **Graph Generation**: Autoregressive (GraphRNN), VAE, and diffusion-based methods
4. **Knowledge Graphs**: Entity-relation facts, embedding methods (TransE), completion tasks
5. **Applications**:
   - Molecules: Property prediction, drug discovery
   - Social Networks: Link prediction, community detection, influence maximization
   - Recommendations: User-item graphs, knowledge-aware systems
6. **Research Directions**: Graph Transformers, equivariance, scalability, explainability, robustness
7. **Best Practices**: Data prep, model selection, training, tuning, evaluation, avoiding pitfalls

---

## Key Takeaways

- Graph heterogeneity is the norm in real-world; simple GNNs insufficient
- Temporal dynamics critical for evolving networks
- Graph generation enables design and discovery
- Knowledge graphs represent structured knowledge, require specialized methods
- Real applications require careful consideration of domain constraints
- Best practices from ML apply to GNNs; adapt them thoughtfully
- Research is rapidly advancing; stay updated on new architectures and methods

---

## Further Reading

### Books
1. **Graph Neural Networks: Foundations, Frontiers, and Applications** by Jiawei Zhang, Yixin Cao, Ziyuan Li, Shuhuai Ren, Yanfeng Zhang, Weinan Zhang, and Yong Li
2. **Knowledge Graphs** by Aidan Hogan, et al.
3. **Geometric Deep Learning** by Bronstein, Bruna, Cohen, Veličković

### Key Papers
1. **Heterogeneous Graph Attention Network** (HAN) - Xiao Wang et al., 2019
2. **Heterogeneous Graph Transformer** (HGT) - Ziniu Hu et al., 2020
3. **Temporal Graph Networks for Deep Learning on Dynamic Graphs** - Emanuele Rossi et al., 2020
4. **GraphRNN: Generating Realistic Graphs via Deep Recursive Models** - Jiaxuan You et al., 2018
5. **Variational Graph Auto-Encoders** - Kipf & Welling, 2016
6. **TransE: Translating Embeddings for Modeling Multi-relational Data** - Bordes et al., 2013
7. **Neural Graph Collaborative Filtering** (NGCF) - Xiang Wang et al., 2019

### Datasets and Resources
- **OGB (Open Graph Benchmark)**: Large-scale graph datasets
- **PytorchGeometric (PyG)**: Graph deep learning library
- **DGL (Deep Graph Library)**: Graph learning framework
- **NetworkX**: Graph analysis and visualization

---

## Exercises

1. Implement a heterogeneous graph with at least 3 node types and 2 edge types
2. Design a temporal graph experiment on network evolution
3. Implement TransE for knowledge graph completion
4. Build a molecular graph classification model on a real dataset
5. Create a recommendation system using bipartite user-item graphs
6. Analyze and explain GNN predictions using node/edge importance

---

**Next Steps**: Apply these advanced concepts to your own research or industry problems. GNNs are a rapidly evolving field with continuous improvements and new applications emerging regularly.
