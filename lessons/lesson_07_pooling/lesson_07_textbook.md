# Lesson 7: Graph Pooling & Hierarchical GNNs

## Table of Contents
1. [Graph-Level Tasks Overview](#graph-level-tasks-overview)
2. [Pooling Operations](#pooling-operations)
3. [DiffPool](#diffpool)
4. [Top-K Pooling](#top-k-pooling)
5. [SAGPool](#sagpool)
6. [Hierarchical Graph Representations](#hierarchical-graph-representations)
7. [Applications](#applications)
8. [Summary](#summary)

---

## Graph-Level Tasks Overview

### Introduction to Graph Classification

So far, we've focused on **node-level tasks** (node classification) and **link prediction** tasks. In this lesson, we explore **graph-level tasks** where the goal is to classify entire graphs or predict properties of whole graphs.

#### Why Graph-Level Tasks Matter

Real-world applications often require understanding entire graphs:
- **Molecular property prediction**: Predict chemical properties of molecules (represented as graphs)
- **Social network analysis**: Classify entire communities or networks
- **Protein structure classification**: Categorize proteins based on their 3D structure
- **Material science**: Predict material properties from atomic structure graphs
- **Drug discovery**: Screen compounds for bioactivity or toxicity

#### Graph-Level vs Node-Level Tasks

| Aspect | Node-Level | Graph-Level |
|--------|-----------|------------|
| **Goal** | Classify/predict properties for individual nodes | Classify/predict properties for entire graphs |
| **Output** | N predictions (one per node) | 1 prediction per graph |
| **Applications** | Citation networks, social networks | Molecules, proteins, materials |
| **Challenge** | Node embeddings already computed | Need to aggregate node embeddings to graph level |

#### The Graph Classification Pipeline

```
Graph Input → Node Embedding Layer → Graph Pooling → Readout Layer → Prediction
                (GNN layers)           (Aggregation)   (MLP)
```

### From Node Embeddings to Graph Embeddings

After processing with GNN layers, we have node embeddings. The challenge is to produce a **single graph embedding** that captures the entire graph's structure.

```
Nodes: h₁, h₂, ..., hₙ → Graph Embedding: g
```

This requires a **readout function** that aggregates node information while:
1. Being permutation invariant (order of nodes shouldn't matter)
2. Being differentiable (for end-to-end learning)
3. Preserving important structural information

---

## Pooling Operations

### What is Pooling?

Pooling is the process of aggregating information from multiple nodes into a single representation. In CNNs, pooling reduces spatial dimensions while preserving important features. In GNNs, pooling serves multiple purposes:

1. **Dimensionality reduction**: Reduce the number of nodes
2. **Information aggregation**: Combine node information hierarchically
3. **Feature extraction**: Learn what's important for the task

### Global Pooling

Global pooling aggregates all nodes in a graph into a single vector.

#### Sum Pooling

$$g = \sum_i h_i$$

**Pros:**
- Simple and interpretable
- Permutation invariant
- Computationally efficient

**Cons:**
- Can lose structural information
- Sensitive to graph size
- May struggle with graphs of very different sizes

#### Mean Pooling

$$g = \frac{1}{n} \sum_i h_i$$

**Pros:**
- Normalized by graph size
- Better for graphs of different sizes
- More stable across graph sizes

**Cons:**
- Loses absolute scale information
- May miss important dense subgraphs

#### Max Pooling

$$g_j = \max_i h_{ij} \text{ (element-wise maximum)}$$

**Pros:**
- Preserves distinctive node features
- Captures peaks in feature distributions
- Effective for detecting specific patterns

**Cons:**
- Loses information about average behavior
- May focus on outliers
- Less smooth gradients

### Readout Functions

A general readout function combines different pooling operations:

$$g = h_{\text{sum}} \, || \, h_{\text{mean}} \, || \, h_{\text{max}}$$

where $||$ denotes concatenation. This combines the benefits of all three approaches.

### Limitations of Global Pooling

Global pooling has significant limitations:

1. **Loss of structure**: All spatial information about node connections is lost
2. **Permutation invariance too strong**: Treats isomorphic structures as identical even if they're topologically important
3. **Bottleneck**: A single aggregation step may lose critical information in complex graphs
4. **Fixed graph size**: Cannot naturally handle variable-size graphs

---

## Hierarchical Pooling

### Motivation for Hierarchical Approaches

Instead of a single global pooling step, hierarchical pooling creates a **coarse-grained graph representation** that preserves structural information while reducing complexity.

```
Graph Level 0: n nodes, m edges
      ↓ (Pooling)
Graph Level 1: n' nodes (n' < n), m' edges
      ↓ (Pooling)
Graph Level 2: n'' nodes (n'' < n'), m'' edges
      ↓ (Global Pooling)
Graph Embedding: g
```

#### Key Idea

Hierarchical pooling operates in **multiple scales**, similar to the multi-scale processing in image CNNs:
- **Layer 1**: Learn node embeddings with local neighborhood context
- **Layer 2**: Cluster similar nodes and create super-nodes
- **Layer 3**: Learn super-node embeddings
- **Repeat**: Multiple levels of hierarchy

#### Benefits of Hierarchical Pooling

1. **Structural preservation**: Maintains graph topology at multiple scales
2. **Better feature learning**: Features learned at each scale
3. **Interpretability**: Can visualize which nodes form important clusters
4. **Efficiency**: Reduces graph size progressively

---

## DiffPool

### Overview

**DiffPool (Differentiable Pooling)** is a learnable hierarchical pooling method that learns **which nodes to cluster together** in an end-to-end differentiable manner.

Paper: "Hierarchical Graph Representation Learning with Differentiable Pooling" (Lee et al., ICLR 2019)

### Key Idea: Learnable Node Assignment

Instead of using heuristics to decide which nodes cluster together, DiffPool learns an **assignment matrix** S:

```
S ∈ ℝ^(n × k)
```

where:
- n = number of nodes in current layer
- k = number of nodes in next (coarser) layer
- S[i,j] = probability that node i is assigned to super-node j

### The DiffPool Operation

Given:
- Node embeddings: $H \in \mathbb{R}^{n \times d}$ (n nodes, d-dimensional features)
- Adjacency matrix: $A \in \mathbb{R}^{n \times n}$

**Step 1: Learn Assignment Matrix**

$$S = \text{softmax}(\text{GNN}(H, A)) \in \mathbb{R}^{n \times k}$$

Each column of S is softmax-normalized, so each node has a probability distribution over super-nodes.

**Step 2: Pool Node Features**

$$H_{\text{pool}} = S^T H \in \mathbb{R}^{k \times d}$$

Each row of $H_{\text{pool}}$ is a weighted combination of original node embeddings.

**Step 3: Pool Adjacency Matrix**

$$A_{\text{pool}} = S^T A S \in \mathbb{R}^{k \times k}$$

This creates a new adjacency matrix for super-nodes, where the weight between super-nodes is the sum of edge weights between their constituent nodes.

### Mathematical Formulation

Let $G = (H, A)$ be a graph with node features H and adjacency A.

At layer $\ell$:
1. Compute assignment matrix: $S_\ell = \text{softmax}(\text{GNN}_\ell(H_\ell, A_\ell))$
2. Update node embeddings: $H_{\ell+1} = \text{MLP}_\ell(H_\ell, A_\ell)$
3. Pool features: $H'_{\ell+1} = S_\ell^T H_{\ell+1}$
4. Pool adjacency: $A'_{\ell+1} = S_\ell^T A_\ell S_\ell$

The GNN that produces S should be different from the GNN that updates node features to ensure they're trained differently.

### Auxiliary Loss for Cluster Quality

To encourage meaningful clustering, DiffPool adds an auxiliary loss term:

$$\mathcal{L}_{\text{aux}} = ||A - SS^T||_F^2$$

where $||\cdot||_F$ is the Frobenius norm. This loss encourages the reconstructed adjacency $SS^T$ to be similar to the actual adjacency A, promoting **cohesive clustering** (nodes in the same cluster should have been connected).

### Full Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{aux}} \cdot \mathcal{L}_{\text{aux}}$$

where:
- $\mathcal{L}_{\text{task}}$ = main task loss (e.g., cross-entropy for classification)
- $\mathcal{L}_{\text{aux}}$ = auxiliary loss (encourages good clusters)
- $\lambda_{\text{aux}}$ = hyperparameter balancing the two losses

### Advantages of DiffPool

1. **End-to-end differentiable**: Pooling is learned as part of the network
2. **Interpretable**: Assignment matrix shows which nodes cluster together
3. **Flexible**: Works with any underlying GNN architecture
4. **Multi-scale learning**: Multiple pooling layers capture different scales

### Disadvantages of DiffPool

1. **Computational cost**: $O(n^2)$ space and time for assignment matrix and pooling
2. **Sensitivity to initialization**: Can get stuck in poor local optima
3. **Hard to scale**: Problematic for very large graphs (millions of nodes)
4. **Complexity**: More hyperparameters to tune

---

## Top-K Pooling

### Overview

**Top-K Pooling** is a simpler pooling method that selects the **top k nodes** based on some scoring mechanism.

Paper: "Graph U-Nets" (Lee et al., ICML 2019)

### The Core Idea

Instead of learning a full assignment matrix, Top-K selects the most "important" nodes:

1. Score each node: $\text{score}_i = \frac{h_i \cdot w}{||w||}$
2. Select top-k nodes by score
3. Prune edges to/from non-selected nodes
4. Continue with reduced graph

### Detailed Algorithm

**Step 1: Compute Node Scores**

$$\text{score}_i = \sigma(h_i \cdot w)$$

where:
- $h_i$ = node embedding
- $w$ = learnable scoring vector
- $\sigma$ = sigmoid function

**Step 2: Select Top-K Nodes**

$$k = \lceil r \cdot n \rceil \quad \text{(where } r \in [0,1] \text{ is pooling ratio)}$$
$$\text{idx} = \text{top\_k}(\text{score}, k)$$

Select the k nodes with highest scores.

**Step 3: Create Pooled Graph**

$$H_{\text{pool}} = H[\text{idx}] \quad \text{(select embeddings of kept nodes)}$$
$$A_{\text{pool}} = A[\text{idx}][:, \text{idx}] \quad \text{(select subgraph induced by kept nodes)}$$

**Step 4: Update Node Scores**

For nodes that remain, maintain their importance scores for potential unpooling.

### Advantages of Top-K Pooling

1. **Computational efficiency**: $O(n \log n)$ or $O(n)$ with partial sorting
2. **Interpretable**: Explicitly selects important nodes
3. **Differentiable**: Gradients flow through selection operation
4. **Flexible pooling ratio**: Can adjust k as hyperparameter

### Disadvantages of Top-K Pooling

1. **Hard selection**: Discrete selection (some nodes completely removed)
2. **Graph disconnection**: May disconnect the graph by removing nodes
3. **Information loss**: Pruned nodes and edges are lost
4. **Limited cluster information**: Doesn't explicitly model node groupings

### Top-K Pooling with Graph U-Nets

Top-K pooling is often used in **Graph U-Net** architectures that combine pooling and unpooling:

```
Input Graph
     ↓ (GNN, Pooling)
Intermediate Representation
     ↓ (GNN, Unpooling)
Output Graph
```

This enables:
- Downsampling (pooling) to capture multi-scale features
- Upsampling (unpooling) to restore spatial resolution
- Skip connections between pooling and unpooling layers

---

## SAGPool

### Overview

**SAGPool (Self-Attention Graph Pooling)** uses a learnable scoring mechanism based on self-attention to decide which nodes to keep.

Paper: "Self-Attention Graph Pooling" (Lee et al., ICLR 2019)

### Key Innovation: Attention-Based Scoring

Instead of a simple learned vector, SAGPool uses a **self-attention mechanism** to compute importance scores:

```
score_i = σ(h_i · W)
```

where W is a learnable weight matrix. This allows the scoring to depend on the node's own features in a more flexible way.

### The SAGPool Algorithm

**Step 1: Compute Self-Attention Scores**

$$\text{scores} = \sigma(H \cdot w + b)$$

where:
- $H \in \mathbb{R}^{n \times d}$ = node embeddings
- $w \in \mathbb{R}^d$ = learnable scoring weights
- $b$ = bias term
- $\sigma$ = sigmoid activation

**Step 2: Select Top-K Nodes**

$$k = \lceil \text{ratio} \cdot n \rceil$$
$$\text{idx} = \text{top\_k}(\text{scores}, k)$$

**Step 3: Create Pooled Graph**

$$H_{\text{pool}} = H[\text{idx}]$$
$$A_{\text{pool}} = A[\text{idx}][:, \text{idx}]$$

**Step 4: Apply GNNs to Pooled Graph**

Apply another GNN layer to the pooled graph to learn representations of the selected nodes.

### Advantages of SAGPool

1. **Attention mechanism**: More sophisticated scoring than simple learned vector
2. **Computational efficiency**: O(n log n) with efficient top-k selection
3. **Interpretability**: Attention scores show importance of each node
4. **Simplicity**: Easy to implement and integrate into existing architectures

### Disadvantages of SAGPool

1. **Limited expressiveness**: Attention scores are based on individual node features
2. **Graph structure ignored**: Doesn't consider connections when scoring
3. **Hard selection**: Discrete selection may lose useful information
4. **Scalability**: Still $O(n \log n)$ for very large graphs

### Comparison: SAGPool vs Top-K vs DiffPool

| Property | Top-K | SAGPool | DiffPool |
|----------|-------|---------|----------|
| **Scoring** | Learned vector | Attention | Learned assignment |
| **Complexity** | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ |
| **Differentiability** | Yes | Yes | Yes |
| **Cluster info** | No | No | Yes |
| **Scalability** | Good | Good | Poor |
| **Interpretability** | Good | Good | Excellent |

---

## Hierarchical Graph Representations

### Multi-Scale Graph Learning

Hierarchical pooling enables learning **multi-scale representations** of graphs:

```
Scale 1: Fine-grained (original nodes)
         ↓
Scale 2: Medium-grained (clusters of nodes)
         ↓
Scale 3: Coarse-grained (clusters of clusters)
         ↓
Global: Graph-level representation
```

### Hierarchical GNN Architecture

A complete hierarchical GNN combines:

1. **Embedding layers**: Convert input features to initial embeddings
2. **Layer 1 GNN**: Process at finest scale
3. **Pooling 1**: Coarsen graph at scale 1
4. **Layer 2 GNN**: Process at intermediate scale
5. **Pooling 2**: Further coarsen graph
6. **Layer 3 GNN**: Process at coarsest scale
7. **Global pooling**: Aggregate to graph level
8. **Readout MLP**: Produce final prediction

```python
class HierarchicalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        # Embedding
        self.embed = nn.Linear(in_channels, hidden_channels)

        # Layer 1
        self.gnn1 = GNNLayer(hidden_channels, hidden_channels)
        self.pool1 = DiffPooling(hidden_channels, hidden_channels // 2)

        # Layer 2
        self.gnn2 = GNNLayer(hidden_channels, hidden_channels)
        self.pool2 = DiffPooling(hidden_channels, hidden_channels // 2)

        # Layer 3
        self.gnn3 = GNNLayer(hidden_channels, hidden_channels)

        # Readout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # Embedding
        x = self.embed(x)

        # Layer 1
        x = self.gnn1(x, edge_index)
        x, edge_index, batch = self.pool1(x, edge_index, batch)

        # Layer 2
        x = self.gnn2(x, edge_index)
        x, edge_index, batch = self.pool2(x, edge_index, batch)

        # Layer 3
        x = self.gnn3(x, edge_index)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Readout
        return self.mlp(x)
```

### Advantages of Hierarchical Representations

1. **Better generalization**: Multiple scales help model complex patterns
2. **Efficiency**: Progressive reduction in graph size
3. **Interpretability**: Can understand which scales are important
4. **Robustness**: Less sensitive to small structural variations

### Information Flow in Hierarchical GNNs

```
Fine-grained features → Coarser features → Graph-level features
(Local patterns)      (Larger patterns)   (Global patterns)
```

---

## Applications

### Molecular Property Prediction

Molecules are naturally represented as graphs where:
- **Nodes**: Atoms
- **Edges**: Chemical bonds
- **Node features**: Atomic properties (type, mass, electronegativity)
- **Task**: Predict molecular properties (toxicity, activity, solubility)

#### Why Pooling Matters for Molecules

1. **Variable size**: Molecules have different numbers of atoms
2. **Size-invariant**: Properties don't scale with molecule size
3. **Hierarchical structure**: Atoms form functional groups, which form larger structures
4. **Multiple scales**: Important features at atom level, group level, and molecule level

#### Example: BACE Dataset

- **Task**: Predict if molecules inhibit BACE (involved in Alzheimer's)
- **Graphs**: 1,513 molecules (5-100 atoms each)
- **Features**: Atom types, bond types
- **Labels**: Binary (inhibitor / non-inhibitor)

### Social Network Analysis

Graphs representing communities:
- **Nodes**: Users
- **Edges**: Friendships
- **Task**: Classify communities or predict network properties

### Protein Structure Classification

Protein structures as contact graphs:
- **Nodes**: Amino acids
- **Edges**: Spatial proximity
- **Task**: Classify protein fold types

### Graph Pattern Recognition

Detecting patterns in large graphs:
- **Nodes**: Entities
- **Edges**: Relations
- **Task**: Classify subgraph patterns or anomaly detection

### Drug Discovery

- **Molecules as graphs**: Represent chemical structures
- **Property prediction**: Bioactivity, toxicity, solubility
- **Lead optimization**: Find molecules with desired properties
- **ADME properties**: Absorption, distribution, metabolism, excretion

---

## Summary

### Key Concepts

1. **Graph-level tasks** require aggregating node information to predict whole-graph properties
2. **Global pooling** (sum, mean, max) is simple but loses structural information
3. **Hierarchical pooling** creates multi-scale representations preserving structure
4. **DiffPool** learns a soft assignment matrix for interpretable clustering
5. **Top-K and SAGPool** are efficient alternatives using node importance scores
6. **Readout functions** often combine multiple pooling types for robustness

### When to Use Each Method

| Method | Best For | Drawbacks |
|--------|----------|-----------|
| **Global mean/sum** | Small graphs, baseline models | Loses structure, size-sensitive |
| **Global max** | Detecting distinctive patterns | Focuses on outliers |
| **Top-K/SAGPool** | Large graphs, efficiency needed | Hard selection, disconnection |
| **DiffPool** | Interpretability, medium graphs | O(n²) complexity |
| **Hierarchical** | Complex graphs, multi-scale patterns | More hyperparameters |

### Important Considerations

1. **Graph size variance**: Different pooling methods handle variable-size graphs differently
2. **Computational budget**: Trade-off between expressiveness and efficiency
3. **Interpretability needs**: Some applications require understanding which nodes matter
4. **Task complexity**: Simple tasks may not need hierarchical pooling
5. **Data scale**: Very large graphs need efficient methods like SAGPool

### Further Reading

- Lee, J., Lee, I., & Kang, J. (2019). "Self-Attention Graph Pooling." ICLR.
- Lee, J., Roses, R., & Kang, J. (2019). "Graph U-Nets." ICML.
- Lee, J., Strathmann, H., Müller, E., & Kang, J. (2019). "Hierarchical Graph Representation Learning with Differentiable Pooling." ICLR.
- Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., & Leskovec, J. (2021). "Subgraph Neural Networks." ICLR.

---

## Exercises

### Exercise 1: Global Pooling
Implement sum, mean, and max pooling functions. Compare their behavior on graphs of different sizes.

### Exercise 2: Top-K Pooling
Build a Top-K pooling layer. Visualize which nodes are selected for different pooling ratios.

### Exercise 3: SAGPool Integration
Implement SAGPool and integrate it into a GNN for graph classification.

### Exercise 4: Multi-Layer Hierarchical GNN
Build a 3-layer hierarchical GNN with pooling at each layer.

### Exercise 5: Molecule Classification
Train a hierarchical GNN on the BACE dataset for drug activity prediction.

### Exercise 6: Visualization
Visualize the hierarchical structure created by multi-layer pooling. Show which atoms cluster together at different scales.

### Exercise 7: Comparison Study
Compare global pooling vs SAGPool vs DiffPool on the same classification task. Analyze accuracy, runtime, and interpretability.
