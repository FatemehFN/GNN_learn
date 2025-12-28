# Lesson 6: Graph Attention Networks (GAT)

## Table of Contents
1. [Introduction](#introduction)
2. [Attention Mechanisms Overview](#attention-mechanisms-overview)
3. [GAT Architecture](#gat-architecture)
4. [Attention Coefficients Calculation](#attention-coefficients-calculation)
5. [Multi-Head Attention](#multi-head-attention)
6. [Mathematical Formulation](#mathematical-formulation)
7. [Advantages over GCN](#advantages-over-gcn)
8. [Computational Complexity](#computational-complexity)
9. [Applications](#applications)
10. [Summary](#summary)

---

## Introduction

Graph Attention Networks (GAT) represent a paradigm shift in graph neural networks by introducing the attention mechanism to the graph learning domain. Rather than using fixed aggregation weights (like in GCN), GAT learns task-specific weights for each edge dynamically, enabling the model to focus on the most relevant neighbors.

**Key Innovation**: Attention mechanisms allow the model to assign different importance weights to different neighbors without requiring edge features or complex matrix operations.

---

## Attention Mechanisms Overview

### What is Attention?

Attention is a mechanism that allows a model to focus on the most relevant information while processing data. It answers the question: "Which parts of the input should I pay attention to?"

### Components of Attention

1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What information do I have?"
3. **Value (V)**: "What information should be used?"

The attention weight quantifies the relationship between a query and a key, determining how much influence the corresponding value has.

### Attention Score Calculation

The basic formula for attention is:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $d_k$ is the dimension of keys (for scaling)
- Softmax normalizes scores to a probability distribution

### Why Attention in Graphs?

In GCNs, all neighbors contribute equally to the aggregation (with normalization by degree). However, not all neighbors are equally important. Attention allows the model to:

- Learn which neighbors are most relevant
- Dynamically adjust weights based on node features
- Handle heterogeneous neighborhoods better
- Capture long-range dependencies more effectively

---

## GAT Architecture

### Overview

A Graph Attention Layer performs the following steps:

1. **Feature Transformation**: Transform node features linearly
2. **Attention Computation**: Compute attention coefficients for each edge
3. **Aggregation**: Aggregate neighbor features weighted by attention
4. **Output**: Generate new node representations

### Single-Head Attention Mechanism

For a node $i$ and its neighborhood $\mathcal{N}(i)$:

1. Linearly transform features: $\mathbf{h}_i' = \mathbf{W} \mathbf{h}_i$
2. Compute attention logits
3. Normalize with softmax
4. Aggregate weighted features

### Architecture Diagram

```
Input Features (h_i, h_j)
        ↓
    Linear Transform (W)
        ↓
    Concatenate (h'_i || h'_j)
        ↓
    Attention Coefficient (LeakyReLU)
        ↓
    Softmax (normalize over neighborhood)
        ↓
    Weight aggregation with attention
        ↓
    Output Features (h'_i)
```

---

## Attention Coefficients Calculation

### Step-by-Step Computation

For an edge between nodes $i$ and $j$:

#### Step 1: Linear Feature Transformation
Transform input features using a learnable weight matrix $\mathbf{W} \in \mathbb{R}^{F' \times F}$:

$$\mathbf{h}_i' = \mathbf{W}\mathbf{h}_i \quad \text{and} \quad \mathbf{h}_j' = \mathbf{W}\mathbf{h}_j$$

Where:
- $F$ = input feature dimension
- $F'$ = output feature dimension
- $\mathbf{h}_i \in \mathbb{R}^F$ = input features of node $i$

#### Step 2: Compute Raw Attention Coefficients

For each edge $(i,j)$, compute an attention logit using a shared attention mechanism:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T[\mathbf{h}_i' \, || \, \mathbf{h}_j']\right)$$

Where:
- $\mathbf{a} \in \mathbb{R}^{2F'}$ is a learnable attention vector
- $[\ \cdot \, || \, \cdot \ ]$ denotes concatenation
- $\text{LeakyReLU}$ is a non-linearity

**Alternative formulation** (sometimes used):
$$e_{ij} = \mathbf{a}_1^T \text{LeakyReLU}(\mathbf{W}_1 \mathbf{h}_i) + \mathbf{a}_2^T \text{LeakyReLU}(\mathbf{W}_2 \mathbf{h}_j)$$

This is computationally more efficient as it doesn't require concatenation.

#### Step 3: Normalize with Softmax

Apply softmax normalization over the neighborhood of node $i$:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(e_{ik})}$$

Where:
- $\alpha_{ij} \in [0,1]$ is the normalized attention weight
- Normalization ensures $\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} = 1$
- The neighborhood typically includes self-loops

### Key Properties of Attention Weights

1. **Local**: Only depends on the source node and target node
2. **Symmetric in formulation**: Same attention mechanism for all edges
3. **Data-dependent**: Changes based on node features
4. **Learnable**: The attention vector $\mathbf{a}$ is learned during training
5. **Interpretable**: High attention weights indicate important neighbors

---

## Multi-Head Attention

### Motivation

Similar to multi-head attention in Transformers, using multiple independent attention heads allows the model to:
- Capture different types of relationships simultaneously
- Provide representational richness
- Stabilize the learning process
- Attend to different representation subspaces

### Implementation

For $K$ attention heads:

1. Apply $K$ different linear transformations: $\mathbf{W}^{(k)}$ for head $k=1,\ldots,K$
2. Compute attention coefficients independently for each head: $\alpha_{ij}^{(k)}$
3. Aggregate features independently: $\mathbf{h}_i'^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j$

### Combining Multi-Head Outputs

**In intermediate layers (concatenation)**:
$$\mathbf{h}_i' = \|_{k=1}^{K} \mathbf{h}_i'^{(k)}$$

Output dimension: $K \times F'$

**In the final layer (averaging or concatenation)**:
- **Averaging**: $\mathbf{h}_i' = \frac{1}{K} \sum_{k=1}^{K} \mathbf{h}_i'^{(k)}$
- **Concatenation**: $\mathbf{h}_i' = \|_{k=1}^{K} \mathbf{h}_i'^{(k)}$

### Benefits of Multi-Head Attention

| Aspect | Benefit |
|--------|---------|
| **Representational Power** | Multiple subspaces capture different graph properties |
| **Robustness** | Different heads learn complementary patterns |
| **Stability** | Redundancy helps with gradient flow |
| **Ensemble Effect** | Multiple perspectives reduce overfitting |

---

## Mathematical Formulation

### Complete GAT Layer

For a single attention head, a GAT layer computes:

$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)$$

Where:
$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]))}$$

And:
- $\sigma$ is an activation function (e.g., ReLU, ELU, or identity in final layer)
- $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is the weight matrix
- $\mathbf{a} \in \mathbb{R}^{2F'}$ is the attention vector

### Multi-Head GAT Layer

$$\mathbf{h}_i' = \|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)$$

### Full Network

For an L-layer GAT:

$$\mathbf{H}^{(l+1)} = \text{GAT}(\mathbf{H}^{(l)}, \mathcal{A})$$

Where $\mathbf{H}^{(l)}$ are node features at layer $l$.

---

## Advantages over GCN

### Comparison Table

| Feature | GCN | GAT |
|---------|-----|-----|
| **Aggregation Weights** | Fixed (degree-based) | Learned (data-dependent) |
| **Flexibility** | All neighbors weighted equally | Dynamic importance weights |
| **Expressiveness** | Limited by fixed aggregation | Higher expressiveness |
| **Edge Features** | Cannot directly use | Can incorporate via attention |
| **Computational Efficiency** | $O(E)$ | $O(E)$ (both linear in edges) |
| **Interpretability** | Hard to understand aggregation | Attention weights are interpretable |
| **Inductive Learning** | Limited | Better generalization to unseen nodes |
| **Neighborhood Heterogeneity** | Cannot adapt to different neighborhoods | Adapts to neighborhood structure |

### Key Advantages

#### 1. **Learned, Task-Specific Weights**
- GCN: Uses fixed normalization by degree
- GAT: Learns which neighbors matter most
- Result: Better suited to heterogeneous graphs

#### 2. **Inductive Learning**
- GAT can generalize to unseen nodes
- Doesn't require the full graph during inference
- Attention mechanism works for any neighborhood size

#### 3. **Improved Performance**
- Often achieves better accuracy on citation networks
- Especially effective on heterogeneous data
- Better handling of noisy edges

#### 4. **Interpretability**
- Attention weights show which neighbors influenced each prediction
- Enable visualization of learned patterns
- Support explanation of model decisions

#### 5. **Flexibility**
- Can be extended to handle edge features
- Multi-head attention provides expressiveness
- Works with directed and undirected graphs

### Computational Complexity Comparison

```
GCN:  Complexity = O(N*F*F') + O(E*F')
      Memory = O(N*F') + sparse adjacency matrix

GAT:  Complexity = O(N*F*F') + O(E*F') + O(E) for attention
      Memory = O(N*F') + sparse attention weights (learned per sample)
```

In practice, both have similar computational costs, but GAT has higher constant factors due to attention computation.

---

## Computational Complexity

### Time Complexity Analysis

#### Per Layer Computation

**Input**:
- Graph with $N$ nodes and $E$ edges
- Node features: $\mathbf{H} \in \mathbb{R}^{N \times F}$
- Input feature dimension: $F$
- Output feature dimension: $F'$

**Operations**:

1. **Linear transformation**: $O(N \cdot F \cdot F')$
   - Transform all node features

2. **Attention computation**: $O(E \cdot F')$
   - For each edge, compute attention logit
   - Self-loops add $O(N \cdot F')$

3. **Softmax normalization**: $O(E \cdot \log(\text{max degree}))$
   - Per-node normalization can be efficient with sparse operations

4. **Aggregation**: $O(E \cdot F')$
   - Weighted sum over neighborhood edges

**Total per layer**: $O(N \cdot F \cdot F' + E \cdot F')$

For multi-head attention with $K$ heads: Multiply by $K$

### Space Complexity Analysis

| Component | Complexity |
|-----------|------------|
| **Node features** | $O(N \cdot F')$ |
| **Weight matrix** | $O(F \cdot F')$ |
| **Attention parameters** | $O(F')$ per head |
| **Attention weights** (cached) | $O(E)$ (sparse) |
| **Gradients** | Same as forward pass |

### Comparison with GCN

```
Operation          GCN              GAT
────────────────────────────────────────────
Feature transform  O(N*F*F')        O(N*F*F')
Message passing    O(E*F')          O(E*F')
Normalization      O(E) or O(N)     O(E*log D) where D=degree
Total              O(N*F*F'+E*F')   O(N*F*F'+E*F'+E*log D)
```

### Practical Considerations

1. **Dense computation vs. sparse**:
   - Dense attention: Fast on GPUs for small graphs
   - Sparse attention: Better for large, sparse graphs

2. **Memory bottleneck**: Often the feature transformation ($O(N \cdot F \cdot F')$)

3. **Multi-head penalty**: $K$ heads multiply computational cost by $K$

4. **Scalability**:
   - GCN generally more efficient for very large graphs
   - GAT competitive on medium-sized graphs (thousands to millions of nodes)

---

## Applications

### Recommended Use Cases

1. **Heterogeneous Graphs**
   - Different node/edge types need different aggregation weights
   - Example: Social networks with multiple relationship types

2. **Sparse Graphs with Important Edges**
   - Some edges carry more information than others
   - Example: Citation networks where certain citations matter more

3. **Inductive Learning**
   - Need to generalize to unseen nodes
   - Example: Cold-start recommendations

4. **Interpretability Required**
   - Attention weights explain model decisions
   - Example: Biomedical networks, knowledge graphs

### Real-World Examples

- **Citation Networks**: Predict paper categories (Cora, Citeseer)
- **Social Networks**: Friend recommendation, influence prediction
- **Molecular Graphs**: Property prediction, drug discovery
- **Knowledge Graphs**: Entity classification, relation extraction
- **Recommendation Systems**: Learning user-item interactions

---

## Summary

### Key Takeaways

1. **Attention Mechanism**: Learns task-specific weights for graph aggregation
   - Dynamic, data-dependent weights
   - Based on node pair similarity

2. **Architecture**: Three main components
   - Linear transformation of features
   - Attention coefficient computation
   - Weighted aggregation

3. **Attention Coefficients**: Computed via
   - Concatenation of transformed features
   - Non-linear scoring (LeakyReLU)
   - Softmax normalization per neighborhood

4. **Multi-Head Attention**: Multiple independent attention heads
   - Captures diverse relationships
   - Improves model capacity
   - Provides robustness

5. **Advantages over GCN**:
   - Learned vs. fixed weights
   - Better expressiveness
   - Inductive capability
   - Interpretability

6. **Computational Cost**: Similar to GCN
   - Linear in number of edges
   - Practical for medium-to-large graphs
   - Multi-head multiplies cost by number of heads

### When to Use GAT

- Heterogeneous neighborhoods with varying importance
- Need for interpretable aggregation
- Inductive learning scenarios
- Graphs where some edges are more important
- Need for strong node classification performance

### When to Consider GCN Instead

- Very large graphs (billions of nodes/edges)
- Fixed, uniform aggregation is appropriate
- Computational efficiency is paramount
- Simpler model required for interpretability

---

## Further Reading

1. Veličković et al. (2018). "Graph Attention Networks." ICLR 2018
2. Multi-head attention: Transformer architecture papers
3. Spectral methods vs. spatial methods in GNNs
4. Advanced attention mechanisms (e.g., multi-hop attention)

