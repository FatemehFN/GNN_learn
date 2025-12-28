# Lesson 5: Advanced GNN Architectures

## Table of Contents
1. [Introduction](#introduction)
2. [GraphSAGE: Inductive Representation Learning](#graphsage)
3. [Graph Isomorphism Networks (GIN)](#gin)
4. [Comparison of Advanced Architectures](#comparison)
5. [Sampling Strategies](#sampling)
6. [Architecture Selection Guide](#selection-guide)
7. [Advanced Topics](#advanced-topics)
8. [Summary](#summary)

---

## Introduction

After mastering basic GNN architectures like GCN and GAT, we explore advanced methods that address specific challenges:

- **Inductive Learning**: Training on unseen nodes and graphs
- **Expressiveness**: Understanding GNN representational power
- **Scalability**: Handling large graphs efficiently
- **Flexibility**: Different aggregation and update strategies

This lesson covers two influential architectures that shaped modern GNN development.

---

## GraphSAGE: Inductive Representation Learning

### Overview

**GraphSAGE** (Graph SAmple and aggreGatE) is an inductive learning framework that generates embeddings for previously unseen nodes. Unlike transductive methods (like GCN) that require all nodes during training, GraphSAGE learns a function that maps node features to embeddings.

### Key Innovations

1. **Inductive Learning**: Learn node embedding functions, not fixed embeddings
2. **Neighborhood Sampling**: Efficient computation via mini-batch sampling
3. **Flexible Aggregation**: Multiple aggregator functions for neighborhoods
4. **Scalability**: Process large graphs without full-batch training

### Mathematical Formulation

#### Forward Propagation

For layer $k$, GraphSAGE follows these steps:

**Step 1: Neighborhood Sampling**
```
N_v^k = Sample(N(v), sample_size)
```
where $N(v)$ is the neighborhood of node $v$ and sample_size is the number of samples.

**Step 2: Aggregate Neighbor Embeddings**
```
h_N(v)^k = AGGREGATE({h_u^{k-1} : u ∈ N_v^k})
```

**Step 3: Update Node Embedding**
```
h_v^k = σ(W^k · [h_v^{k-1} || h_N(v)^k])
```

where:
- $||$ denotes concatenation
- $W^k$ is a learnable weight matrix
- $σ$ is a non-linearity (ReLU)

### Aggregation Functions

GraphSAGE proposes multiple aggregators:

#### 1. Mean Aggregator
```
h_N(v)^k = 1/|N_v^k| Σ_{u ∈ N_v^k} h_u^{k-1}
```
Simple average of neighbor embeddings.

#### 2. LSTM Aggregator
```
h_N(v)^k = LSTM([h_{u1}^{k-1}, h_{u2}^{k-1}, ...])
```
Applies LSTM sequentially over neighbor embeddings. Order matters.

#### 3. Pooling Aggregator
```
h_N(v)^k = max_pool({MLP(h_u^{k-1}) : u ∈ N_v^k})
```
Element-wise max pooling over neighbor features transformed by MLP.

### Training Procedure

**Input**: Graph $G = (V, E)$, node features $X$, labels for subset $S ⊂ V$

**Mini-batch Stochastic Gradient Descent**:

```
for epoch = 1 to num_epochs:
    for batch B ⊂ S:
        1. Sample neighborhoods N_v^k for all v in extended batch
        2. Forward pass: compute embeddings h_v^L for v in B
        3. Compute loss: L = Σ_{v ∈ B} loss(ŷ_v, y_v)
        4. Backward pass: gradient descent update
```

### Inductive vs Transductive Learning

| Aspect | Inductive (GraphSAGE) | Transductive (GCN) |
|--------|----------------------|-------------------|
| **Training** | Learn aggregation functions | Learn fixed embeddings |
| **Inference** | Apply learned functions to new nodes | Only works with seen nodes |
| **Scalability** | Efficient mini-batching | Requires full graph |
| **Generalization** | Generalizes to unseen graphs | Cannot generalize beyond training set |
| **Use Case** | Production systems, evolving graphs | Static graph analysis |

### Advantages and Limitations

**Advantages**:
- Handles inductive settings naturally
- Efficient mini-batch training
- Generalizes to unseen nodes and graphs
- Multiple aggregation strategies for flexibility

**Limitations**:
- Sampling variance can affect training
- Hyperparameter sensitivity (layer sizes, sample sizes)
- Performance depends heavily on neighborhood quality
- May miss important long-range dependencies

---

## Graph Isomorphism Networks (GIN)

### Overview

**GIN** (Graph Isomorphism Networks) is a theoretically grounded architecture that maximizes GNN expressiveness. It addresses the question: "How powerful can GNNs be?"

### Theoretical Background

#### Weisfeiler-Lehman (WL) Test

The Weisfeiler-Lehman test is a classical graph isomorphism heuristic:

**Algorithm**:
```
1. Assign each node a unique initial color c_0(v) = features(v)
2. For iteration k = 1 to K:
   a. For each node v:
      - Collect colors of neighbors: C(v) = {c_{k-1}(u) : u ∈ N(v)}
      - Define c_k(v) = hash(c_{k-1}(v), SORT(C(v)))
3. Return multiset of final colors
```

Two graphs are distinguished by WL test if they have different final color multisets.

#### WL-Equivalence and GNN Expressive Power

**Theorem**: A GNN can distinguish graphs that the WL test can distinguish if its aggregation function is injective.

### Mathematical Formulation

GIN is designed to match WL test expressiveness using injective aggregation.

#### GIN Layer

```
h_v^{k+1} = MLP^k((1 + ε_k) · h_v^k + Σ_{u ∈ N(v)} h_u^k)
```

where:
- $ε_k$ is a learnable parameter (or fixed, e.g., 0)
- $MLP^k$ is a multi-layer perceptron
- $(1 + ε_k) · h_v^k$ ensures node's own embedding is distinct from aggregation

#### Injective Aggregation Guarantee

The sum aggregation with MLPs is provably injective:

**Lemma**: If $f$ and $h$ are injective and continuous, then:
```
x ↦ f(x + Σ_{i=1}^n h(y_i))
```
is injective in $x$ and all $y_i$ (under mild conditions).

### WL Expressiveness Analysis

#### WL Expressiveness Hierarchy

GIN can match different WL variants:

| Variant | Method | Expressive Power |
|---------|--------|------------------|
| **Standard WL** | Sum aggregation | Distinguishes regular structures |
| **k-WL** | More complex aggregation | Higher-order structural patterns |
| **Subgraph WL** | Node labeling + edge features | Very high expressiveness |

### Implementation Details

#### GIN Architecture

1. **Input**: Node features $x_v ∈ ℝ^{d_{in}}$
2. **Hidden Layers**: For $k = 0$ to $K-1$:
   ```
   a_v^{k+1} = Σ_{u ∈ N(v) ∪ {v}} h_u^k
   h_v^{k+1} = ReLU(W_1[h_v^k || (a_v^{k+1} - h_v^k)])
   ```
3. **Readout**: Graph-level representation (if needed):
   ```
   h_G = Σ_v h_v^K
   ```

### Expressive Power Limitations

GIN can distinguish structures that WL test can distinguish, but:

- **Cannot distinguish**: Some non-isomorphic graphs with same degree distribution
- **Limitation**: Limited to local structural patterns
- **Extension**: Higher-order WL tests or subgraph features improve expressiveness

### Advantages and Limitations

**Advantages**:
- Theoretically grounded expressiveness guarantees
- Simple, clean mathematical formulation
- Competitive empirical performance
- Efficient computation (sum aggregation)

**Limitations**:
- Limited by WL test expressiveness (still limited)
- May oversmooth with many layers
- Simple aggregation may miss important information
- Requires careful MLP design

---

## Comparison of Advanced Architectures

### GraphSAGE vs GIN

#### Design Philosophy

| Property | GraphSAGE | GIN |
|----------|-----------|-----|
| **Goal** | Inductive learning + scalability | Theoretical expressiveness |
| **Primary Innovation** | Neighborhood sampling | Injective aggregation |
| **Flexibility** | Multiple aggregators | Standard aggregation |
| **Theory** | Empirical | Theoretically grounded |

#### Computational Complexity

**GraphSAGE** (per layer):
```
Time: O(|S| · sample_size · d_{in} · d_{out})
Space: O(sample_size · d_{in})
S = batch size
```

**GIN** (per layer):
```
Time: O(|E| · d_{in} · d_{out}) [for full batch]
Space: O(|V| · d_{out})
```

#### Empirical Performance

| Dataset | Task | GraphSAGE | GIN | Notes |
|---------|------|-----------|-----|-------|
| Citation Networks | Node Classification | ~90% | ~91% | Similar performance |
| Social Networks | Link Prediction | ~94% | ~92% | GraphSAGE better |
| Graph Classification | Graph-level | ~73% | ~75% | GIN better |
| Large Graphs | Scalability | Excellent | Good | GraphSAGE wins |

### Advanced Architectures Overview

#### GraphSAGE
- **Inductive**: Yes
- **Scalable**: Excellent (mini-batch)
- **Theoretical guarantees**: No
- **Aggregation**: Flexible
- **Best for**: Large graphs, inductive settings

#### GIN
- **Inductive**: No (but can be adapted)
- **Scalable**: Moderate (standard training)
- **Theoretical guarantees**: Yes (WL expressiveness)
- **Aggregation**: Standard (sum)
- **Best for**: Graph-level tasks, when theory matters

#### Other Notable Architectures

**PinSAGE** (Pinterest): Scalable variant with importance sampling
**GraphSAINT**: Mini-batch sampling with variance reduction
**FastGCN**: Fast learning via importance sampling
**Subgraph Sampling**: Balance variance and compute

---

## Sampling Strategies

### Motivation

Full-batch training on large graphs is computationally expensive:
- Memory: Store all node embeddings
- Computation: Process entire graph per iteration
- Time: Training on billions of nodes infeasible

Sampling reduces complexity while maintaining performance.

### Neighborhood Sampling (GraphSAGE)

#### Uniform Sampling

Sample $K$ neighbors uniformly at random:
```
N_v^k = Sample_uniform(N(v), K)
```

**Pros**: Simple, unbiased
**Cons**: May miss important nodes, high variance

#### Importance Sampling

Sample neighbors with probability proportional to edge weights or importance scores:
```
P(u) ∝ importance(u, v)
N_v^k = Sample(N(v), K, P)
```

**Pros**: Better captures important structures
**Cons**: Requires precomputed importance scores

### Layer-wise Sampling (GraphSAINT)

Sample subgraphs and perform layer-wise aggregation:

```
1. Sample subgraph S from G
2. For each layer:
   a. Compute aggregate on S
   b. Adjust node degrees to correct for sampling bias
3. Aggregate predictions across subgraphs
```

**Advantages**: Better variance reduction, correlates with graph structure

### Node-wise vs Layer-wise Sampling

#### Node-wise (GraphSAGE)
```
For each node v:
  For each layer k:
    Sample neighbors N_v^k
```
Each node has independent samples per layer.

#### Layer-wise (GraphSAINT)
```
For each layer k:
  Sample node set V_k
```
Same node set used across layers.

**Trade-off**: Node-wise more flexible, layer-wise better variance

### Importance Sampling Schemes

#### Popularity-based (PinSAGE)
Sample neighbors by click/interaction popularity:
```
P(u) = clicks(u, v) / Σ_w clicks(w, v)
```

#### Degree-based
Sample neighbors proportionally to their degree:
```
P(u) = degree(u) / Σ_w degree(w)
```

#### Personalized PageRank
Sample using personalized PageRank scores:
```
P(u) = PageRank(u, v) / Σ_w PageRank(w, v)
```

### Sampling Variance Analysis

#### Variance Reduction

Uniform sampling variance:
```
Var[ĥ_v^k] ≈ O(|N(v)|² / K)
```

With importance sampling:
```
Var[ĥ_v^k] ≈ O(min(|N(v)|²/K, variance(importance)))
```

**Key insight**: Importance sampling reduces variance when importance scores correlate with contribution to final loss.

### Mini-batch Sampling Pipeline

```python
# Pseudocode for mini-batch sampling
def sample_neighbors(node_id, k_hops, sample_size):
    """Sample k-hop neighborhood of node"""
    frontier = {node_id}
    for hop in range(k_hops):
        next_frontier = set()
        for node in frontier:
            samples = sample(neighbors(node), sample_size)
            next_frontier.update(samples)
        frontier.update(next_frontier)
    return frontier

def create_mini_batch(nodes, k_hops, sample_size):
    """Create mini-batch with sampled neighborhoods"""
    subgraph_nodes = set()
    subgraph_edges = []
    for node in nodes:
        neighbors = sample_neighbors(node, k_hops, sample_size)
        subgraph_nodes.update(neighbors)
        for u, v in edges_between(neighbors):
            subgraph_edges.append((u, v))
    return subgraph_nodes, subgraph_edges
```

---

## Architecture Selection Guide

### Decision Tree

```
Problem: How to choose architecture?

1. Is your graph static (won't change)?
   → Yes: Skip inductive-specific architectures

2. Do you need to generalize to unseen nodes?
   → Yes: Use GraphSAGE or adapt GIN inductively
   → No: GCN/GAT may be sufficient

3. Is your graph very large (billions of nodes)?
   → Yes: GraphSAGE with proper sampling
   → No: GIN or other options

4. Do you need theoretical guarantees?
   → Yes: GIN (WL expressiveness)
   → No: GraphSAGE (empirical performance)

5. Task type?
   → Node classification: GraphSAGE or GCN
   → Link prediction: GraphSAGE (inductive)
   → Graph classification: GIN
   → Graph generation: Specialized models
```

### Selection Criteria

#### For Node Classification

**Use GraphSAGE if**:
- Need to handle new nodes post-training
- Graph is very large
- Node features are informative
- Budget allows mini-batch training

**Use GIN if**:
- All nodes known at training time
- Need theoretical expressiveness guarantees
- Want simpler implementation
- Graph-level patterns important

**Use GCN if**:
- Simple baselines are preferred
- Graph is small-medium size
- No special requirements

#### For Link Prediction

**Use GraphSAGE if**:
- Need inductive link prediction
- Graph structure changes frequently
- Computational efficiency critical

**Use attention-based methods (GAT) if**:
- Want to understand which neighbors matter
- Graph has strong structural patterns

#### For Graph Classification

**Use GIN if**:
- Want maximum expressiveness
- Graph-level patterns important
- Theoretical guarantees valued

**Use GraphSAGE if**:
- Extremely large graphs
- Need fast inference
- Flexibility in aggregation important

### Scalability Considerations

#### Computational Resources

| Architecture | Memory | Training Time | Inference Time |
|-------------|--------|--------------|-----------------|
| **GCN** | O(\|V\|·d) | Medium | Fast |
| **GraphSAGE** | O(batch·sample·d) | Fast | Fast |
| **GIN** | O(\|V\|·d) | Medium | Fast |
| **GAT** | O(\|E\|·d) | Slow | Slow |

#### Scalability Strategies

1. **Layer Reduction**: Fewer layers, more selective aggregation
2. **Feature Reduction**: Lower embedding dimensions
3. **Subgraph Methods**: Process graph partitions separately
4. **Knowledge Distillation**: Train large, infer small
5. **Mixed Precision**: Use float16 for some computations

---

## Advanced Topics

### Variants and Extensions

#### GraphSAGE Variants

**Unsupervised GraphSAGE**: Learn embeddings without labels using negative sampling
```
L = -log(σ(h_u · h_v)) - Σ_{neg} log(1 - σ(h_u · h_neg))
```

**Heterogeneous GraphSAGE**: Handle graphs with multiple node/edge types
```
h_v^{k+1} = σ(Σ_{rel} W_{rel}^k · AGGREGATE_rel(N_v^k))
```

#### GIN Variants

**WL-GIN**: Explicitly incorporate WL node coloring
```
c_v^{k+1} = hash(c_v^k, multiset({c_u^k : u ∈ N(v)}))
h_v^{k+1} = MLP(h_v^k + Σ_u h_u^k)
```

**Subgraph GIN**: Use subgraph features for higher expressiveness
```
features(v) ← subgraph isomorphism counts around v
```

### Combining Architectures

**Hybrid Models**:
- GraphSAGE aggregation + GIN updates
- Attention-weighted sampling in GraphSAGE
- Hierarchical: Use GIN for local regions, GraphSAGE for global

### Handling Dynamic Graphs

Both architectures can be adapted for dynamic settings:

**Temporal GraphSAGE**: Include temporal information in sampling
**Temporal GIN**: Update node colors based on temporal patterns

---

## Summary

### Key Takeaways

1. **GraphSAGE** enables inductive learning through neighborhood sampling and flexible aggregation
2. **GIN** maximizes expressiveness matching the Weisfeiler-Lehman test
3. **Sampling strategies** are crucial for scaling GNNs to large graphs
4. **Architecture choice** depends on problem requirements (inductive, scalability, theory)
5. **Trade-offs** exist between expressiveness, scalability, and theoretical guarantees

### Learning Path

1. Understand GraphSAGE's sampling mechanism deeply
2. Study WL test and its connection to GNN expressiveness
3. Implement both architectures from scratch
4. Experiment with different sampling strategies
5. Apply to real-world problems (node/link/graph classification)

### Further Reading

**Papers**:
- Hamilton et al. "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- Xu et al. "How Powerful are Graph Neural Networks?" (GIN)
- Zeng et al. "GraphSAINT: Graph Sampling Based Inductive Learning Method"

**Topics to Explore**:
- Higher-order GNN expressiveness (k-WL test)
- Heterogeneous graphs (HAN, HGT)
- Temporal graphs (DyRep, EvolveGCN)
- Knowledge graph embeddings
- Graph matching and similarity
