# Lesson 3: Message Passing & GNN Foundations

## Introduction

**Message Passing** is the fundamental operation in Graph Neural Networks. It's how nodes communicate with their neighbors to learn meaningful representations.

Think of it like this: each node gathers information from its neighbors, processes it, and updates its own representation. This happens iteratively across the entire graph.

---

## The Message Passing Framework

### Core Concept

For each node in a graph, we want to create a representation (embedding) that captures both:
1. **Its own features**
2. **Information from its neighborhood**

### The Three-Step Process

At each layer k, message passing consists of:

1. **MESSAGE**: Create messages from neighbors
2. **AGGREGATE**: Combine messages from all neighbors
3. **UPDATE**: Update node representation

---

## Mathematical Formulation

### General Message Passing

For node v at layer k:

**h_v^(k) = UPDATE^(k)(h_v^(k-1), AGGREGATE^(k)({MESSAGE^(k)(h_v^(k-1), h_u^(k-1)) : u ∈ N(v)}))**

Where:
- **h_v^(k)**: Representation of node v at layer k
- **N(v)**: Neighbors of node v
- **MESSAGE**: Function that creates messages
- **AGGREGATE**: Function that combines messages
- **UPDATE**: Function that updates node representation

### Simplified Form

**h_v^(k) = σ(W^(k) · AGGREGATE({h_u^(k-1) : u ∈ N(v)}))**

Where:
- **W^(k)**: Learnable weight matrix at layer k
- **σ**: Non-linear activation function (ReLU, tanh, etc.)

---

## Message Functions

### Basic Message

Simply pass neighbor's representation:

**m_u→v = h_u**

### Edge-Weighted Message

Include edge information:

**m_u→v = w(u,v) · h_u**

### Learned Message

Apply neural network:

**m_u→v = MLP(h_u, h_v, e_uv)**

Where:
- **h_u**: Source node features
- **h_v**: Target node features
- **e_uv**: Edge features

---

## Aggregation Functions

The aggregation function must be **permutation invariant**: the order of neighbors shouldn't matter.

### 1. Sum Aggregation

**AGGREGATE({h_u : u ∈ N(v)}) = ∑_{u ∈ N(v)} h_u**

**Properties**:
- Simple and efficient
- Preserves information
- Can distinguish different multisets

**Issue**: Affected by node degree (high-degree nodes have larger values)

### 2. Mean Aggregation

**AGGREGATE({h_u : u ∈ N(v)}) = 1/|N(v)| ∑_{u ∈ N(v)} h_u**

**Properties**:
- Normalized by degree
- Each neighbor contributes equally
- Less affected by degree

**Issue**: Loses information about neighborhood size

### 3. Max Aggregation

**AGGREGATE({h_u : u ∈ N(v)}) = MAX({h_u : u ∈ N(v)})**

Element-wise maximum across all neighbor representations.

**Properties**:
- Captures most prominent features
- Invariant to neighborhood size

**Issue**: Can lose information (only keeps maximum)

### 4. Attention-Based Aggregation

**AGGREGATE({h_u : u ∈ N(v)}) = ∑_{u ∈ N(v)} α_uv · h_u**

Where α_uv are learned attention weights.

**Properties**:
- Learns which neighbors are important
- Adaptive weighting
- More expressive

We'll cover this in detail in Lesson 6 on Graph Attention Networks.

---

## Update Functions

### Concatenation + Linear

**h_v^(k) = σ(W · [h_v^(k-1) || m_v^(k)])**

Where:
- **||**: Concatenation
- **m_v^(k)**: Aggregated messages

### Addition + Linear

**h_v^(k) = σ(W_1 · h_v^(k-1) + W_2 · m_v^(k))**

### Gated Update (GRU-style)

**h_v^(k) = (1 - z_v) ⊙ h_v^(k-1) + z_v ⊙ h̃_v^(k)**

Where z_v is a learned gate.

---

## Permutation Invariance

### Why It Matters

Graphs have no inherent node ordering. Our functions must be **permutation invariant**:

**f({x_1, x_2, ..., x_n}) = f({x_π(1), x_π(2), ..., x_π(n)})**

For any permutation π.

### Permutation Invariant Functions

These operations are permutation invariant:
- **Sum**: ∑_i x_i
- **Mean**: 1/n ∑_i x_i
- **Max**: max_i x_i
- **Product**: ∏_i x_i

These are NOT permutation invariant:
- **Concatenation**: [x_1, x_2, ..., x_n]
- **RNN**: processes sequentially

---

## Multi-Layer Message Passing

### Stacking Layers

Apply message passing multiple times:

**Layer 1**: h^(1) = MP^(1)(h^(0))
**Layer 2**: h^(2) = MP^(2)(h^(1))
**Layer K**: h^(K) = MP^(K)(h^(K-1))

Where MP^(k) is the message passing operation at layer k.

### Receptive Field

After k layers of message passing:
- Node v has information from k-hop neighbors
- **1 layer**: direct neighbors
- **2 layers**: neighbors and neighbors-of-neighbors
- **k layers**: k-hop neighborhood

### Visualization

```
Initial: [A] - [B] - [C] - [D]

After 1 layer:
- A knows about: A, B
- B knows about: A, B, C
- C knows about: B, C, D
- D knows about: C, D

After 2 layers:
- A knows about: A, B, C
- B knows about: A, B, C, D
- C knows about: A, B, C, D
- D knows about: B, C, D

After 3 layers:
- All nodes know about: A, B, C, D
```

---

## Matrix Form

### Single Layer

**H^(k) = σ(Ã H^(k-1) W^(k))**

Where:
- **H^(k) ∈ ℝ^(n×d)**: Node representations at layer k
- **Ã**: Normalized adjacency matrix
- **W^(k) ∈ ℝ^(d×d')**: Learnable weights
- **σ**: Activation function

### With Self-Loops

**H^(k) = σ((A + I)H^(k-1) W^(k))**

Or with normalization:

**H^(k) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(k-1) W^(k))**

Where Ã = A + I

---

## Node-Level, Edge-Level, and Graph-Level Tasks

### Node-Level Tasks

Predict properties of individual nodes.

**Output**: h_v^(K) for each node v

**Examples**:
- Node classification (predict user category)
- Node regression (predict protein function score)

### Edge-Level Tasks

Predict properties of edges or node pairs.

**Output**: Combine node representations

**Methods**:
- Concatenation: [h_u || h_v]
- Dot product: h_u^T h_v
- Learned function: MLP(h_u, h_v)

**Examples**:
- Link prediction (will edge exist?)
- Edge classification (type of relationship)

### Graph-Level Tasks

Predict properties of entire graphs.

**Output**: Single vector per graph

**Readout Function** (must be permutation invariant):
- **Sum**: h_G = ∑_{v ∈ V} h_v
- **Mean**: h_G = 1/|V| ∑_{v ∈ V} h_v
- **Max**: h_G = MAX({h_v : v ∈ V})
- **Attention**: h_G = ∑_{v ∈ V} α_v h_v

**Examples**:
- Molecule property prediction
- Graph classification

---

## Over-Smoothing Problem

### The Issue

With many layers, node representations become too similar:

**h_v^(K) ≈ h_u^(K)** for all v, u

### Why It Happens

- Each layer mixes neighbor features
- After many layers, all nodes see the entire graph
- Representations converge to the same value

### Solutions

1. **Limit depth**: Use fewer layers (typically 2-4)
2. **Skip connections**: h^(k) = h^(k-1) + f(h^(k-1))
3. **Node-specific transformations**: Different weights per node
4. **Jumping knowledge**: Combine representations from all layers

---

## Over-Squashing Problem

### The Issue

Information from distant nodes gets "squashed" through bottleneck nodes.

### Example

```
[A] - [B] - [C] - [D]
      ↑
      [E]
```

Information from A, D must pass through B to reach E. B becomes an information bottleneck.

### Solutions

1. **More layers**: Allow information to flow
2. **Virtual nodes**: Add global node connected to all nodes
3. **Graph rewiring**: Add skip connections in graph
4. **Higher-order methods**: Use subgraphs instead of nodes

---

## Expressive Power

### Weisfeiler-Leman Test

The **Weisfeiler-Leman (WL) test** is an algorithm for graph isomorphism testing.

**Key Result**: Message Passing GNNs are at most as powerful as the 1-WL test.

This means there exist non-isomorphic graphs that GNNs cannot distinguish.

### Graph Isomorphism Network (GIN)

GIN is provably as powerful as 1-WL test:

**h_v^(k) = MLP^(k)((1 + ε^(k)) · h_v^(k-1) + ∑_{u ∈ N(v)} h_u^(k-1))**

Where ε is either learned or fixed.

---

## Practical Considerations

### Normalization

**Batch Normalization**:
- Normalize features across batch
- Helps training stability

**Layer Normalization**:
- Normalize features per node
- Better for graphs with varying sizes

**Graph Normalization**:
- Normalize per graph in batch
- Useful for graph-level tasks

### Dropout

Apply dropout to:
- Node features
- Edge connections (DropEdge)
- Attention weights

### Residual Connections

**h^(k) = h^(k-1) + f(h^(k-1))**

Benefits:
- Prevents over-smoothing
- Easier gradient flow
- Allows deeper networks

---

## Summary

In this lesson, we covered:

1. **Message passing framework**: MESSAGE, AGGREGATE, UPDATE
2. **Aggregation functions**: Sum, mean, max, attention
3. **Permutation invariance**: Why and how
4. **Multi-layer GNNs**: Receptive fields and depth
5. **Tasks**: Node, edge, and graph-level
6. **Challenges**: Over-smoothing and over-squashing
7. **Expressive power**: WL test and limitations

---

## Key Takeaways

- Message passing is the core of GNNs
- Aggregation must be permutation invariant
- Number of layers = receptive field size
- More layers ≠ always better (over-smoothing)
- Choice of aggregation affects what the network can learn

---

## Further Reading

1. **Gilmer et al. (2017)**: "Neural Message Passing for Quantum Chemistry"
2. **Xu et al. (2019)**: "How Powerful are Graph Neural Networks?" (GIN paper)
3. **Hamilton (2020)**: "Graph Representation Learning" (Chapter 3)

---

## Exercises

1. Implement message passing from scratch using numpy
2. Prove that sum, mean, and max are permutation invariant
3. Calculate receptive field size for different graph structures
4. Analyze how over-smoothing affects node representations
5. Compare different aggregation functions empirically

---

**Next Lesson**: We'll implement our first real GNN - Graph Convolutional Networks (GCNs)!
