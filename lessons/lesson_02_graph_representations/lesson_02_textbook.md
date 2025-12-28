# Lesson 2: Graph Representations

## Introduction

To work with graphs computationally, especially for machine learning, we need efficient ways to represent them in memory. The choice of representation affects:
- Memory usage
- Computational efficiency
- Ease of implementation
- Suitability for different algorithms

In this lesson, we'll explore different graph representations and learn when to use each one.

---

## Core Graph Representations

### 1. Adjacency Matrix

An **adjacency matrix** is a square matrix A where:
- Rows and columns represent nodes
- $A[i, j] = 1$ if there's an edge from node i to node j
- $A[i, j] = 0$ otherwise

**For undirected graphs**: A is symmetric ($A[i, j] = A[j, i]$)

**For weighted graphs**: $A[i, j]$ = weight of edge $(i, j)$

#### Mathematical Definition

For a graph $G = (V, E)$ with $n = |V|$ nodes:

$$A \in \mathbb{R}^{n \times n}$$

where:

$$A[i, j] = \begin{cases}
1 & \text{if } (i, j) \in E \text{ (unweighted)} \\
w(i, j) & \text{if } (i, j) \in E \text{ (weighted)} \\
0 & \text{otherwise}
\end{cases}$$

#### Example

For a graph with edges: {(0,1), (0,2), (1,2), (2,3)}

```
    0  1  2  3
0 [ 0  1  1  0 ]
1 [ 1  0  1  0 ]
2 [ 1  1  0  1 ]
3 [ 0  0  1  0 ]
```

#### Advantages
- $O(1)$ edge lookup: Check if edge exists in constant time
- Easy to implement matrix operations
- Natural for dense graphs
- Supports weighted graphs naturally

#### Disadvantages
- $O(n^2)$ space complexity (wasteful for sparse graphs)
- $O(n^2)$ time to iterate all edges
- Most real-world graphs are sparse ($|E| \ll n^2$)

#### Space Complexity Analysis

- **Dense graph** ($|E| \approx n^2$): Efficient
- **Sparse graph** ($|E| \approx n$): Wasteful (stores many zeros)

For example, a social network with 1 million users and average 100 friends:
- Adjacency matrix: $10^{12}$ entries (mostly zeros)
- Actual edges: $10^8$

---

### 2. Edge List

An **edge list** is a list of all edges in the graph.

**For unweighted**: List of tuples (u, v)

**For weighted**: List of tuples (u, v, w)

#### Mathematical Definition

$$E = [(u_1, v_1), (u_2, v_2), \ldots, (u_m, v_m)]$$

where $m = |E|$

#### Example

For the same graph:
```python
edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
```

With weights:
```python
edges = [(0, 1, 0.5), (0, 2, 0.3), (1, 2, 0.7), (2, 3, 0.9)]
```

#### Advantages
- $O(m)$ space complexity where $m = |E|$
- Compact for sparse graphs
- Easy to iterate over all edges
- Simple to implement

#### Disadvantages
- $O(m)$ edge lookup time
- $O(m)$ to find neighbors of a node
- Inefficient for many graph algorithms

---

### 3. Adjacency List

An **adjacency list** stores for each node a list of its neighbors.

#### Mathematical Definition

For each $v \in V$:
- $\text{Adj}[v] = \{u \in V : (v, u) \in E\}$

#### Example

For the same graph:
```python
adj_list = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2]
}
```

With weights:
```python
adj_list = {
    0: [(1, 0.5), (2, 0.3)],
    1: [(0, 0.5), (2, 0.7)],
    2: [(0, 0.3), (1, 0.7), (3, 0.9)],
    3: [(2, 0.9)]
}
```

#### Advantages
- $O(n + m)$ space complexity
- Efficient neighbor lookup: $O(\deg(v))$ for node $v$
- Good for sparse graphs
- Natural for many algorithms (BFS, DFS)

#### Disadvantages
- $O(\deg(v))$ edge lookup time
- Slightly more complex to implement than edge list

---

### 4. Incidence Matrix

An **incidence matrix** B has:
- Rows representing nodes
- Columns representing edges
- B[i, j] indicates if node i is incident to edge j

#### Mathematical Definition

For graph $G = (V, E)$ with n nodes and m edges:

$$B \in \mathbb{R}^{n \times m}$$

For undirected graphs:
$$B[i, j] = \begin{cases}
1 & \text{if node } i \text{ is incident to edge } j \\
0 & \text{otherwise}
\end{cases}$$

For directed graphs:
$$B[i, j] = \begin{cases}
1 & \text{if edge } j \text{ leaves node } i \\
-1 & \text{if edge } j \text{ enters node } i \\
0 & \text{otherwise}
\end{cases}$$

#### Example (Undirected)

Edges: $e_1=(0,1)$, $e_2=(0,2)$, $e_3=(1,2)$, $e_4=(2,3)$

```
      e_1  e_2  e_3  e_4
  0 [  1    1    0    0  ]
  1 [  1    0    1    0  ]
  2 [  0    1    1    1  ]
  3 [  0    0    0    1  ]
```

#### Advantages
- Useful for certain algorithms (network flow)
- Explicit edge representation

#### Disadvantages
- $O(n \times m)$ space complexity
- Rarely used in practice for GNNs
- Inefficient for most operations

---

## Representations for Machine Learning

### Feature Matrices

For machine learning, we augment graph structure with node and edge features.

#### Node Feature Matrix

$$X \in \mathbb{R}^{n \times d}$$

where:
- $n$ = number of nodes
- $d$ = feature dimension
- $X[i, :]$ = feature vector for node $i$

#### Example

For a social network:
```python
# Node features: [age, num_posts, account_age_days]
X = [
    [25, 100, 365],   # User 0
    [30, 250, 730],   # User 1
    [28, 180, 500],   # User 2
    [35, 300, 1000]   # User 3
]
```

#### Edge Feature Matrix

$$E_{\text{feat}} \in \mathbb{R}^{m \times f}$$

where:
- $m$ = number of edges
- $f$ = edge feature dimension

#### Example

```python
# Edge features: [interaction_count, message_length_avg]
edge_features = [
    [50, 120],   # Edge 0
    [30, 80],    # Edge 1
    [100, 150],  # Edge 2
    [25, 60]     # Edge 3
]
```

---

## PyTorch Geometric Format

PyTorch Geometric uses a specific format optimized for GNNs:

### COO (Coordinate) Format

Edges stored as two lists (like edge list, but optimized):

```python
edge_index = torch.tensor([
    [0, 0, 1, 2],  # Source nodes
    [1, 2, 2, 3]   # Target nodes
], dtype=torch.long)
```

**Shape**: [2, num_edges]

**Advantages**:
- Efficient for sparse graphs
- GPU-friendly
- Easy to implement message passing

### Data Object

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,           # [num_nodes, num_features]
    edge_index=edge_index,     # [2, num_edges]
    edge_attr=edge_features,   # [num_edges, edge_features]
    y=labels                   # [num_nodes] or [num_graphs]
)
```

---

## Sparse Matrix Representations

### Why Sparse Matrices?

Real-world graphs are typically sparse:
- Social networks: avg degree << num_users
- Citation networks: avg citations << num_papers
- Molecular graphs: atoms have bounded valency

### COO (Coordinate) Format

Stores three arrays:
- **row**: Row indices
- **col**: Column indices
- **data**: Values

```python
# Sparse matrix in COO format
row = [0, 0, 1, 2, 2, 2]
col = [1, 2, 0, 0, 1, 3]
data = [1, 1, 1, 1, 1, 1]
```

**Space**: O(nnz) where nnz = number of non-zero elements

### CSR (Compressed Sparse Row) Format

Optimized for row access:
- **data**: Non-zero values
- **indices**: Column indices
- **indptr**: Row pointer array

```python
# CSR format
data = [1, 1, 1, 1, 1, 1]
indices = [1, 2, 0, 0, 1, 3]
indptr = [0, 2, 3, 6, 6]  # indptr[i]:indptr[i+1] gives row i
```

**Advantages**: Efficient row slicing, matrix-vector multiplication

### CSC (Compressed Sparse Column) Format

Similar to CSR but for column access.

---

## Normalization Techniques

### Why Normalize?

In GNNs, aggregating neighbor features can lead to:
- **Scale issues**: High-degree nodes dominate
- **Training instability**: Gradients explode/vanish

### Symmetric Normalization

Used in Graph Convolutional Networks (GCNs):

$$\tilde{A} = D^{-1/2} A D^{-1/2}$$

where:
- $A$: Adjacency matrix
- $D$: Degree matrix (diagonal with $D[i,i] = \deg(i)$)

**Intuition**: Normalize by degree of both source and target nodes

### Row Normalization

$$\tilde{A} = D^{-1} A$$

where $D^{-1}[i,i] = \frac{1}{\deg(i)}$

**Intuition**: Average over neighbors (each neighbor contributes equally)

### Adding Self-Loops

$$\tilde{A} = A + I$$

where $I$ is the identity matrix

**Intuition**: Include node's own features in aggregation

---

## Batching Graphs

For mini-batch training with multiple graphs:

### Diagonal Stacking

Create a block diagonal adjacency matrix:

$$A_{\text{batch}} = \begin{bmatrix}
A_1 & 0 & 0 \\
0 & A_2 & 0 \\
0 & 0 & A_3
\end{bmatrix}$$

Node features: $X_{\text{batch}} = [X_1; X_2; X_3]$ (vertical concatenation)

### Batch Vector

Track which graph each node belongs to:

```python
batch = [0, 0, 0, 1, 1, 2, 2, 2, 2]
#        graph 0  graph 1  graph 2
```

**PyG handles this automatically!**

---

## Memory and Complexity Comparison

| Representation    | Space      | Edge Lookup | Neighbor Lookup | Best For           |
|-------------------|------------|-------------|-----------------|-------------------|
| Adjacency Matrix  | $O(n^2)$      | $O(1)$        | $O(n)$            | Dense graphs      |
| Edge List         | $O(m)$       | $O(m)$        | $O(m)$            | Simple storage    |
| Adjacency List    | $O(n + m)$   | $O(\deg(v))$   | $O(\deg(v))$       | Most algorithms   |
| Incidence Matrix  | $O(n \times m)$   | $O(m)$        | $O(m)$            | Theoretical use   |
| COO Sparse        | $O(m)$       | $O(m)$        | $O(m)$            | GNNs, GPU         |

where:
- $n$ = number of nodes
- $m$ = number of edges
- $\deg(v)$ = degree of node $v$

---

## Practical Considerations

### When to Use What?

**Adjacency Matrix**:
- Dense graphs ($|E| \approx n^2$)
- Spectral methods
- Small graphs
- Need fast edge lookup

**Edge List**:
- Simple storage
- File I/O
- Initial graph construction

**Adjacency List**:
- Traditional graph algorithms (BFS, DFS, Dijkstra)
- Sparse graphs
- Need efficient neighbor iteration

**COO Format (PyTorch Geometric)**:
- GNNs
- GPU computation
- Sparse graphs
- Mini-batch training

---

## Graph Data Structures in Code

### NetworkX

```python
import networkx as nx

# Internally uses adjacency dict (like adjacency list)
G = nx.Graph()
G.add_edge(0, 1)
```

### NumPy

```python
import numpy as np

# Adjacency matrix
A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]])
```

### SciPy Sparse

```python
from scipy.sparse import csr_matrix

# Sparse adjacency matrix
row = [0, 0, 1, 2, 2, 2]
col = [1, 2, 0, 0, 1, 3]
data = [1, 1, 1, 1, 1, 1]
A_sparse = csr_matrix((data, (row, col)), shape=(4, 4))
```

### PyTorch Geometric

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 0, 1, 2, 2, 2],
                           [1, 2, 0, 0, 1, 3]], dtype=torch.long)
x = torch.randn(4, 16)  # 4 nodes, 16 features each
data = Data(x=x, edge_index=edge_index)
```

---

## Summary

In this lesson, we covered:

1. **Core representations**: Adjacency matrix, edge list, adjacency list, incidence matrix
2. **Feature matrices**: Node and edge features for ML
3. **PyTorch Geometric format**: COO format and Data objects
4. **Sparse representations**: Memory-efficient storage
5. **Normalization**: Preparing graphs for GNNs
6. **Batching**: Handling multiple graphs
7. **Complexity analysis**: When to use each representation

---

## Key Takeaways

- **Most real-world graphs are sparse** → Use sparse representations
- **GNNs use COO format** (edge_index in PyG)
- **Feature matrices** (X) are as important as structure (A)
- **Normalization** prevents scale issues in GNNs
- **Choice of representation** affects both memory and speed

---

## Further Reading

1. **Graph Representation Learning** by William L. Hamilton (Chapter 2)
2. **PyTorch Geometric Documentation**: Data handling
3. **SciPy Sparse Matrix Tutorial**

---

## Exercises

1. Convert between adjacency matrix, edge list, and adjacency list
2. Implement symmetric normalization from scratch
3. Calculate memory usage for different representations
4. Create a PyG Data object from a NetworkX graph
5. Implement batching for multiple graphs manually

---

**Next Lesson**: We'll learn about message passing, the fundamental operation in Graph Neural Networks!
