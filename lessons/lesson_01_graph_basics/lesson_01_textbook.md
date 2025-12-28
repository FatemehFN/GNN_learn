# Lesson 1: Graph Basics

## Introduction to Graphs

### What is a Graph?

A **graph** is a mathematical structure used to model pairwise relations between objects. Formally, a graph G is defined as:

**G = (V, E)**

Where:
- **V** is a set of vertices (also called nodes)
- **E** is a set of edges (also called links or connections)

### Why Graphs?

Graphs are everywhere in the real world:
- **Social Networks**: People (nodes) connected by friendships (edges)
- **Molecules**: Atoms (nodes) connected by chemical bonds (edges)
- **Transportation**: Cities (nodes) connected by roads (edges)
- **Internet**: Web pages (nodes) connected by hyperlinks (edges)
- **Knowledge Graphs**: Entities (nodes) connected by relationships (edges)

---

## Types of Graphs

### 1. Undirected Graphs

In an undirected graph, edges have no direction. If there's an edge between nodes A and B, you can traverse from A to B and from B to A.

**Example**: Facebook friendships (if A is friends with B, then B is friends with A)

**Mathematical Notation**:
- Edge between u and v: {u, v} or (u, v)
- If (u, v) ∈ E, then (v, u) ∈ E

### 2. Directed Graphs (Digraphs)

In a directed graph, edges have direction. An edge from A to B doesn't imply an edge from B to A.

**Example**: Twitter follows (A can follow B without B following A)

**Mathematical Notation**:
- Edge from u to v: (u, v) or u → v
- (u, v) ∈ E does NOT imply (v, u) ∈ E

### 3. Weighted Graphs

Edges have weights (numerical values) associated with them.

**Example**: Road networks where weights represent distances

**Mathematical Notation**:
- Weight function: w: E → ℝ
- w(u, v) represents the weight of edge (u, v)

### 4. Multigraphs

Graphs that allow multiple edges between the same pair of nodes.

**Example**: Multiple flights between two cities

### 5. Self-loops

Edges that connect a node to itself.

**Example**: A web page linking to itself

---

## Graph Properties and Terminology

### Basic Definitions

**Degree**: The number of edges connected to a node
- In undirected graphs: deg(v) = number of edges incident to v
- In directed graphs:
  - **In-degree** (deg⁻(v)): number of incoming edges
  - **Out-degree** (deg⁺(v)): number of outgoing edges

**Path**: A sequence of nodes where each consecutive pair is connected by an edge
- Example: v₁ → v₂ → v₃ → v₄

**Path Length**: Number of edges in a path

**Cycle**: A path that starts and ends at the same node

**Connected Graph**: A graph where there's a path between any pair of nodes

**Connected Component**: A maximal connected subgraph

### Advanced Properties

**Diameter**: The longest shortest path between any two nodes
- diameter(G) = max(d(u, v)) for all u, v ∈ V
- where d(u, v) is the shortest path distance

**Clustering Coefficient**: Measures how much nodes tend to cluster together
- For node v: C(v) = (number of edges between neighbors of v) / (possible edges between neighbors)

**Density**: Ratio of actual edges to possible edges
- For undirected graph: density = 2|E| / (|V|(|V|-1))
- For directed graph: density = |E| / (|V|(|V|-1))

---

## Special Types of Graphs

### 1. Complete Graph (Kₙ)

A graph where every pair of distinct nodes is connected by an edge.
- Number of edges: |E| = n(n-1)/2 for undirected graphs

### 2. Tree

A connected acyclic graph (no cycles).

**Properties**:
- For a tree with n nodes: |E| = n - 1
- There's exactly one path between any two nodes
- Removing any edge disconnects the graph

### 3. Bipartite Graph

A graph whose nodes can be divided into two disjoint sets such that every edge connects nodes from different sets.

**Example**: Students and courses (edges represent enrollment)

**Mathematical Definition**:
- V = U ∪ W where U ∩ W = ∅
- Every edge (u, v) ∈ E has u ∈ U and v ∈ W

### 4. Planar Graph

A graph that can be drawn on a plane without edge crossings.

**Example**: Maps of countries sharing borders

---

## Graph Mathematics

### Adjacency and Incidence

**Adjacent Nodes**: Two nodes connected by an edge
- u and v are adjacent if (u, v) ∈ E

**Incident Edge**: An edge is incident to a node if the node is one of the edge's endpoints

**Neighborhood**: The set of nodes adjacent to a given node
- N(v) = {u ∈ V : (v, u) ∈ E}

### Subgraphs

A graph G' = (V', E') is a **subgraph** of G = (V, E) if:
- V' ⊆ V
- E' ⊆ E

**Induced Subgraph**: Given a subset V' ⊆ V, the induced subgraph contains all edges from E that connect nodes in V'

### Isomorphism

Two graphs G₁ = (V₁, E₁) and G₂ = (V₂, E₂) are **isomorphic** if there exists a bijection f: V₁ → V₂ such that:
- (u, v) ∈ E₁ if and only if (f(u), f(v)) ∈ E₂

Isomorphic graphs have the same structure, just different labels.

---

## Graph Algorithms (Preview)

### Traversal Algorithms

**Breadth-First Search (BFS)**:
- Explores nodes level by level
- Uses a queue
- Time complexity: O(|V| + |E|)

**Depth-First Search (DFS)**:
- Explores as far as possible along each branch
- Uses a stack (or recursion)
- Time complexity: O(|V| + |E|)

### Shortest Path

**Dijkstra's Algorithm**:
- Finds shortest paths from a source node to all other nodes
- Works for non-negative edge weights
- Time complexity: O((|V| + |E|) log |V|) with priority queue

**Bellman-Ford Algorithm**:
- Handles negative edge weights
- Time complexity: O(|V| × |E|)

---

## Mathematical Formulations

### Handshaking Lemma

For any undirected graph:

**∑(v∈V) deg(v) = 2|E|**

This means the sum of all degrees equals twice the number of edges (since each edge contributes to two nodes' degrees).

### Erdős–Rényi Random Graph

A random graph model where each possible edge exists with probability p.

**Expected number of edges**: E[|E|] = p × n(n-1)/2

**Expected degree**: E[deg(v)] = p(n-1)

---

## Summary

In this lesson, we covered:

1. **Definition of graphs**: G = (V, E)
2. **Types of graphs**: Undirected, directed, weighted, multigraphs
3. **Graph properties**: Degree, path, cycle, connectivity
4. **Special graphs**: Complete, trees, bipartite, planar
5. **Mathematical concepts**: Adjacency, subgraphs, isomorphism
6. **Basic algorithms**: BFS, DFS, shortest paths

---

## Key Takeaways

- Graphs are powerful structures for representing relationships
- Understanding basic graph properties is essential for GNNs
- Different graph types require different representations and algorithms
- Graph structure significantly impacts algorithm complexity

---

## Further Reading

1. **Introduction to Graph Theory** by Douglas B. West
2. **Networks** by Mark Newman
3. **Graph Theory and Complex Networks** by Maarten van Steen

---

## Exercises

1. Prove the handshaking lemma
2. Calculate the density of a complete graph Kₙ
3. Show that a tree with n nodes has exactly n-1 edges
4. Determine if two given graphs are isomorphic
5. Implement BFS and DFS from scratch

---

**Next Lesson**: We'll learn how to represent graphs computationally using adjacency matrices, edge lists, and other data structures essential for implementing GNNs.
