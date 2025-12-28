# Graph Neural Networks: From Basics to Advanced

A comprehensive, hands-on learning path for Graph Neural Networks (GNNs), designed to take you from fundamental graph concepts to state-of-the-art architectures.

## Course Structure

Each lesson includes:
- **Textbook**: Detailed explanations of concepts, mathematical foundations, and theory
- **Jupyter Notebook**: Hands-on implementation and practical exercises

## Curriculum

### Lesson 1: Graph Basics
**Topics**: Graph theory fundamentals, nodes, edges, types of graphs, basic properties
- [Textbook](lessons/lesson_01_graph_basics/lesson_01_textbook.md)
- [Hands-on Notebook](lessons/lesson_01_graph_basics/lesson_01_notebook.ipynb)

### Lesson 2: Graph Representations
**Topics**: Adjacency matrices, edge lists, adjacency lists, feature matrices, graph data structures
- [Textbook](lessons/lesson_02_graph_representations/lesson_02_textbook.md)
- [Hands-on Notebook](lessons/lesson_02_graph_representations/lesson_02_notebook.ipynb)

### Lesson 3: Message Passing & GNN Foundations
**Topics**: Message passing framework, aggregation functions, neighborhood aggregation, permutation invariance
- [Textbook](lessons/lesson_03_message_passing/lesson_03_textbook.md)
- [Hands-on Notebook](lessons/lesson_03_message_passing/lesson_03_notebook.ipynb)

### Lesson 4: Graph Convolutional Networks (GCNs)
**Topics**: Spectral graph theory basics, graph convolutions, GCN architecture, semi-supervised learning
- [Textbook](lessons/lesson_04_gcn/lesson_04_textbook.md)
- [Hands-on Notebook](lessons/lesson_04_gcn/lesson_04_notebook.ipynb)

### Lesson 5: Advanced GNN Architectures
**Topics**: GraphSAGE, GIN (Graph Isomorphism Networks), expressive power of GNNs
- [Textbook](lessons/lesson_05_advanced_architectures/lesson_05_textbook.md)
- [Hands-on Notebook](lessons/lesson_05_advanced_architectures/lesson_05_notebook.ipynb)

### Lesson 6: Graph Attention Networks (GATs)
**Topics**: Attention mechanisms, multi-head attention for graphs, GAT architecture
- [Textbook](lessons/lesson_06_gat/lesson_06_textbook.md)
- [Hands-on Notebook](lessons/lesson_06_gat/lesson_06_notebook.ipynb)

### Lesson 7: Graph Pooling & Hierarchical GNNs
**Topics**: Graph-level tasks, pooling operations, DiffPool, hierarchical representations
- [Textbook](lessons/lesson_07_pooling/lesson_07_textbook.md)
- [Hands-on Notebook](lessons/lesson_07_pooling/lesson_07_notebook.ipynb)

### Lesson 8: Advanced Topics & Applications
**Topics**: Heterogeneous graphs, temporal GNNs, graph generation, real-world applications
- [Textbook](lessons/lesson_08_advanced_topics/lesson_08_textbook.md)
- [Hands-on Notebook](lessons/lesson_08_advanced_topics/lesson_08_notebook.ipynb)

## Prerequisites

- Basic Python programming
- Linear algebra (vectors, matrices)
- Basic calculus (derivatives, gradients)
- Basic machine learning concepts (neural networks, backpropagation)

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- Matplotlib
- Jupyter

## How to Use This Repository

1. Start with Lesson 1 and progress sequentially
2. Read the textbook first to understand the theory
3. Work through the Jupyter notebook for hands-on practice
4. Complete all exercises before moving to the next lesson
5. Experiment with the code and try variations

## Learning Path

```
Graph Basics → Graph Representations → Message Passing
    ↓
Graph Convolutions (GCN) → Advanced Architectures (GraphSAGE, GIN)
    ↓
Attention Mechanisms (GAT) → Graph Pooling
    ↓
Advanced Topics & Applications
```

## Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [NetworkX Documentation](https://networkx.org/)
- Research papers and references included in each lesson

## Contributing

Feel free to open issues or submit pull requests to improve the content.

## License

MIT License - Feel free to use this for educational purposes.

---

**Happy Learning!** Start with [Lesson 1](lessons/lesson_01_graph_basics/) to begin your GNN journey.
