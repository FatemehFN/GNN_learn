# Quick Start Guide - Lesson 3

## What You'll Learn in 30 Minutes

This notebook teaches **Message Passing** - the core mechanism of Graph Neural Networks.

### Key Concept
```
Node sends its features to neighbors → 
Neighbors send messages back → 
Node aggregates all messages → 
Node updates its representation based on neighborhood information
```

## File to Open
```
/Users/fsfatemi/GNN_learn/lessons/lesson_03_message_passing/lesson_03_notebook.ipynb
```

## 5-Minute Overview

1. **Import Libraries** (Cell 3)
   - PyTorch, NetworkX, Matplotlib
   
2. **Understand the Concept** (Cell 4)
   - Read the mathematical formulation
   - Understand the 4-step process

3. **Run SimpleMessagePassing** (Cell 6)
   - See the implementation from scratch
   - Understand message_fn, aggregate_fn, update_fn

4. **Visualize Evolution** (Cells 8-9)
   - Watch features change through layers
   - See how node colors (features) evolve

5. **Compare Aggregations** (Cells 11-12)
   - See mean vs sum vs max aggregation
   - Understand trade-offs

## Key Visualizations

### 1. Message Passing Evolution
Shows how node features (colors) change through 3 layers of message passing.

### 2. Aggregation Comparison
3 different strategies side-by-side:
- **Mean**: Smooth, stable aggregation
- **Sum**: Accumulating information
- **Max**: Keeping strongest signals

### 3. Receptive Field Growth
Demonstrates how a node's neighborhood expands with each layer.

### 4. Over-smoothing Problem
Shows why deep GNNs become problematic - node features converge.

## Running in Order

```
1. Cell 3  - Imports
2. Cell 4  - Concept explanation  
3. Cell 6  - SimpleMessagePassing class
4. Cell 8  - Create graph
5. Cell 9  - Visualize
6. Cell 11 - Compare aggregations
7. Cell 13 - Receptive field
8. Cell 15 - Over-smoothing
9. Cell 17 - GNNModel
10. Cell 18 - Dataset creation
11. Cell 19 - Training function
12. Cell 20 - Results visualization
```

## Interactive Experiments

**Try these modifications:**

1. **Change aggregation type:**
   ```python
   mp = SimpleMessagePassing(..., aggregation='max')  # Try 'mean', 'sum', 'max'
   ```

2. **More layers:**
   ```python
   _, history = mp.forward(num_layers=5)  # Try different depths
   ```

3. **Different graph:**
   ```python
   G = nx.barabasi_albert_graph(20, 3)  # Try other graph types
   ```

4. **Adjust training:**
   ```python
   train_losses, val_accs, test_acc = train_gnn(
       model, X, y, adj, train_mask, val_mask, test_mask,
       epochs=200,  # More training
       lr=0.001     # Different learning rate
   )
   ```

## 3 Core Classes to Understand

### 1. SimpleMessagePassing
```python
mp = SimpleMessagePassing(num_nodes=9, feature_dim=3, aggregation='mean')
mp.add_edge(0, 1)  # Add edges
features, history = mp.forward(num_layers=3, return_history=True)
```

### 2. GNNModel
```python
model = GNNModel(in_features=16, hidden_features=32, out_features=3, num_layers=2)
logits = model(X, adj)  # Forward pass
```

### 3. Training Loop
```python
train_losses, val_accs, test_acc = train_gnn(
    model, X, y, adj, train_mask, val_mask, test_mask,
    epochs=100, lr=0.01
)
```

## Output to Expect

After running the full notebook:

1. **Visualization of feature evolution** - 4 graphs showing color changes
2. **Aggregation comparison grid** - 12 graphs (3 methods × 4 layers)
3. **Receptive field panels** - 6 visualizations of neighborhood growth
4. **Over-smoothing plots** - 2 graphs showing the problem
5. **Training progress** - 2 curves (loss and accuracy)

## Important Concepts

| Term | Meaning |
|------|---------|
| **Message** | Feature vector sent from one node to another |
| **Aggregation** | Combining messages from multiple neighbors |
| **Receptive Field** | All nodes that can influence a given node |
| **Over-smoothing** | Problem where node features become too similar |
| **Layer** | One round of message passing across the graph |

## Common Questions

**Q: Why message passing?**
A: It's how neural networks learn from graph structure - information flows along edges.

**Q: Why different aggregations?**
A: Different strategies (mean, sum, max) capture different aspects of neighborhood.

**Q: What's over-smoothing?**
A: After many layers, all nodes become similar - defeating the purpose.

**Q: How many layers should I use?**
A: Usually 2-4 for most problems. More layers → over-smoothing risk.

## Challenge Exercises

After understanding the basics:

1. **Implement Attention** - Weight messages by importance
2. **Add Skip Connections** - Preserve original features
3. **Test Different Depths** - How many layers is optimal?
4. **Create Custom Dataset** - Use your own data
5. **Compare Empirically** - Which aggregation is best?

## Resources

**Inside the Notebook:**
- Cell 4: Mathematical formulation
- Cell 6: Complete implementation example
- Cell 21-22: Theory and exercises
- Cell 23: Reference papers

**External:**
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- DGL Documentation: https://docs.dgl.ai/

## Next Lesson

After mastering message passing:
- **Lesson 4**: Graph Convolutional Networks (GCN)
  - Specific type of message passing
  - Normalization tricks
  - Spectral interpretation

---

**Estimated Time:** 30 min to understand, 2-3 hours to fully explore

**Status:** Ready to run immediately!

**Tips:**
- Run cells in sequence (they depend on each other)
- Experiment with parameters
- Try the exercises
- Don't skip the visualizations - they're key to understanding
