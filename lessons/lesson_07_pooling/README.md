# Lesson 7: Graph Pooling & Hierarchical GNNs

## Quick Start Guide

Welcome! This lesson covers graph pooling operations and hierarchical graph neural networks for graph-level prediction tasks.

### Files in This Directory

- **lesson_07_textbook.md** - Comprehensive textbook covering all theory and concepts
- **lesson_07_notebook.ipynb** - Interactive Jupyter notebook with implementations and experiments
- **README.md** - This file

### What You'll Learn

This lesson covers:
1. Graph-level tasks and classification
2. Global pooling operations (sum, mean, max)
3. Hierarchical pooling for multi-scale learning
4. **DiffPool** - Learnable differentiable pooling
5. **Top-K Pooling** - Importance-based node selection
6. **SAGPool** - Self-attention graph pooling
7. Applications to molecular property prediction
8. Comparing and visualizing pooling methods

### Quick Navigation

#### Textbook Sections
- **Section 1**: Graph-level tasks overview
- **Section 2**: Pooling operations fundamentals
- **Section 3**: Hierarchical pooling motivation
- **Section 4**: DiffPool in depth
- **Section 5**: Top-K Pooling
- **Section 6**: SAGPool with attention
- **Section 7**: Hierarchical architectures with code
- **Section 8**: Real-world applications
- **Section 9**: Summary and comparisons

#### Notebook Parts
- **Part 1**: Global pooling implementation
- **Part 2**: Visualizing pooling effects
- **Part 3**: Top-K pooling with visualization
- **Part 4**: SAGPool implementation
- **Part 5**: Graph classification model
- **Part 6**: Synthetic dataset creation
- **Part 7**: Training and evaluation
- **Part 8**: Molecular property prediction
- **Part 9**: Hierarchical visualization
- **Part 10**: Method comparison
- **Exercises**: 8 practical exercises

### How to Use These Materials

#### For Beginners
1. Read textbook sections 1-2 (30 minutes)
2. Run notebook parts 1-2 (30 minutes)
3. Study section 3 (20 minutes)
4. Try exercises 1-2 (30 minutes)

#### For Intermediate Learners
1. Read full textbook (1 hour)
2. Run notebook parts 3-7 (1.5 hours)
3. Work on exercises 3-5 (1 hour)
4. Experiment with variations (1 hour)

#### For Advanced Learners
1. Deep dive into sections 4-6 (1.5 hours)
2. Run parts 8-10 (1.5 hours)
3. Complete exercises 6-8 (1.5 hours)
4. Apply to real datasets (2+ hours)

### Key Concepts

#### Global Pooling
- **Sum Pooling**: g = Σhᵢ
- **Mean Pooling**: g = (1/n)Σhᵢ
- **Max Pooling**: gⱼ = maxᵢ(hᵢⱼ)

#### Hierarchical Methods
- **DiffPool**: Learnable soft assignment matrix
- **Top-K**: Hard selection by importance score
- **SAGPool**: Attention-based importance scoring

### Main Classes

```python
# Global pooling
GlobalPooling(method='concat')  # sum, mean, max, or concat

# Hierarchical pooling
TopKPooling(in_channels=32, ratio=0.8)
SAGPooling(in_channels=32, ratio=0.8)

# Full model
HierarchicalGNN(in_channels=8, hidden_channels=32, 
                num_classes=2, num_layers=3)
```

### Running the Notebook

```bash
# Install dependencies (if needed)
pip install torch torch-geometric networkx matplotlib numpy pandas scikit-learn

# Open Jupyter
jupyter notebook lesson_07_notebook.ipynb
```

### Expected Output

- Training curves showing accuracy and loss
- Visualizations of node selection at different pooling ratios
- Hierarchical graph structure visualizations
- Comparison of pooling methods on test datasets
- Analysis of computational complexity

### Key Datasets

1. **Synthetic Graphs**: 100 graphs, binary classification
   - Class 0: Random graphs
   - Class 1: Community-structured graphs

2. **Synthetic Molecules**: 100 molecules, binary classification
   - Active vs Inactive compounds
   - Variable atom counts (5-20 atoms)

### Learning Outcomes

After completing this lesson, you will:
- Understand graph-level prediction tasks
- Master multiple pooling techniques
- Build hierarchical GNN architectures
- Apply GNNs to molecular data
- Compare pooling methods empirically
- Visualize and interpret graph hierarchies

### Common Questions

**Q: Which pooling method should I use?**
- Start with concatenated global pooling
- Use Top-K or SAGPool for efficiency
- Use DiffPool if interpretability is important

**Q: How many pooling layers do I need?**
- Usually 2-3 layers work well
- More layers for complex graphs
- See Part 10 for empirical comparison

**Q: Can I use this for real molecules?**
- Yes! Use RDKit to convert SMILES to graphs
- See Textbook Section 8 and Notebook Exercise 7

**Q: What's the computational cost?**
- Global: O(n)
- Top-K/SAGPool: O(n log n)
- DiffPool: O(n²)

### Further Learning

See the Academic References section in the textbook for papers:
- DiffPool (Lee et al., ICLR 2019)
- Top-K/Graph U-Nets (Lee et al., ICML 2019)
- SAGPool (Lee et al., ICLR 2019)

### Tips for Success

1. **Start simple**: Begin with global pooling before hierarchical methods
2. **Visualize**: Use the visualization functions to understand pooling
3. **Experiment**: Modify hyperparameters and observe effects
4. **Compare**: Run Part 10 to see relative performance
5. **Apply**: Try the exercises and real data applications

### Need Help?

- Check the textbook's "Applications" section for your use case
- Review the notebook's visualization cells for intuition
- Look at the exercises for guided practice
- Study the code comments and docstrings

---

**Happy Learning!** Start with the textbook or notebook and progress at your own pace.
