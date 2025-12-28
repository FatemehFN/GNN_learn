# Setup Guide - GNN Learning Repository

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to the repository
cd GNN_learn

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; import torch_geometric; import networkx; print('All packages installed successfully!')"
```

### 3. Start Learning

```bash
# Launch Jupyter
jupyter notebook

# Navigate to lessons/lesson_01_graph_basics/lesson_01_notebook.ipynb
```

## Installation Troubleshooting

### PyTorch Geometric Installation Issues

If you encounter issues installing PyTorch Geometric, use the official installation command:

```bash
# For CUDA 11.8 (check your CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric

# For CPU-only
pip install torch torchvision torchaudio
pip install torch-geometric
```

### Platform-Specific Notes

**macOS (M1/M2)**:
```bash
# Use conda for better compatibility
conda install pytorch torchvision torchaudio -c pytorch
conda install pyg -c pyg
```

**Windows**:
- Ensure you have Visual Studio Build Tools installed
- Consider using Anaconda for easier package management

**Linux**:
```bash
# Usually works out of the box
pip install -r requirements.txt
```

## Learning Path

### Recommended Order (8-10 weeks, 5-10 hours per week)

**Week 1: Foundations**
- Lesson 1: Graph Basics
- Read textbook, complete notebook exercises

**Week 2: Representations**
- Lesson 2: Graph Representations
- Practice converting between formats

**Week 3: Core Concepts**
- Lesson 3: Message Passing & GNN Foundations
- Implement message passing from scratch

**Week 4: First GNN**
- Lesson 4: Graph Convolutional Networks
- Train GCN on real dataset

**Week 5: Advanced Architectures**
- Lesson 5: GraphSAGE & GIN
- Compare different architectures

**Week 6: Attention**
- Lesson 6: Graph Attention Networks
- Visualize learned attention weights

**Week 7: Graph-Level Learning**
- Lesson 7: Graph Pooling
- Build graph classifier

**Week 8: Real-World Applications**
- Lesson 8: Advanced Topics
- Complete end-to-end project

### Intensive Fast Track (2-3 weeks full-time)

**Day 1-2**: Lessons 1-2 (basics and representations)
**Day 3-4**: Lessons 3-4 (message passing and GCNs)
**Day 5-7**: Lessons 5-6 (advanced architectures and attention)
**Day 8-10**: Lessons 7-8 (pooling and applications)
**Day 11-14**: Final project and deep dives

### Casual Learning (3-4 months, weekends)

**Month 1**: Lessons 1-3 (foundations)
**Month 2**: Lessons 4-5 (core GNNs)
**Month 3**: Lessons 6-7 (advanced topics)
**Month 4**: Lesson 8 + projects

## Study Tips

### For Each Lesson:

1. **Read the textbook first** (30-60 min)
   - Understand mathematical concepts
   - Note key formulas

2. **Work through the notebook** (1-3 hours)
   - Run every cell
   - Experiment with parameters
   - Add print statements to understand internals

3. **Complete exercises** (1-2 hours)
   - Try without looking at solutions
   - Look up documentation when stuck

4. **Build something extra** (optional, 2-4 hours)
   - Apply to your own dataset
   - Extend the implementation
   - Try variations

### Effective Learning Strategies

**Active Learning**:
- Don't just read code - type it out
- Break things intentionally to understand errors
- Modify parameters and observe effects

**Visualization**:
- Use visualization tools extensively
- Draw graph structures by hand
- Sketch message passing steps

**Documentation**:
- Keep a learning journal
- Write summaries in your own words
- Create cheat sheets for formulas

**Community**:
- Join GNN communities (Reddit, Discord)
- Discuss concepts with peers
- Contribute to open-source projects

## Prerequisites Check

Before starting, ensure you understand:

### Python Programming
- [ ] Functions, classes, loops
- [ ] NumPy basics (arrays, indexing)
- [ ] Basic plotting with Matplotlib

### Mathematics
- [ ] Linear algebra (vectors, matrices, dot products)
- [ ] Basic calculus (derivatives, chain rule)
- [ ] Basic probability

### Machine Learning
- [ ] Neural networks basics
- [ ] Backpropagation concept
- [ ] Loss functions and optimization

**Don't worry if you're rusty** - you can review as you go!

## Resources for Prerequisites

**Python**:
- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- NumPy tutorials

**Linear Algebra**:
- 3Blue1Brown's Essence of Linear Algebra (YouTube)
- Khan Academy Linear Algebra

**Machine Learning**:
- Andrew Ng's ML course (Coursera)
- Fast.ai Practical Deep Learning

**Graph Theory**:
- Introduction to Graph Theory by Douglas West
- NetworkX tutorials

## Getting Help

### Common Issues

**Import Errors**:
```python
# Make sure packages are installed in the right environment
pip list | grep torch
pip list | grep networkx
```

**CUDA/GPU Issues**:
```python
# Check if CUDA is available
import torch
print(torch.cuda.is_available())
```

**Memory Errors**:
- Reduce batch size
- Use smaller graphs for testing
- Close other applications

### Where to Ask Questions

1. **Repository Issues**: For course-specific questions
2. **PyTorch Geometric Forum**: For library-specific issues
3. **Stack Overflow**: For general programming questions
4. **Reddit r/MachineLearning**: For theoretical discussions

## Development Environment

### Recommended Setup

**IDE**: VS Code with extensions:
- Python
- Jupyter
- GitLens

**Alternative**: JupyterLab
```bash
pip install jupyterlab
jupyter lab
```

### Useful Jupyter Extensions

```bash
# Table of contents
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Variable inspector
pip install jupyter_nbextensions_configurator
```

## Project Structure

```
GNN_learn/
├── README.md              # Main overview
├── SETUP.md              # This file
├── requirements.txt      # Python dependencies
├── lessons/
│   ├── lesson_01_graph_basics/
│   │   ├── lesson_01_textbook.md
│   │   └── lesson_01_notebook.ipynb
│   ├── lesson_02_graph_representations/
│   ├── lesson_03_message_passing/
│   ├── lesson_04_gcn/
│   ├── lesson_05_advanced_architectures/
│   ├── lesson_06_gat/
│   ├── lesson_07_pooling/
│   └── lesson_08_advanced_topics/
└── [your projects here]/
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Verify installation
3. ✅ Read Lesson 1 textbook
4. ✅ Complete Lesson 1 notebook
5. ✅ Move to Lesson 2

**Happy Learning!** 🚀

---

*For questions or issues, please refer to the main README.md or create an issue.*
