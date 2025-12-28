# Quick Start Guide - Get Learning in 5 Minutes!

## 🚀 Fastest Way to Start

```bash
# 1. Clone or navigate to the repository
cd GNN_learn

# 2. Install dependencies
pip install torch torch-geometric networkx numpy matplotlib jupyter

# 3. Start Jupyter
jupyter notebook

# 4. Open the first lesson
# Navigate to: lessons/lesson_01_graph_basics/lesson_01_notebook.ipynb
```

**That's it!** You're ready to learn GNNs.

---

## 📖 First 30 Minutes

### Step 1: Read the Textbook (10 min)
Open `lessons/lesson_01_graph_basics/lesson_01_textbook.md`

Skim through:
- What is a Graph?
- Types of Graphs
- Basic Properties

### Step 2: Run the Notebook (15 min)
Open `lessons/lesson_01_graph_basics/lesson_01_notebook.ipynb`

Run all cells (Cell → Run All)
- See graphs visualized
- Watch code execute
- Understand outputs

### Step 3: Try One Exercise (5 min)
Pick Exercise 1 from the notebook:
```python
# Create a random graph
G_random = nx.erdos_renyi_graph(10, 0.3)
nx.draw(G_random, with_labels=True)
```

---

## 📚 What You'll Learn

### Lesson 1 (Today, 2-3 hours)
- Graph basics and types
- Creating graphs with NetworkX
- Graph algorithms (BFS, DFS)

### Lesson 2 (Tomorrow, 2-3 hours)
- Graph representations (matrices, lists)
- PyTorch Geometric format
- Sparse matrices

### Lesson 3 (Day 3, 3-4 hours)
- Message passing framework
- Your first GNN implementation
- Aggregation functions

### Lesson 4 (Day 4-5, 4-5 hours)
- Graph Convolutional Networks
- Training on real dataset (Cora)
- Visualizing embeddings

### Continue through Lessons 5-8
- Advanced architectures
- Attention mechanisms
- Graph pooling
- Real-world applications

---

## 🎯 Your First Week Plan

**Day 1 (2-3h)**: Lesson 1 - Graph Basics
- Understand graphs fundamentally
- Create your first graphs
- Run graph algorithms

**Day 2 (2-3h)**: Lesson 2 - Representations
- Learn data structures
- Convert between formats
- Use PyTorch Geometric

**Day 3 (3-4h)**: Lesson 3 - Message Passing
- Core GNN concept
- Implement from scratch
- Understand aggregation

**Day 4-5 (4-5h)**: Lesson 4 - GCN
- Your first real GNN
- Train on citation network
- See it work!

**Day 6-7 (4-5h)**: Review & Practice
- Redo exercises
- Build mini-project
- Solidify concepts

**Total**: 15-20 hours in Week 1
**Achievement**: Understand and implement GNNs!

---

## 💻 System Requirements

**Minimum**:
- Python 3.8+
- 4GB RAM
- 2GB disk space
- CPU (no GPU needed for learning)

**Recommended**:
- Python 3.9+
- 8GB RAM
- 5GB disk space
- GPU (optional, for faster training in later lessons)

---

## 🔍 What If Something Breaks?

### Can't install packages?
```bash
# Try with conda instead
conda create -n gnn python=3.9
conda activate gnn
conda install pytorch torchvision torchaudio -c pytorch
conda install pyg -c pyg
pip install networkx matplotlib jupyter
```

### Import errors?
```python
# Check what's installed
import sys
print(sys.executable)  # Should be in your venv

# Install in correct environment
!pip install package-name  # In Jupyter
```

### Notebook won't run?
- Restart kernel: Kernel → Restart
- Clear outputs: Cell → All Output → Clear
- Reinstall Jupyter: `pip install --upgrade jupyter`

---

## 📖 Reading vs Doing

### Don't Just Read - DO!

❌ **Bad approach**:
- Read all textbooks first
- Just run notebooks without understanding
- Skip exercises

✅ **Good approach**:
- Read textbook section by section
- Run and modify code as you go
- Complete at least 2-3 exercises per lesson
- Build something small with each lesson

---

## 🎓 Learning Philosophy

### This Course Is:
✅ Hands-on (learn by doing)
✅ Progressive (easy → hard)
✅ Practical (real datasets, working code)
✅ Comprehensive (theory + practice)

### This Course Is NOT:
❌ Just theory (no passive reading)
❌ Just code (no cookbook without understanding)
❌ A shortcut (requires effort and time)

---

## 🏆 Your Goal

After this course, you'll be able to:
- **Understand** how GNNs work deeply
- **Implement** GNNs from scratch
- **Apply** GNNs to real problems
- **Read** research papers confidently
- **Build** production systems

---

## 🚦 Ready, Set, Go!

### Right Now:
1. Open a terminal
2. Navigate to GNN_learn
3. Run: `jupyter notebook`
4. Open Lesson 1 notebook
5. Start learning!

### In 1 Hour:
You'll understand what graphs are and how to work with them.

### In 1 Day:
You'll know how to represent graphs for machine learning.

### In 1 Week:
You'll have built and trained your first GNN!

### In 1 Month:
You'll be a GNN expert ready for advanced applications.

---

**Stop reading. Start coding.** 

Open `lessons/lesson_01_graph_basics/lesson_01_notebook.ipynb` NOW! 🚀

---

*Questions? Check SETUP.md for detailed installation instructions.*
*Confused? Read COURSE_OVERVIEW.md for the big picture.*
*Ready? Go to Lesson 1 and begin!*
