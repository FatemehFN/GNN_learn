# Lesson 4: Graph Convolutional Networks (GCNs)

## Introduction to Graph Convolutional Networks

Graph Convolutional Networks (GCNs) are one of the most influential and widely-used Graph Neural Network architectures. They bridge classical signal processing on graphs with modern deep learning, providing an elegant framework for semi-supervised learning on graphs.

**Why GCNs?**
- Simple yet powerful architecture
- Theoretically grounded in spectral graph theory
- Efficient and scalable to large graphs
- Excellent performance on node classification tasks
- Interpretable: clear layer-wise propagation rule

---

## Spectral Graph Theory Basics

### What is Spectral Graph Theory?

Spectral graph theory studies graphs through the eigenvalues and eigenvectors of matrices associated with graphs. The main idea is to analyze graph properties through the **spectrum** (set of eigenvalues) of graph matrices.

### Graph Adjacency Matrix

The adjacency matrix $A$ is the fundamental representation:

$$A \in \{0,1\}^{n \times n}$$ where:
- $A[i,j] = 1$ if edge exists between nodes i and j
- $A[i,j] = 0$ otherwise
- For undirected graphs: $A$ is symmetric

**Example**: For a simple graph:
```
  A - B - C
```
```
     A  B  C
A [  0  1  0 ]
B [  1  0  1 ]
C [  0  1  0 ]
```

### Degree Matrix

The degree matrix $D$ is diagonal, where diagonal element is node degree:

$$D[i,i] = \deg(i) = \sum_j A[i,j]$$

For our example:
```
     A  B  C
A [  1  0  0 ]
D [  0  2  0 ]
C [  0  0  1 ]
```

---

## The Graph Laplacian

### Unnormalized Laplacian

The **Graph Laplacian** is defined as:

$$L = D - A$$

This is one of the most important matrices in spectral graph theory.

**Properties**:
- **Symmetric**: $L = L^T$
- **Positive semi-definite**: All eigenvalues $\geq 0$
- **Zero eigenvalue exists**: L has smallest eigenvalue $\lambda_0 = 0$
  - Eigenvector: all ones vector $\mathbf{1}$
  - Connected graphs have exactly one zero eigenvalue
- **Row sum is zero**: Each row sums to 0

**Example** (continuing above):
```
L = D - A =
     A   B   C
A [  1  -1   0 ]
B [ -1   2  -1 ]
C [  0  -1   1 ]
```

### Normalized Laplacian

To handle varying node degrees, we normalize:

$$\tilde{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

Or using symmetric normalization:

$$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$$

This normalized version:
- Has eigenvalues in $[0, 2]$
- Better conditioned numerically
- Often more stable for optimization

### Random Walk Normalized Laplacian

An alternative normalization following random walk perspective:

$$L_{\text{rw}} = I - D^{-1} A$$

This has eigenvalues in $[0, 2]$ and is used in some GNN variants.

---

## Eigenvalues and Eigenvectors

### Spectral Decomposition

Any symmetric matrix can be decomposed as:

$$L = U \Lambda U^T$$

Where:
- $U$: Matrix of eigenvectors (columns are eigenvectors)
- $\Lambda$: Diagonal matrix of eigenvalues
- $U^T$: Transpose of U (orthogonal matrix)

### Interpretation

Eigenvalues and eigenvectors tell us about graph structure:

1. **Zero eigenvalue**: Indicates connected components
   - Multiple zeros = disconnected graph
2. **Small eigenvalues**: Smooth eigenvectors
   - Vary slowly across graph
   - Capture global structure
3. **Large eigenvalues**: Rapidly varying eigenvectors
   - Change frequently across edges
   - Capture local details

**Intuition**: Like Fourier analysis on graphs:
- Low frequencies (small eigenvalues) = global patterns
- High frequencies (large eigenvalues) = local details

---

## Spectral Convolutions

### Convolutions in Different Domains

On regular grids (images):
- Convolution = local weighted sum
- Works in spatial domain

On graphs:
- Graphs have irregular structure
- No natural notion of locality
- Solution: Work in spectral domain

### Spectral Convolution Definition

A spectral convolution with filter $g_\theta$ is defined as:

$$y = g_\theta(L) x = U g_\theta(\Lambda) U^T x$$

Where:
- $x$: Input (e.g., node features)
- $g_\theta(\Lambda)$: Filter applied to eigenvalues
- $U$: Eigenvectors of Laplacian
- $y$: Output

This is analogous to convolution in signal processing:
- Fourier transform → spectral decomposition
- Multiply by filter → apply g_θ to eigenvalues
- Inverse Fourier → transform back

### Problem: Computational Cost

Direct spectral convolution requires:
1. Computing eigendecomposition: $O(n^3)$
2. Multiplying by eigenvector matrix: $O(n^2)$

Infeasible for large graphs!

**Solution**: Use **Chebyshev polynomial approximation**

---

## Chebyshev Polynomial Approximation

### Motivation

Instead of computing eigendecomposition, we can approximate the filter using Chebyshev polynomials:

$$g_\theta(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$

Where:
- $T_k$: Chebyshev polynomials
- $\tilde{\Lambda}$: Normalized eigenvalues (rescaled to $[-1, 1]$)
- $\theta_k$: Learnable coefficients
- $K$: Order of polynomial approximation

### Chebyshev Polynomials

Chebyshev polynomials satisfy:
$$T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$$

With base cases:
- $T_0(x) = 1$
- $T_1(x) = x$

First few polynomials:
- $T_0(x) = 1$
- $T_1(x) = x$
- $T_2(x) = 2x^2 - 1$
- $T_3(x) = 4x^3 - 3x$

### Key Property

**T_k(Λ) = U T_k(Λ̃) U^T**

So we can compute spectral filter as:

**g_θ(L) x = ∑_{k=0}^{K} θ_k T_k(Λ̃) x**

Where **Λ̃** is normalized Laplacian eigenvalues.

### Recursive Computation

Instead of matrix multiplication, we can compute recursively:

**T_0(L̃) x = x**
**T_1(L̃) x = L̃ x**
**T_k(L̃) x = 2 L̃ T_{k-1}(L̃) x - T_{k-2}(L̃) x**

**Computational cost: O(K × |E|)** instead of O(n³)!

This makes spectral methods practical for large graphs.

---

## From Spectral to Spatial: The GCN Layer

### The Key Insight

Instead of using high-order Chebyshev polynomials, GCNs use **first-order approximation** (K=1):

**g_θ(L) x ≈ θ_0 x + θ_1 L x = θ_0 x + θ_1 (D - A) x**

Rearranging:
**g_θ(L) x = θ_0 x + θ_1 (D x - A x)**

### Further Simplification

Assume **θ_0 = θ_1 = θ** (weight sharing):

**g_θ(L) x = θ (x + D x - A x) = θ (I + D - A) x**

Since D - A is the unnormalized Laplacian:
**(D - A) x** means subtract weighted neighborhood sum

So **x + (D-A)x = x + x - Ax = 2x - Ax** conceptually represents "self + neighbors"

### Using Normalized Laplacian

For better numerical stability, use normalized Laplacian L̃ = I - D^(-1/2) A D^(-1/2):

**g_θ(L̃) x = θ (I + D^(-1/2) A D^(-1/2)) x**

This leads to the core GCN update:

**H^(k) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(k-1) W^(k))**

Where:
- **Ã = A + I** (add self-loops)
- **D̃**: Degree matrix of Ã
- **W^(k)**: Learnable weight matrix
- **σ**: Activation function (ReLU)

---

## GCN Architecture and Layer Formulation

### The GCN Layer

The fundamental operation in a GCN layer is:

$$Z^{(k+1)} = \sigma(\tilde{A} H^{(k)} W^{(k)})$$

Where:
- $H^{(k)} \in \mathbb{R}^{n \times d_k}$: Node representations at layer k
  - $n$ = number of nodes
  - $d_k$ = dimension at layer k
- $W^{(k)} \in \mathbb{R}^{d_k \times d_{k+1}}$: Weight matrix
- $\tilde{A} = D^{-1/2}(A + I)D^{-1/2}$: Normalized adjacency with self-loops
- $\sigma$: Nonlinearity (typically ReLU)
- $Z^{(k+1)} \in \mathbb{R}^{n \times d_{k+1}}$: Output representations

### Normalized Adjacency Matrix Details

The normalized adjacency matrix is crucial:

$$\tilde{A} = \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2}$$

Where:
- $A$: Original adjacency matrix
- $I$: Identity matrix (self-loops)
- $\tilde{D}$: Diagonal degree matrix of $(A + I)$
- $\tilde{D}[i,i] = \deg(i) + 1$ (degree plus self-loop)

**Symmetrically normalized** version:
$$\tilde{A}[i,j] = \frac{(A+I)[i,j]}{\sqrt{(\deg(i)+1)(\deg(j)+1)}}$$

**Why this normalization?**
1. **Self-loops**: Each node incorporates its own features
2. **Degree normalization**: Prevents high-degree nodes from dominating
3. **Symmetric**: Preserves undirected nature
4. **Numerical stability**: Keeps values in reasonable range

### Multi-Layer GCN

Stack multiple GCN layers:

**Input**: $X^{(0)} = X$ (node features)

$$\text{Layer 1: } Z^{(1)} = \sigma(\tilde{A} X^{(0)} W^{(0)})$$

$$\text{Layer 2: } Z^{(2)} = \sigma(\tilde{A} Z^{(1)} W^{(1)})$$

$$\text{Layer L: } Z^{(L)} = \sigma(\tilde{A} Z^{(L-1)} W^{(L-1)})$$

**Output**: For node classification, typically:
- Remove activation from last layer: $\tilde{Y} = \tilde{A} Z^{(L-1)} W^{(L-1)}$
- Or keep activation but use softmax: $\hat{Y} = \text{softmax}(\tilde{A} Z^{(L-1)} W^{(L-1)})$

---

## Layer-Wise Propagation Rule

### Message Passing Perspective

Each GCN layer can be viewed as message passing:

**1. Message computation**: No explicit message function; node features are the messages

**2. Aggregation**:
$$m_v^{(k)} = \sum_{u \in N(v) \cup \{v\}} \frac{(A+I)[u,v]}{\sqrt{(\deg(u)+1)(\deg(v)+1)}} \cdot h_u^{(k-1)}$$

This is normalized neighbor aggregation.

**3. Update**:
$$h_v^{(k)} = \sigma(W^{(k)} \cdot m_v^{(k)})$$

Apply weight matrix and activation.

### Intuition

Each layer:
1. **Gathers**: Collect information from neighbors (normalized)
2. **Transforms**: Apply learnable weight matrix
3. **Activates**: Pass through nonlinearity

After k layers, node has receptive field of k-hop neighborhood.

### Receptive Field Growth

```
1-layer GCN:  Each node sees direct neighbors
2-layer GCN:  Each node sees 2-hop neighborhoods
k-layer GCN:  Each node sees k-hop neighborhoods
```

### Adding Bias Terms

The full GCN layer with bias:

**Z^(k+1) = σ(Ã H^(k) W^(k) + b^(k))**

Where **b^(k) ∈ ℝ^(d_{k+1})** is a learnable bias vector.

---

## Semi-Supervised Learning with GCNs

### Problem Setup

**Semi-supervised node classification**:
- Graph with n nodes and their features
- Only m nodes have labels (m << n)
- Goal: Predict labels for unlabeled nodes
- All nodes contribute to learning (via unlabeled data)

### Training Objective

Standard cross-entropy loss on labeled nodes only:

$$\mathcal{L} = -\sum_{i \in \mathcal{Y}_L} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})$$

Where:
- $\mathcal{Y}_L$: Set of labeled nodes
- $y_{ic}$: True label (one-hot)
- $\hat{y}_{ic} = \text{softmax}(Z^{(L)})_{ic}$: Predicted probability
- $C$: Number of classes

**Key difference from supervised learning**: Loss is computed on small labeled set, but network uses structural information from entire graph.

### Intuition: Why It Works

1. **Structure matters**: Similar nodes in graph tend to have similar labels
2. **Propagation**: Information flows through network via edges
3. **Regularization**: Unlabeled nodes (via graph structure) act as regularizer
4. **Inductive bias**: Graph structure encodes domain knowledge

### Training Process

1. **Forward pass**:
   - Compute node embeddings through GCN layers
   - Get predictions for all nodes

2. **Loss computation**:
   - Compute loss only on labeled nodes
   - Unlabeled nodes still affect gradients via graph propagation

3. **Backward pass**:
   - Gradients flow through entire network
   - Network learns to predict labels and respect graph structure

4. **Optimization**:
   - Update weights using SGD or Adam
   - Typically: small learning rate, weight decay regularization

### Inference

Once trained:
1. Forward pass through entire graph
2. Get predictions for all nodes
3. Optionally: use predicted labels for downstream tasks

---

## Mathematical Derivations

### Derivation of GCN from Spectral Convolution

**Step 1: Spectral convolution** (Bruna et al.)
**h_v^(k) = σ(∑_{j=1}^{d_k} g_θ(L) x_j)**

**Step 2: Chebyshev approximation** (Hammond et al.)
**g_θ(L) ≈ ∑_{r=0}^{K} θ_r T_r(L̃)**

With rescaled Laplacian: **L̃ = 2L/λ_max - I**

**Step 3: First-order approximation** (Kipf & Welling)
Keep only K=1 terms:
**g_θ(L̃) ≈ θ_0 + θ_1 L̃**

**Step 4: Assume θ_0 = -θ_1** (weight tying):
**g_θ(L̃) ≈ θ(I - L̃) = θ(2I - 2L/λ_max) = θ(I + D^(-1/2) A D^(-1/2))**

(Assuming λ_max ≈ 2)

**Step 5: Add self-loops and reformulate**:
**h_v^(k) = σ(W^(k) (D+I)^(-1/2) (A+I) (D+I)^(-1/2) H^(k-1))**

This is the GCN layer!

### Gradient Flow Analysis

For a k-layer GCN, the loss gradient w.r.t. layer l weights:

**∂L/∂W^(l) = (∂L/∂Ŷ) · (∂Ŷ/∂Z^(k)) · ... · (∂Z^(l+1)/∂Z^(l)) · (∂Z^(l)/∂W^(l))**

The chain of Jacobian terms can suffer from:
1. **Vanishing gradients**: If Jacobians < 1
2. **Exploding gradients**: If Jacobians > 1

Solutions:
- **Normalization**: Batch norm, layer norm
- **Skip connections**: Preserve gradients
- **Limited depth**: Keep network shallow (2-4 layers typical)

---

## Complexity Analysis

### Computational Complexity

**Per forward pass**:

1. **Sparse matrix multiplication Ã H^(k)**:
   - Cost: **O(|E| × d_k)** (where |E| is number of edges)
   - H^(k) is n × d_k dense matrix
   - Ã is sparse n × n matrix

2. **Dense matrix multiplication result × W^(k)**:
   - Cost: **O(n × d_k × d_{k+1})**

3. **Activation function**: **O(n × d_{k+1})**

**Total per layer**: **O(|E| × d_k + n × d_k × d_{k+1})**

For most sparse graphs: **O(|E| × d)** where d is feature dimension

**For L layers**: **O(L × (|E| × d + n × d²))**

### Space Complexity

**Storage**:
- Adjacency matrix Ã: **O(|E|)** (sparse storage)
- Node features: **O(n × d)**
- Weight matrices: **O(L × d²)**
- Activations (batch): **O(n × d)**

**Total**: **O(|E| + n × d + L × d²)**

### Memory-Efficient Variants

For very large graphs, use:

1. **Mini-batch training**: Not all nodes, random sampling
   - Reduces memory: **O(batch_size × d)**
   - Each node has fixed-size neighborhood

2. **Sampling neighbors**: GraphSAGE-style sampling
   - Trade-off accuracy for memory/speed
   - Use weighted sampling based on edge weights

3. **Layer-wise sampling**: Cluster-GCN
   - Sample subgraphs for each layer
   - Further reduce memory footprint

### Scalability Properties

**GCN scales well for**:
- Sparse graphs (social networks, citation networks)
- Moderate feature dimensions (< 512)
- Semi-supervised setting (compute loss on small labeled set)

**GCN challenges**:
- Dense graphs: O(n²) storage for adjacency
- High feature dimensions: O(d²) weights per layer
- Very deep networks: Over-smoothing problem

---

## Practical Considerations

### Weight Initialization

For GCN weights, use:

**Glorot uniform** (Xavier initialization):
**W^(l) ~ Uniform[-√(6/(d_in + d_out)), √(6/(d_in + d_out))]**

Why: Maintains signal variance through network.

### Regularization

1. **L2 regularization**: Penalize large weights
   **L_total = L_task + λ ||W||_F²**

2. **Dropout**: Random node feature dropout
   - Drop features with probability p
   - Scale remaining by 1/(1-p)

3. **Early stopping**: Monitor validation accuracy
   - Stop when validation accuracy plateaus

4. **Label smoothing**: Soften hard labels
   **y_soft = (1 - ε) y_hard + ε/C**

### Batch Normalization

Often helpful in GCNs:

**h = BN(Ã H^(k) W^(k))**

Parameters:
- Learnable scale γ and shift β
- Running statistics (momentum = 0.1)

Benefits: Accelerates convergence, reduces variance.

### Optimization Tips

1. **Learning rate**: Usually 0.01 to 0.001
   - Too high: divergence
   - Too low: slow convergence

2. **Weight decay**: 5e-4 to 5e-3
   - Prevents overfitting on small labeled sets

3. **Optimizer**: Adam often better than SGD
   - Adaptive learning rates help

4. **Epochs**: Often converges in 100-300 epochs
   - With early stopping

---

## Limitations and Challenges

### Over-Smoothing

After many layers, node representations converge:
- All nodes have similar embeddings
- Loss of discriminative power
- Typical limit: 2-4 layers

**Solutions**:
- Residual connections: h^(k) = h^(k-1) + f(h^(k-1))
- Skip connections to input layer
- Jumping knowledge: Combine outputs from all layers

### Limited Receptive Field

With k layers, only see k-hop neighborhood. For sparse graphs with high diameter:
- May need many layers (causes over-smoothing)
- Bottleneck for very distant nodes

**Solutions**:
- Graph rewiring: Add edges strategically
- Virtual nodes: Global node connected to all
- Attention mechanisms: Focus on important distant nodes

### Fair Graph Representation

GCNs can amplify biases in training data:
- Minority node classes may be poorly represented
- Graph structure might be biased

**Solutions**:
- Balanced sampling
- Class weights in loss
- Fairness-aware training

### Scalability

GCN training on massive graphs can be slow:

**Solutions**:
- Mini-batch training with neighbor sampling
- Distributed training across machines
- Approximate inference

---

## Comparison with Other Architectures

### vs. Graph Attention Networks (GAT)

**GCN**:
- Fixed normalization (D^(-1/2) A D^(-1/2))
- Computational efficient
- Less expressive (rigid aggregation)

**GAT**:
- Learned attention weights
- More expressive
- More computationally expensive

### vs. GraphSAGE

**GCN**:
- Uses entire graph (transductive)
- Better for semi-supervised learning
- Not naturally inductive

**GraphSAGE**:
- Uses neighborhood sampling
- Works inductively on unseen nodes
- More memory efficient

### vs. GIN (Graph Isomorphism Network)

**GCN**:
- First-order Chebyshev polynomial
- Simpler model
- Less theoretically motivated

**GIN**:
- Based on Weisfeiler-Leman test
- Provably optimal expressiveness
- Often requires more parameters

---

## Summary

In this lesson, we covered:

1. **Spectral graph theory**: Laplacian, eigenvalues, eigenvectors
2. **Spectral convolutions**: How to convolve on graphs
3. **Chebyshev approximation**: Making spectral methods practical
4. **GCN layer**: From spectral to spatial interpretation
5. **Multi-layer GCNs**: Stacking layers and receptive fields
6. **Semi-supervised learning**: Training with limited labels
7. **Complexity analysis**: Time and space requirements
8. **Practical implementation**: Initialization, regularization, optimization

---

## Key Takeaways

- GCNs bridge spectral and spatial perspectives on graphs
- The normalized adjacency matrix is the core operation
- First-order Chebyshev approximation leads to simple yet effective layer
- Semi-supervised learning leverages graph structure as regularization
- Limited depth (2-4 layers) works best due to over-smoothing
- Efficient sparse matrix operations make GCNs scalable

---

## Further Reading

1. **Kipf & Welling (2017)**: "Semi-Supervised Classification with Graph Convolutional Networks"
2. **Hammond et al. (2011)**: "Wavelets on Graphs via Spectral Graph Theory"
3. **Bruna et al. (2014)**: "Spectral Networks and Deep Locally Connected Networks on Graphs"
4. **Shuman et al. (2013)**: "The Emerging Field of Signal Processing on Graphs"
5. **Chung (1997)**: "Spectral Graph Theory" (comprehensive reference)

---

## Exercises

1. Derive the GCN layer equation from first principles using spectral graph theory
2. Implement normalized adjacency matrix computation for a simple graph
3. Verify that the normalized adjacency matrix is symmetric
4. Analyze how degree normalization affects aggregation
5. Compute receptive field sizes for networks of different depths
6. Implement from scratch: GCN layer without using PyTorch Geometric
7. Experiment with different values of K in Chebyshev approximation
8. Analyze over-smoothing: plot node embedding similarity with depth
9. Compare convergence with and without residual connections
10. Implement mini-batch training with neighbor sampling

---

**Next Lesson**: We'll implement GCNs in PyTorch and apply them to real datasets!
