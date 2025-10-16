# network_inference_attention

---

**"The Connectome Pays Attention: Learning Neural Connectivity Through Self-Supervised Reconstruction"**

### Core Hypothesis
A connectome—the comprehensive map of neural connections in a brain—can be reinterpreted as a self-attention matrix over neural tokens, where each neuron is a discrete token. This reframing suggests that connectivity patterns (how neurons are "wired" together) may emerge naturally from optimizing self-supervised objectives that require neurons to reconstruct each other's representations.

### Conceptual Framework

**Traditional View**: A connectome is an N×N adjacency matrix representing physical or functional connections between N neurons, with edge weights quantifying connection strength (synaptic weights, correlation coefficients, etc.).

**Proposed Reframing**: This same N×N matrix can be viewed as an attention matrix in a self-attention mechanism, where the weight A[i,j] represents how much neuron i needs to "pay attention to" neuron j to reconstruct its own representation.

**Key Analogy to Sequence Modeling**: 
- In causal language modeling: tokens are sequence elements, the task is next-token prediction, and a causal mask prevents "seeing the future"
- In connectome modeling: tokens are individual neurons, the task is self-reconstruction via other neurons, and a diagonal mask prevents "seeing yourself"

### Simplified Implementation Strategy (Focus on Transcriptomic Data)

**Neural Tokens as Transcriptomic Embeddings**:
For C. elegans, you have bulk-sorted transcriptome data (gene expression profiles) for ~41 neuron classes across ~31,000 genes. Each neuron class can be embedded into a D-dimensional space using dimensionality reduction (PCA, t-SNE, etc.) on this transcriptomic data, creating an (N, D) matrix where N is the number of neuron classes.

**The Reconstruction Task**:
The model must reconstruct each neuron's embedding using only the embeddings of all other neurons. This is enforced through a diagonal attention mask that prevents self-attention (neuron i cannot attend to itself). Mathematically, for neuron i with embedding e_i, the model learns:

ê_i = f(e_1, e_2, ..., e_{i-1}, e_{i+1}, ..., e_N)

where f involves computing attention weights (the emerging connectome) and using them to combine other neurons' embeddings.

**Data Augmentation via Random Projections**:
Since you have only one (N, D) embedding matrix from the original transcriptome data, you need to generate multiple training samples. Your innovation is to create synthetic but meaningful variations by:
1. Initializing random MLPs (e.g., 3-layer networks mapping D→D)
2. Passing the original (N, D) matrix through these random MLPs
3. Treating each random MLP as generating one sample in your dataset

Critically, each MLP transforms all N neurons identically (N acts as a batch dimension), preserving the relative relationships between neuron embeddings while creating novel projections. This allows generation of effectively unlimited training samples.

**Model Architecture**:
An `AttentionHeads` module that:
1. Takes input of shape (batch, N, D) where batch contains different random projections
2. Projects each neuron embedding to queries, keys, and values
3. Computes attention scores between all neuron pairs
4. Applies a diagonal mask (forcing self-attention weights to zero)
5. Uses the masked attention weights to reconstruct each neuron's embedding as a weighted combination of others
6. Optimizes reconstruction loss: ||e_i - ê_i||²

The learned N×N attention weight matrix represents how much each neuron "attends to" every other neuron, which is your learned connectome.

**Baseline Comparison**:
To validate that the nonlinear attention mechanism captures something beyond simple linear relationships, implement a multiple linear regression baseline for each neuron:
- For neuron i: regress e_i onto all other neuron embeddings {e_j : j ≠ i}
- The regression coefficients form an alternative "connectome" matrix
- Compare this linear baseline to the attention-based approach

### Validation Strategy
Compare the learned attention matrices (connectomes) to ground-truth connectivity data for C. elegans:
1. **Wired connectome**: Physical synaptic connections from electron microscopy (Witvliet et al. 2020 data available in your workspace)
2. **Functional connectome**: Correlation-based connectivity from paired stimulation experiments (Randi et al. 2023 data in your demo notebook)

The hypothesis is that transcriptomic similarity (as captured by the embeddings) should correlate with connectivity—neurons with similar gene expression patterns may be more likely to connect or have functional relationships.

### Why C. elegans?
C. elegans is ideal because:
1. Complete wired connectome is known (302 neurons in hermaphrodite)
2. Functional connectomes from controlled experiments exist
3. Single-cell transcriptomic data available for most neuron classes
4. Ground truth allows validation of the learned attention matrices

### Broader Vision
While this implementation focuses on transcriptomic embeddings with a reconstruction objective, the framework generalizes to:
- Functional connectomes: using temporal activity patterns as embeddings + next-token prediction objective
- Other organisms: applying the same principle wherever embeddings and ground-truth connectivity exist
- Multi-modal integration: combining transcriptomic, functional, and anatomical data

The key insight is that different types of connectomes (wired, functional, transcriptomic) might all emerge from optimizing different self-supervised objectives using appropriately chosen neural embeddings and attention masking strategies.

---