# `network_inference_attention`

---

# Does the Connectome Pay Attention?  
## Learning Neural Connectivity Through Self-Supervised Reconstruction

---

## Abstract

We present a reconception of a connectome—the comprehensive map of neural connections in a brain—as a self-attention matrix over neural tokens, where each token is a high-dimensional vector representation of a neuron’s state. Using this reframing, we propose that connectivity patterns (how neurons are "wired" together) may emerge naturally from optimizing self-supervised objectives that require neurons to reconstruct each other's representations.

The most compressed representation of a connectome is an $N \times N$ weighted adjacency matrix, representing physical or functional connections between $N$ neurons. This adjacency matrix is the computational equivalent of a directed graph, where edge weights quantify connection strength (synaptic weights, correlation coefficients, etc.) between the $N$ nodes. We propose that this same $N \times N$ matrix can be viewed as an attention matrix in a self-attention mechanism, where the weight $A[i,j]$ represents how much neuron $i$ needs to "pay attention to" neuron $j$ to reconstruct its own representation.

---

## Related Work

- **Knowing Oneself By Knowing Others** (PI: Eva Dyer)

---

## Key Analogy to Sequence Modeling

- **Causal language modeling**: tokens are sequence elements, the task is next-token prediction, and a causal mask prevents "seeing the future".
- **Connectome modeling**: tokens are individual neurons, one possible task is self-reconstruction via other neurons, and a diagonal mask prevents "seeing yourself".

---

## Neural Tokens as Transcriptomic Embeddings

For *C. elegans*, we have bulk-sorted transcriptome data (gene expression profiles) for ~41 neuron classes across ~31,000 genes. Each neuron class can be embedded into a $D$-dimensional space using dimensionality reduction (PCA, t-SNE, etc.) on this transcriptomic data, creating an $(N, D)$ matrix where $N$ is the number of neuron classes.

---

## Neural Dynamics as Embeddings

We have calcium activity traces from labeled neurons in *C. elegans*. We can attempt to represent or ‘capture’ the neural dynamics history of each neuron as an embedding by using the last hidden state vector from some recurrent network model.

---

## The Reconstruction Task

The model must reconstruct each neuron's embedding using only the embeddings of all other neurons. This is enforced through a diagonal attention mask that prevents self-attention (neuron $i$ cannot attend to itself).

Mathematically, for neuron $i$ with embedding $e_i$, the model learns:
$$
\hat{e}_i = f(e_1, e_2, \ldots, e_{i-1}, e_{i+1}, \ldots, e_N)
$$
where $f$ involves computing attention weights (the emerging connectome) and using them to combine other neurons' embeddings.

---

## Model Architecture

A **SimpleSelfAttention** module that:

1. Takes input of shape `(batch, N, D)` where batch contains different random projections.
2. Projects each neuron’s embedding to a query and key. 
3. Value projection is an identity (values are the original embeddings).
4. Computes attention scores between all neuron pairs.
5. Applies a diagonal mask (forcing self-attention weights to zero).
6. Uses the masked attention weights to reconstruct each neuron's embedding as a weighted combination of others.
7. Optimizes reconstruction loss: $$||e_i - \hat{e}_i||^2$$.

The learned $N \times N$ attention weight matrix represents how much each neuron "attends to" every other neuron, which is your learned connectome.

---

## Validation Strategy

Compare the learned attention matrices (connectomes) to ground-truth connectivity data for *C. elegans*:

1. **Wired connectome:** Physical synaptic connections from electron microscopy (Witvliet et al. 2020; data available in your workspace)
2. **Functional connectome:** Correlation-based connectivity from paired stimulation experiments (Randi et al. 2023; data in your demo notebook)

The hypothesis is that transcriptomic similarity (as captured by the embeddings) should correlate with connectivity: neurons with similar gene expression patterns may be more likely to connect or have functional relationships.

### Why *C. elegans*?

*C. elegans* is the only multicellular organism for which all cells and cell types are defined, as is its entire developmental lineage.

*C. elegans* is ideal because:

1. Complete wired connectome is known (~300 neurons in hermaphrodite)
2. Functional connectomes from controlled experiments exist
3. Single-cell transcriptomic data available for most neuron classes
4. Ground truth allows for comparison to the learned attention matrices

---

## Broader Vision

While this implementation focuses on transcriptomic embeddings with a reconstruction objective, the framework generalizes to:

- **Functional connectomes:** using temporal activity patterns as embeddings + next-token prediction objective  
- **Other organisms:** applying the same principle wherever embeddings and ground-truth connectivity exist  
- **Multi-modal integration:** combining transcriptomic, functional, and anatomical data  

The key insight is that different types of connectomes (wired, functional, transcriptomic) might all emerge from optimizing different self-supervised objectives using appropriately chosen neural embeddings and attention masking strategies.

---