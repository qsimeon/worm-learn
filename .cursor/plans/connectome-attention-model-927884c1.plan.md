<!-- 927884c1-8692-4cde-923b-a5b19ef7830f 5962d09d-bbb0-4927-b9f4-a14bf20f2da3 -->
# Connectome as Attention: Transcriptomic Reconstruction

## Overview

Build a novel framework treating the connectome as a self-attention matrix over neural tokens. Using C. elegans single-cell transcriptomic data, train an attention model to reconstruct each neuron's embedding from others, with the learned attention weights representing connectivity.

## Implementation Steps

### 1. Data Preparation and Embedding Generation

**File**: `new_notebook.ipynb`

Load and process the single-cell transcriptome data:

- Read `Single_Cell_TPM_threshold2.csv` (~13,669 genes × 128 neuron classes)
- Skip gene metadata columns (Gene_Name, Sequence_Name, Wormbase_ID)
- Transpose to (neurons × genes) format
- Apply log1p transformation and standardization
- Perform PCA to reduce to D dimensions (default D=1024, configurable)
- Generate initial (N, D) embedding matrix where N=128 neuron classes

Key decision: Use PCA over t-SNE for embeddings since we need to preserve linear structure and generate variations. Single-cell data provides much more granular neuron-level resolution than bulk-sorted data.

### 2. Custom Dataset with Random Projection Augmentation

**File**: `new_notebook.ipynb`

Implement `TranscriptomeEmbeddingDataset` class:

- Store the original (N, D) embedding matrix
- On-the-fly generation of samples via random MLP projections
- Random MLPs: 3-layer networks mapping D→D (e.g., D→2D→2D→D with ReLU)
- Each MLP initialized with different random seed
- Returns: augmented (N, D) matrices that preserve neuron relationships

Parameters to expose:

- `n_samples`: number of unique random projections per epoch
- `hidden_dim`: MLP hidden layer dimension
- `n_layers`: number of MLP layers

### 3. Attention Model Architecture

**File**: `new_notebook.ipynb`

Implement `AttentionHeads` module (adapt from existing code):

```
Input: (batch, N, D) tensor of neuron embeddings
Output: 
  - reconstructed embeddings: (batch, N, D)
  - attention weights: (batch, num_heads, N, N)
  - attention scores: (batch, num_heads, N, N)
```

Architecture components:

- Linear projections for Q, K, V (separate or shared across heads)
- Scaled dot-product attention with diagonal masking
- Multi-head attention (configurable number of heads)
- Value-weighted reconstruction: combine attended embeddings
- Output projection to reconstruct original embedding dimension

Key implementation detail: Diagonal mask sets attention weights A[i,i] = 0 (prevent self-attention).

### 4. Reconstruction Loss and Training Loop

**File**: `new_notebook.ipynb`

Training setup:

- Loss: MSE between original and reconstructed embeddings
- Optimizer: Adam with learning rate scheduling
- Batch size: flexible (e.g., 32 samples)
- Epochs: ~500-1000 (monitor convergence)

Track metrics:

- Reconstruction loss (train/val split)
- Average attention sparsity
- Attention weight statistics per neuron pair

Validation: Hold out some random projections for validation set.

### 5. Linear Regression Baseline

**File**: `new_notebook.ipynb`

Implement baseline comparison:

- For each neuron i, fit linear regression: e_i ~ Σ β_j * e_j (j ≠ i)
- Coefficients β form a "linear connectome" matrix
- Use regularization (Ridge or Lasso) to promote sparsity
- Train on same augmented data as attention model

Compare:

- Reconstruction error (baseline vs attention)
- Sparsity patterns in weight matrices
- Structure of learned weight matrices

### 6. Visualization of Learned Attention Matrices

**File**: `new_notebook.ipynb`

Visualize the learned attention patterns:

- Heatmaps: learned attention matrix aggregated across heads
- Compare single-head vs multi-head attention patterns
- Sort by neuron type if metadata available
- Attention weight distributions per neuron class
- Sparsity analysis: which neuron pairs have strongest attention

Note: Skip ground-truth connectome comparison for initial implementation. Focus on:

1. Does reconstruction work?
2. What patterns emerge in the learned attention?
3. Does attention outperform linear baseline?

### 7. Results and Initial Interpretation

**File**: `new_notebook.ipynb`

Generate initial results:

- Reconstruction performance: baseline vs attention (quantitative comparison)
- Visualize learned attention matrices
- Neuron class analysis: which classes are most/least predictable from transcriptomics?
- Head-specific analysis (if multi-head): do different heads capture different patterns?
- Ablation studies: effect of embedding dimension D, number of heads, training samples

Key questions to address:

1. Does the attention model outperform linear baseline in reconstruction?
2. What structure emerges in the learned attention matrices?
3. Which neuron classes are most/least predictable from transcriptomics?
4. Do multi-head patterns separate into interpretable connectivity types?

### 8. Documentation and Writeup

**File**: `new_notebook.ipynb`

Include markdown cells throughout with:

- Clear motivation and hypothesis
- Method descriptions with mathematical notation
- Interpretation of results
- Connections to broader "connectome as attention" framework
- Limitations and future directions (including ground-truth comparison)

This notebook serves as foundation for your paper draft.

## Key Files and Data

- Input data: `Single_Cell_TPM_threshold2.csv` (single-cell transcriptome, 128 neuron classes)
- Reference: `neuron_master_sheet.csv` (neuron metadata for type annotations)
- Output notebook: `new_notebook.ipynb`
- Existing reference: `qsimeon_CElegansNeuralDataDemo.ipynb` (for PCA/embedding methods)
- Ground-truth connectomes (for future use): `witvliet_2020_7.csv`, `witvliet_2020_8.csv`

## Technical Considerations

- Use PyTorch for model implementation (consistent with existing code)
- Leverage existing embedding generation code from demo notebook
- Default embedding dimension: D=1024 (configurable)
- Everything implemented in single notebook for easy debugging and visualization
- Focus on reconstruction task first; connectome comparison deferred to future work

### To-dos

- [ ] Load transcriptome data and generate initial PCA embeddings (N, D)
- [ ] Implement custom Dataset class with random MLP projection augmentation
- [ ] Implement AttentionHeads model with diagonal masking and reconstruction
- [ ] Set up training loop with reconstruction loss and validation
- [ ] Implement linear regression baseline for comparison
- [ ] Load and preprocess wired and functional connectome ground truth data
- [ ] Compare learned attention to ground truth connectomes with visualizations
- [ ] Generate comprehensive results analysis and interpretation