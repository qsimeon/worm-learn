<!-- 927884c1-8692-4cde-923b-a5b19ef7830f 7366ff48-c94b-403e-895c-e33efb0685b2 -->
# Simplify Attention Model and Connectome Comparison

## Overview

Refactor the attention model to be simpler and more interpretable: use a single attention head, remove value projection (treat input X directly as values), remove output projection, and combine Witvliet connectomes into one averaged/symmetrized ground-truth matrix.

## Changes

### 1. Update CONFIG (Cell 0)

- Remove `num_heads` parameter (always use 1 head)
- Remove `hidden_dim` parameter references in CONFIG (or keep for Q/K projection dimension)
- Keep `use_attention_scores` option

### 2. Simplify AttentionHeads Model (Cell ~10)

**Current architecture:**

- W_Q, W_K, W_V projections with multi-head support
- Output projection to map back to d_model
- Complex head concatenation logic

**New simplified architecture:**

```python
class AttentionHeads(nn.Module):
    def __init__(self, d_model, n_hidden, use_attention_scores=False):
        # Remove num_heads parameter (always 1)
        # Remove W_V projection (use input X directly as values)
        # Remove output_proj (no projection needed after attention)
        self.W_Q = nn.Linear(d_model, n_hidden)
        self.W_K = nn.Linear(d_model, n_hidden)
        self.d_model = d_model
        self.n_hidden = n_hidden
        self.use_attention_scores = use_attention_scores
    
    def forward(self, x, attn_mask=None):
        # x: (batch, n_neurons, d_model)
        # Compute Q, K projections
        Q = self.W_Q(x)  # (batch, n_neurons, n_hidden)
        K = self.W_K(x)  # (batch, n_neurons, n_hidden)
        
        # Compute attention scores: Q @ K.T / sqrt(n_hidden)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.n_hidden ** 0.5)
        # scores: (batch, n_neurons, n_neurons)
        
        # Apply diagonal mask
        # ... masking logic ...
        
        # Compute attention weights (softmax)
        attn_weights = torch.softmax(masked_scores, dim=-1)
        
        # Apply attention directly to input X (not to a projected V)
        if self.use_attention_scores:
            reconstructed = torch.bmm(attn_scores, x)  # (batch, n_neurons, d_model)
        else:
            reconstructed = torch.bmm(attn_weights, x)  # (batch, n_neurons, d_model)
        
        return reconstructed, attn_weights, attn_scores
```

**Key changes:**

- Remove `num_heads` parameter and all multi-head logic
- Remove `W_V` projection layer
- Remove `output_proj` layer
- Use input `x` directly as values in attention computation
- Simplify einsum operations to bmm (batch matrix multiply)
- Return shapes: attn_weights and attn_scores are now (batch, n_neurons, n_neurons)

### 3. Update Model Instantiation (Cell ~10)

Remove `num_heads` from model creation:

```python
model = AttentionHeads(
    d_model=CONFIG['embedding_dim'],
    n_hidden=CONFIG['hidden_dim'],
    use_attention_scores=CONFIG['use_attention_scores']
).to(device)
```

### 4. Update All References to Attention Shapes

Throughout the notebook, attention matrices are currently `(batch, num_heads, n_neurons, n_neurons)`. Update to `(batch, n_neurons, n_neurons)`:

- Visualization cells that compute `attn_weights.mean(dim=(0, 1))` → change to `attn_weights.mean(dim=0)`
- Any indexing like `attn_weights[0, 0]` → change to `attn_weights[0]`

### 5. Combine Witvliet 7 and 8 Connectomes (Cell ~21)

**Current approach:** Compare attention to witvliet_7 and witvliet_8 separately

**New approach:** Combine into single ground-truth matrix

```python
# Create full adjacency matrices for all neurons in union
all_neurons = list(neurons_7_all.union(neurons_8_all))
gt_adj_7_full = create_adjacency_matrix(witvliet_7, all_neurons, synapse_type='chemical')
gt_adj_8_full = create_adjacency_matrix(witvliet_8, all_neurons, synapse_type='chemical')

# Average element-wise (treating missing connections as 0)
gt_adj_combined = (gt_adj_7_full + gt_adj_8_full) / 2

# Symmetrize: (A + A.T) / 2
gt_adj_combined_sym = (gt_adj_combined + gt_adj_combined.T) / 2

# Extract submatrix for overlapping neurons with transcriptome
overlap_neurons = list(transcriptome_neurons.intersection(set(all_neurons)))
overlap_indices_gt = [all_neurons.index(n) for n in overlap_neurons]
overlap_indices_attn = [neuron_classes.index(n) for n in overlap_neurons]

gt_adj_final = gt_adj_combined_sym[np.ix_(overlap_indices_gt, overlap_indices_gt)]
attention_submatrix = attention_matrix[np.ix_(overlap_indices_attn, overlap_indices_attn)]
```

### 6. Update Visualization and Comparison (Cell ~22)

- Remove separate comparisons to witvliet_7 and witvliet_8
- Show single comparison: Learned Attention vs Combined GT Connectome
- Update plot titles and labels accordingly
- Simplify correlation analysis to single comparison

## Files to Modify

- `main_notebook.ipynb` (all cells with AttentionHeads model, training, evaluation, visualization, and ground-truth comparison)

## Testing

After changes, verify:

1. Model trains successfully with simplified architecture
2. Attention matrix shape is (batch, N, N) not (batch, heads, N, N)
3. Combined connectome matrix is symmetric
4. Visualizations display correctly with single ground-truth comparison

### To-dos

- [x] Remove num_heads from CONFIG, keep hidden_dim and use_attention_scores
- [x] Refactor AttentionHeads: remove W_V and output_proj, use X directly as values, single head only
- [x] Remove num_heads parameter from model creation calls
- [ ] Update all code that assumes (batch, heads, N, N) to (batch, N, N)
- [ ] Create combined averaged and symmetrized Witvliet 7+8 connectome matrix
- [ ] Update ground-truth comparison to show single combined connectome instead of separate 7 and 8