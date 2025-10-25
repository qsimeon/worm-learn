# Phase 2: Neural Dynamics - Attention Learning from Time Series

## Overview
Replace transcriptomic embeddings with neural dynamics time series as the representation to reconstruct. Train the same `SingleHeadSelfAttention` model to learn how neurons attend to each other's temporal activity patterns. Compare results with Phase 1 (transcriptomics).

## Part 1: Data Loading Pipeline

### 1.1 Create `load_neural_dynamics.ipynb`

**Location**: `notebooks/load_neural_dynamics.ipynb`

**Purpose**: Load and process Atanas 2023 neural activity data from JSON files into numpy format.

**Implementation**:
```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# List all available JSON files
data_dir = Path('../data/dynamics/wormwideweb/json')
json_files = sorted(list(data_dir.glob('*.json')))
print(f"Found {len(json_files)} JSON files")
for f in json_files:
    print(f"  {f.name}")

# Load first JSON file (or specify by name)
json_file = json_files[0]
print(f"\nLoading: {json_file.name}")

with open(json_file, 'r') as f:
    data = json.load(f)

# Extract key information
print(f"Available keys: {data.keys()}")

# Get neural activity data
trace_array = np.array(data['trace_array'])  # (N_neurons, time)
labeled = data['labeled']  # List of neuron identifiers

print(f"\nData shapes:")
print(f"  trace_array: {trace_array.shape}")
print(f"  labeled: {len(labeled)} neurons")
print(f"  Time points: {trace_array.shape[1]}")

# Examine neuron labels
print(f"\nSample neuron labels:")
for i in range(min(10, len(labeled))):
    print(f"  {i}: {labeled[i]}")

# Count identified vs unidentified neurons
identified = [label for label in labeled if isinstance(label, str) and not label.isdigit()]
unidentified = [label for label in labeled if not (isinstance(label, str) and not label.isdigit())]
print(f"\nNeuron identification:")
print(f"  Identified: {len(identified)}")
print(f"  Unidentified: {len(unidentified)}")

# Visualize sample traces
fig, axes = plt.subplots(3, 1, figsize=(15, 8))
for i in range(min(3, len(labeled))):
    axes[i].plot(trace_array[i, :1000])  # Plot first 1000 timepoints
    axes[i].set_title(f'Neuron {i}: {labeled[i]}')
    axes[i].set_ylabel('Activity (z-scored)')
axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()

# Save processed data
output_dir = Path('../data/dynamics/processed')
output_dir.mkdir(exist_ok=True)

output_file = output_dir / f"{json_file.stem}_processed.npz"
np.savez(output_file,
         trace_array=trace_array,
         labeled=labeled,
         metadata={'source': json_file.name,
                   'n_neurons': trace_array.shape[0],
                   'n_timepoints': trace_array.shape[1]})

print(f"\nSaved processed data to: {output_file}")
```

**Expected Output**: `data/dynamics/processed/{animal_id}_processed.npz` containing:
- `trace_array`: (N_neurons, T) neural activity matrix
- `labeled`: list of neuron identifiers
- `metadata`: dict with source info

---

## Part 2: Windowed Time Series Dataset

### 2.1 Create `NeuralDynamicsDataset` Class

**Location**: New cell in `main_notebook.ipynb` after existing dataset classes

**Purpose**: Generate overlapping temporal windows from neural activity time series.

**Implementation**:
```python
class NeuralDynamicsDataset(Dataset):
    """
    Dataset that creates overlapping temporal windows from neural activity time series.
    Each sample is a (N_neurons, seq_len) tensor representing a temporal chunk.
    """
    
    def __init__(self, trace_array, seq_len, stride=1, device='cpu'):
        """
        Args:
            trace_array: (N_neurons, T) numpy array of neural activity
            seq_len: length of each temporal window
            stride: stride for sliding window (default=1 for maximum overlap)
            device: torch device to move data to
        """
        self.trace_array = trace_array  # (N_neurons, T)
        self.seq_len = seq_len
        self.stride = stride
        self.device = device
        
        self.N_neurons, self.T = trace_array.shape
        
        # Calculate number of possible windows
        self.n_windows = (self.T - seq_len) // stride + 1
        
        print(f"NeuralDynamicsDataset:")
        print(f"  N_neurons: {self.N_neurons}")
        print(f"  Total timepoints: {self.T}")
        print(f"  seq_len: {seq_len}")
        print(f"  stride: {stride}")
        print(f"  n_windows: {self.n_windows}")
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        """
        Returns:
            tensor of shape (N_neurons, seq_len)
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        
        # Extract window
        window = self.trace_array[:, start_idx:end_idx]  # (N_neurons, seq_len)
        
        # Convert to tensor and move to device
        window_tensor = torch.tensor(window, dtype=torch.float32, device=self.device)
        
        return window_tensor
```

---

## Part 3: Refactor Reusable Code

### 3.1 Modularize Training Loop

**Create function**: `train_attention_model()`

```python
def train_attention_model(model, train_dataset, val_dataset, config, device):
    """
    Generic training function for attention model.
    
    Args:
        model: SingleHeadSelfAttention model
        train_dataset: training Dataset
        val_dataset: validation Dataset
        config: dict with hyperparameters
        device: torch device
    
    Returns:
        dict with training history and final model
    """
    # Set up training
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Track losses
    history = {
        'train_losses': [],
        'train_recon_losses': [],
        'train_consistency_losses': [],
        'val_losses': [],
        'batch_numbers': []
    }
    
    # Training loop
    total_batches = 0
    for batch_idx, train_batch in enumerate(train_loader):
        total_batches += 1
        train_batch = train_batch.to(device)
        
        # Forward pass
        model.train()
        train_reconstructed, train_attn_weights, train_attn_scores = model(train_batch, mask_other=False)
        
        # Reconstruction loss
        recon_loss = criterion(train_reconstructed, train_batch)
        
        # Batch consistency loss
        if config['consistency_type'] == 'frobenius':
            consistency_loss = batch_consistency_loss_frobenius(train_attn_weights, 
                                                                 weight=config['consistency_weight'])
        elif config['consistency_type'] == 'variance':
            consistency_loss = batch_consistency_loss_variance(train_attn_weights,
                                                                weight=config['consistency_weight'])
        
        train_loss = recon_loss + consistency_loss
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Track losses
        history['train_losses'].append(train_loss.item())
        history['train_recon_losses'].append(recon_loss.item())
        history['train_consistency_losses'].append(consistency_loss.item())
        history['batch_numbers'].append(total_batches)
        
        # Validation
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                val_reconstructed, val_attn_weights, _ = model(val_batch, mask_other=False)
                val_loss = criterion(val_reconstructed, val_batch) + \
                          batch_consistency_loss_variance(val_attn_weights, weight=config['consistency_weight'])
                batch_val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(batch_val_losses)
        history['val_losses'].append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"\tBatch {total_batches} - Recon: {recon_loss.item():.6f}, "
                  f"Consistency: {consistency_loss.item():.6f}, "
                  f"Total: {train_loss.item():.6f}, Val: {avg_val_loss:.6f}")
    
    return history
```

### 3.2 Modularize Analysis Functions

**Create function**: `analyze_attention_matrix()`

```python
def analyze_attention_matrix(model, dataset, neuron_labels, device, 
                             ground_truth_connectomes=None):
    """
    Extract and analyze learned attention matrix.
    
    Args:
        model: trained SingleHeadSelfAttention model
        dataset: Dataset used for evaluation
        neuron_labels: list of neuron identifiers
        device: torch device
        ground_truth_connectomes: optional dict with connectome data
    
    Returns:
        dict with attention matrix and analysis results
    """
    model.eval()
    
    # Collect attention matrices from all samples
    all_attn_weights = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, attn_weights, _ = model(batch, mask_other=False)
            all_attn_weights.append(attn_weights.cpu().numpy())
    
    # Average across all samples
    all_attn_weights = np.concatenate(all_attn_weights, axis=0)
    attention_matrix = all_attn_weights.mean(axis=0)  # (N, N)
    
    results = {
        'attention_matrix': attention_matrix,
        'neuron_labels': neuron_labels,
        'sparsity': (attention_matrix < 0.01).sum() / attention_matrix.size,
        'mean': attention_matrix.mean(),
        'std': attention_matrix.std()
    }
    
    # Visualize
    plt.figure(figsize=(12, 10))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('Learned Attention Matrix (Averaged)')
    plt.xlabel('Target Neuron')
    plt.ylabel('Source Neuron')
    
    # Add neuron labels if not too many
    if len(neuron_labels) <= 50:
        plt.xticks(range(len(neuron_labels)), neuron_labels, rotation=90, fontsize=6)
        plt.yticks(range(len(neuron_labels)), neuron_labels, fontsize=6)
    
    plt.tight_layout()
    plt.show()
    
    # Compare with ground truth if provided
    if ground_truth_connectomes:
        results['ground_truth_correlations'] = compare_with_ground_truth(
            attention_matrix, neuron_labels, ground_truth_connectomes
        )
    
    return results
```

**Create function**: `compare_with_ground_truth()`

```python
def compare_with_ground_truth(attention_matrix, neuron_labels, connectomes):
    """
    Compare learned attention with ground-truth connectomes.
    Reusable for both transcriptomics and dynamics phases.
    """
    from scipy.stats import pearsonr
    
    results = {}
    
    # Find overlapping neurons
    for connectome_name, connectome_data in connectomes.items():
        # ... existing comparison logic from Phase 1 ...
        # (correlation calculation, visualization, etc.)
        pass
    
    return results
```

---

## Part 4: Phase 2 Main Notebook Section

### 4.1 Configuration for Neural Dynamics

**Location**: New section in `main_notebook.ipynb`

```python
# ============================================================================
# PHASE 2: NEURAL DYNAMICS
# ============================================================================

# Configuration for neural dynamics phase
CONFIG_DYNAMICS = {
    'seq_len': 1024,           # Temporal window length (embedding dimension)
    'stride': 1,               # Sliding window stride
    'hidden_dim': 256,         # Hidden dimension for Q/K/V
    'batch_size': 32,          # Batch size
    'learning_rate': 1e-3,     # Learning rate
    'consistency_weight': 0.1, # Batch consistency regularization
    'consistency_type': 'variance',  # 'variance' or 'frobenius'
    'train_val_split': 0.8,    # Train/validation split
}

print("Phase 2 Configuration (Neural Dynamics):")
for key, value in CONFIG_DYNAMICS.items():
    print(f"  {key}: {value}")
```

### 4.2 Load Neural Dynamics Data

```python
# Load processed neural dynamics data
dynamics_file = Path('data/dynamics/processed/atanas_kim_2023-2022-07-20-01_processed.npz')

with np.load(dynamics_file, allow_pickle=True) as data:
    trace_array = data['trace_array']      # (N_neurons, T)
    neuron_labels_dynamics = data['labeled'].tolist()
    metadata = data['metadata'].item()

print(f"Loaded neural dynamics data:")
print(f"  Source: {metadata['source']}")
print(f"  N_neurons: {metadata['n_neurons']}")
print(f"  N_timepoints: {metadata['n_timepoints']}")
print(f"  Shape: {trace_array.shape}")

# Store neuron count
N_neurons_dynamics = trace_array.shape[0]
```

### 4.3 Create Datasets

```python
# Calculate train/val split
T_total = trace_array.shape[1]
train_end_time = int(T_total * CONFIG_DYNAMICS['train_val_split'])

# Split time series temporally
train_trace = trace_array[:, :train_end_time]
val_trace = trace_array[:, train_end_time:]

print(f"Data split:")
print(f"  Train: {train_trace.shape}")
print(f"  Val: {val_trace.shape}")

# Create datasets
train_dataset_dynamics = NeuralDynamicsDataset(
    trace_array=train_trace,
    seq_len=CONFIG_DYNAMICS['seq_len'],
    stride=CONFIG_DYNAMICS['stride'],
    device=device
)

val_dataset_dynamics = NeuralDynamicsDataset(
    trace_array=val_trace,
    seq_len=CONFIG_DYNAMICS['seq_len'],
    stride=CONFIG_DYNAMICS['stride'],
    device=device
)
```

### 4.4 Initialize Model

```python
# Initialize model for dynamics phase
# d_model = seq_len (temporal window is the "embedding")
model_dynamics = SingleHeadSelfAttention(
    d_model=CONFIG_DYNAMICS['seq_len'],
    n_hidden=CONFIG_DYNAMICS['hidden_dim'],
    use_attention_scores=False
).to(device)

print(f"\nModel architecture:")
print(f"  d_model (seq_len): {CONFIG_DYNAMICS['seq_len']}")
print(f"  n_hidden: {CONFIG_DYNAMICS['hidden_dim']}")
print(f"  Total parameters: {sum(p.numel() for p in model_dynamics.parameters()):,}")
```

### 4.5 Train Model

```python
# Train the model
print("\n" + "="*80)
print("Training attention model on neural dynamics...")
print("="*80 + "\n")

history_dynamics = train_attention_model(
    model=model_dynamics,
    train_dataset=train_dataset_dynamics,
    val_dataset=val_dataset_dynamics,
    config=CONFIG_DYNAMICS,
    device=device
)

# Plot training curves
plot_training_curves(history_dynamics, title="Neural Dynamics Training")
```

### 4.6 Analyze Results

```python
# Analyze learned attention matrix
results_dynamics = analyze_attention_matrix(
    model=model_dynamics,
    dataset=val_dataset_dynamics,
    neuron_labels=neuron_labels_dynamics,
    device=device,
    ground_truth_connectomes={
        'witvliet_combined': witvliet_combined_matrix,
        'functional': func_connectivity_matrix
    }
)

# MDS visualization with metadata (if neuron labels match master sheet)
# ... similar to Phase 1 MDS visualization ...
```

---

## Part 5: Comparison Between Phases

### 5.1 Side-by-Side Attention Matrices

```python
# Compare Phase 1 (Transcriptomics) vs Phase 2 (Dynamics)
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Phase 1: Transcriptomics
im1 = axes[0].imshow(attention_matrix_transcriptomics, cmap='viridis', aspect='auto')
axes[0].set_title('Phase 1: Attention from Transcriptomics', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Target Neuron')
axes[0].set_ylabel('Source Neuron')
plt.colorbar(im1, ax=axes[0], label='Attention Weight')

# Phase 2: Dynamics
im2 = axes[1].imshow(results_dynamics['attention_matrix'], cmap='viridis', aspect='auto')
axes[1].set_title('Phase 2: Attention from Neural Dynamics', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Target Neuron')
axes[1].set_ylabel('Source Neuron')
plt.colorbar(im2, ax=axes[1], label='Attention Weight')

plt.tight_layout()
plt.show()
```

### 5.2 Correlation Comparison

```python
# Compare correlations with ground truth
comparison_df = pd.DataFrame({
    'Method': ['Transcriptomics', 'Neural Dynamics'],
    'Witvliet Correlation': [
        corr_transcriptomics_witvliet,
        results_dynamics['ground_truth_correlations']['witvliet_combined']['correlation']
    ],
    'Functional Correlation': [
        corr_transcriptomics_functional,
        results_dynamics['ground_truth_correlations']['functional']['correlation']
    ]
})

print("\nPhase Comparison:")
print(comparison_df)

# Visualize
comparison_df.plot(x='Method', y=['Witvliet Correlation', 'Functional Correlation'],
                   kind='bar', figsize=(10, 6))
plt.title('Attention-Connectome Correlation: Transcriptomics vs Dynamics')
plt.ylabel('Pearson Correlation')
plt.xticks(rotation=0)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Implementation Order

1. **Create data loading notebook** - `load_neural_dynamics.ipynb`
2. **Process first JSON file** - Save to `processed/` directory
3. **Create `NeuralDynamicsDataset` class** - In main notebook
4. **Refactor training function** - Extract reusable `train_attention_model()`
5. **Refactor analysis functions** - Extract `analyze_attention_matrix()`, `compare_with_ground_truth()`
6. **Create Phase 2 section** - New section in main notebook
7. **Load and prepare dynamics data** - Load processed .npz file
8. **Train dynamics model** - Use refactored training function
9. **Analyze dynamics results** - Use refactored analysis functions
10. **Create comparison visualizations** - Side-by-side Phase 1 vs Phase 2

## Expected Outcomes

### Neural Dynamics Phase
- Attention matrix learned from temporal patterns of neural activity
- Model learns which neurons' activity is predictive of other neurons' activity
- Potentially stronger correlation with functional connectome (vs structural)
- Insights into dynamic vs static neural relationships

### Comparison with Transcriptomics
- Different attention patterns from different modalities
- Assessment of which modality produces better connectome approximation
- Understanding of complementary information in transcriptomics vs dynamics