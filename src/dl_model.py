"""
dl_model.py
===========
Implementation of Phase 2 Deep Learning architectures: Deep Tabular Embedding Network.

### Architecture Logic
We use an Entity-Embedding Architecturally designed for relational tabular data:
1. **Categorical Embeddings**: High-cardinality nominal values (e.g., Sales Agent, Product)
   are directly parameterized into dynamic `nn.Embedding` lookup tables. The network
   learns the "distance" or semantic correlations between distinct agents and products,
   a feature purely impossible for Phase 1 Logistic models using static sparse OHE arrays.
2. **Numeric Integration layer**: Directly processes metrics (Cycle days, Employees) alongside Embeddings.
3. **MLP Depth**: Allows deep feature intersections between `sales_agent` properties and `product` viability.

### Theoretical Rigor
- **Latent Spaces**: Replaces high-dimensional sparse representations (OHE) with lower-dimensional
  dense continuous mathematical spaces. This ensures gradients reliably propagate back to individual
  agents rather than suffering the "curse of dimensionality" common with Random Forests or linear models.
- **Normalization and Dropouts**: Extensive use of `BatchNorm1d` smooths non-convex loss 
  landscapes, enabling `AdamW` to accelerate effectively.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# ─── Dataset ─────────────────────────────────────────────────────────────────

class RelationalSalesDataset(Dataset):
    """
    Custom Dataset ingesting numerical arrays and categorical integer indices.
    """
    def __init__(self, cat_data: np.ndarray, num_data: np.ndarray, labels: np.ndarray):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat_data[idx], self.num_data[idx], self.labels[idx]

# ─── Multi-Modal Network Architecture ────────────────────────────────────────

class CategoricalEmbeddingNet(nn.Module):
    """
    Deep Tabular setup incorporating Embeddings for Categories, 
    Dense layers for Numerics, and a Fusion hierarchy.
    
    Ablation modes:
    - mode="full": Hybrid integration (Embeddings + Numerics)
    - mode="categorical_only": Only deeply learns Agent/Product semantic space.
    - mode="numeric_only": Pure MLP evaluating revenue and time arrays.
    """
    def __init__(self, embedding_sizes: list, num_numeric: int, 
                 mode="full", dropout_rate=0.4, hidden_dims=[128, 64]):
        super(CategoricalEmbeddingNet, self).__init__()
        self.mode = mode
        
        # ─── Embeddings Branch ───
        # List of (num_categories, embedding_dim) tuples
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cats, emb_dim) for num_cats, emb_dim in embedding_sizes
        ])
        
        n_emb_total = sum(e for _, e in embedding_sizes)
        
        # ─── Numeric Branch ───
        self.num_norm = nn.BatchNorm1d(num_numeric) if num_numeric > 0 else None
        
        # ─── Fusion Mechanics ───
        if self.mode == "categorical_only":
            input_dim = n_emb_total
        elif self.mode == "numeric_only":
            input_dim = num_numeric
        else:
            input_dim = n_emb_total + num_numeric
            
        # Classifier
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = h
            
        layers.append(nn.Linear(input_dim, 1))  # Output probability map
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        # 1. Process Categorical
        if self.mode in ["full", "categorical_only"]:
            y_cat = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            y_cat = torch.cat(y_cat, dim=1) # Concatenate along feature dimension
            
        # 2. Process Numeric
        if self.mode in ["full", "numeric_only"]:
            # Small fallback check if batch normalization fails on size=1
            if x_num.size(0) > 1 and self.num_norm is not None:
                y_num = self.num_norm(x_num)
            else:
                y_num = x_num
                
        # 3. Fuse & Classify
        if self.mode == "categorical_only":
            fused = y_cat
        elif self.mode == "numeric_only":
            fused = y_num
        else:
            fused = torch.cat((y_cat, y_num), dim=1)
            
        logits = self.classifier(fused)
        return logits
