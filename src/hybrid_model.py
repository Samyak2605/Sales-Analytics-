"""
hybrid_model.py
===============
Implementation of Phase 3 Hybrid Innovation: Neural-Probabilistic Gating Network.

### Hybrid Innovation Logic (Symbiotic)
Rather than naively averaging the predictions of Phase 1 (ML) and Phase 2 (DL),
this architecture implements a differentiable *Gating Mechanism*. 

1. The Phase 1 Logistic Regression acts as an explicit, well-calibrated probabilistic prior.
2. The Phase 2 Deep Tabular model learns non-linear latent embeddings.
3. The Fusion Gate learns to dynamically weight between the ML prior and the DL non-linearities 
   based on the input's latent representation. For instance, if the DL model detects a rare 
   product combination that the linear ML model misinterprets, the gate shifts weight to the DL branch.
   This achieves true Neuro-Symbolic / Differentiable Programming synergy.
"""

import torch
import torch.nn as nn
from src.dl_model import CategoricalEmbeddingNet

class ProbabilisticGatingFusionNet(nn.Module):
    """
    Symbiotic Hybrid Network mapping ML priors alongside DL latent spaces.
    """
    def __init__(self, embedding_sizes, num_numeric, dl_hidden_dims=[128, 64], dropout_rate=0.4):
        super(ProbabilisticGatingFusionNet, self).__init__()
        
        # ─── DL Feature Extractor (Phase 2 Component) ───
        # We reuse the core architecture of Phase 2, but remove its final classification head
        # to expose the deep latent representation.
        self.dl_feature_extractor = CategoricalEmbeddingNet(
            embedding_sizes=embedding_sizes,
            num_numeric=num_numeric,
            hidden_dims=dl_hidden_dims,
            mode="full",
            dropout_rate=dropout_rate
        )
        # Remove the last linear layer (which was outputting 1 logit)
        # to expose the 64-dim latent space
        latent_dim = dl_hidden_dims[-1]
        self.dl_feature_extractor.classifier = self.dl_feature_extractor.classifier[:-1]
        
        # ─── Symbiotic Gating Mechanism ───
        # Learns how much to trust the DL latent representation vs the ML prior
        self.gate_layer = nn.Sequential(
            nn.Linear(latent_dim + 1, 32), # +1 for the ML prior
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output is between 0 and 1
        )
        
        # ─── Final Differentiable Classifier ───
        self.dl_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, cat_x, num_x, ml_prior):
        """
        cat_x: Categorical ordinal tensors
        num_x: Scaled numeric tensors
        ml_prior: The probability P(Win) from Phase 1 Logistic Regression, shape (Batch, 1)
        """
        # 1. Extract DL Latent Representation
        dl_latent = self.dl_feature_extractor(cat_x, num_x) # Shape: (Batch, latent_dim)
        
        # 2. Compute the Gate (Dynamic Attention)
        # Combine latent context with the ML prior to decide fusion weight
        gate_input = torch.cat((dl_latent, ml_prior), dim=1)
        gate_weight = self.gate_layer(gate_input) # Shape: (Batch, 1)
        
        # 3. DL Prediction Branch
        dl_logit = self.dl_head(dl_latent) # Shape: (Batch, 1)
        
        # Convert ML prior (prob) to logit for mathematical fusion stability
        # logit(p) = log(p / (1 - p)). Add epsilon to prevent log(0)
        eps = 1e-7
        ml_prob_clamped = torch.clamp(ml_prior, eps, 1.0 - eps)
        ml_logit = torch.log(ml_prob_clamped / (1.0 - ml_prob_clamped))
        
        # 4. Synergistic Fusion
        # If gate_weight is close to 1, rely on DL. If close to 0, rely on ML.
        final_logit = (gate_weight * dl_logit) + ((1.0 - gate_weight) * ml_logit)
        
        return final_logit
