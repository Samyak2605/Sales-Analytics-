"""
train_hybrid.py
===============
Trains the Phase 3 Symbiotic Neural-Probabilistic Hybrid model.
Extracts priors from Phase 1 Baseline and jointly trains the Phase 2 DL embeddings + Gating mechanisms.
"""

import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import load_data, handle_missing_values, split_data
from src.train_dl import prepare_data as prepare_dl_data
from src.hybrid_model import ProbabilisticGatingFusionNet

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6

class HybridSalesDataset(Dataset):
    def __init__(self, cat_data: np.ndarray, num_data: np.ndarray, ml_priors: np.ndarray, labels: np.ndarray):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.ml_priors = torch.tensor(ml_priors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat_data[idx], self.num_data[idx], self.ml_priors[idx], self.labels[idx]

def prepare_hybrid_data():
    """Generates DL tensors AND extracts ML phase 1 priors."""
    # 1. Get Base Data
    df = load_data(DATA_DIR)
    df = handle_missing_values(df)
    X_train_full, X_test_full, y_train_full, y_test_full = split_data(df)
    
    # 2. Get DL Tensors (From Phase 2 preps)
    _, _, preps = prepare_dl_data()
    
    # Reprocess strictly for Hybrid
    numeric_cols = ["sales_cycle_duration_days", "revenue", "employees", "year_established"]
    cat_cols = ["sales_agent", "product", "sector", "office_location"]
    
    X_train_num = preps['num_scaler'].transform(X_train_full[numeric_cols])
    X_test_num = preps['num_scaler'].transform(X_test_full[numeric_cols])
    
    X_train_cat = preps['cat_encoder'].transform(X_train_full[cat_cols]) + 1
    X_test_cat = preps['cat_encoder'].transform(X_test_full[cat_cols]) + 1
    
    # 3. Get ML Priors (From Phase 1 Baseline)
    ml_model_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
    if not os.path.exists(ml_model_path):
        raise FileNotFoundError("Phase 1 baseline_model.pkl not found. Please run src/train.py first.")
        
    ml_pipeline = joblib.load(ml_model_path)
    
    # Predict probabilities for class 1 (Won)
    ml_prior_train = ml_pipeline.predict_proba(X_train_full)[:, 1].reshape(-1, 1)
    ml_prior_test = ml_pipeline.predict_proba(X_test_full)[:, 1].reshape(-1, 1)
    
    # Targets
    y_train = y_train_full.values.reshape(-1, 1).astype(np.float32)
    y_test = y_test_full.values.reshape(-1, 1).astype(np.float32)
    
    # Create Datasets
    train_dataset = HybridSalesDataset(X_train_cat, X_train_num, ml_prior_train, y_train)
    test_dataset = HybridSalesDataset(X_test_cat, X_test_num, ml_prior_test, y_test)
    
    return train_dataset, test_dataset, preps

def train_hybrid_model():
    train_dataset, test_dataset, preps = prepare_hybrid_data()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"\\n🚀 Using Device: {device} for Hybrid Training")
    
    # Initialize Hybrid Architecture
    model = ProbabilisticGatingFusionNet(
        embedding_sizes=preps['embedding_sizes'],
        num_numeric=preps['numeric_dim']
    ).to(device)
    
    class_ratio = (train_dataset.labels == 0).sum().item() / max(1, (train_dataset.labels == 1).sum().item())
    pos_weight = torch.tensor([class_ratio], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for cat_x, num_x, ml_priors, labels in train_loader:
            cat_x, num_x, ml_priors, labels = cat_x.to(device), num_x.to(device), ml_priors.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(cat_x, num_x, ml_priors)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * cat_x.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cat_x, num_x, ml_priors, labels in val_loader:
                cat_x, num_x, ml_priors, labels = cat_x.to(device), num_x.to(device), ml_priors.to(device), labels.to(device)
                logits = model(cat_x, num_x, ml_priors)
                loss = criterion(logits, labels)
                val_loss += loss.item() * cat_x.size(0)
                
        epoch_val_loss = val_loss / len(test_dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\\n🛑 Early Stopping at Epoch {epoch+1}")
                break
                
    model.load_state_dict(best_weights)
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(MODEL_DIR, "hybrid_model.pth"))
    
    # Plot Learning Curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.title("Learning Curves (Hybrid Mode)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "hybrid_learning_curves.png"))
    plt.close()
    
    print("✅ Hybrid Model Training Complete!")
    return model

if __name__ == "__main__":
    train_hybrid_model()
