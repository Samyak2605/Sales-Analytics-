"""
train_dl.py
===========
Deep Learning Training Pipeline for Entity-Embedding architectures (Phase 2).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import load_data, handle_missing_values, split_data
from src.dl_model import RelationalSalesDataset, CategoricalEmbeddingNet

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Data Preparation ───────────────────────────────────────────────────────

def prepare_data():
    df = load_data(DATA_DIR)
    df = handle_missing_values(df)
    
    # Stratified Split
    X_train_full, X_test_full, y_train_full, y_test_full = split_data(df)
    
    # Feature splits
    numeric_cols = ["sales_cycle_duration_days", "revenue", "employees", "year_established"]
    cat_cols = ["sales_agent", "product", "sector", "office_location"]
    
    # 1. Numeric Processor
    num_scaler = StandardScaler()
    X_train_num = num_scaler.fit_transform(X_train_full[numeric_cols])
    X_test_num = num_scaler.transform(X_test_full[numeric_cols])
    
    # 2. Categorical Encoders targeting Embeddings (Latent mapping)
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    # Fit and transform
    X_train_cat = cat_encoder.fit_transform(X_train_full[cat_cols])
    X_test_cat = cat_encoder.transform(X_test_full[cat_cols])
    
    # Fix the unknown values to be 0 and shift all other categories up by 1
    # This ensures no negative indices hit the PyTorch embedding layer.
    X_train_cat = X_train_cat + 1
    X_test_cat = X_test_cat + 1
    
    # Target conversion
    y_train = y_train_full.values.reshape(-1, 1).astype(np.float32)
    y_test = y_test_full.values.reshape(-1, 1).astype(np.float32)
    
    # 3. Create datasets
    train_dataset = RelationalSalesDataset(X_train_cat, X_train_num, y_train)
    test_dataset = RelationalSalesDataset(X_test_cat, X_test_num, y_test)
    
    # 4. Compute Embedding dims dynamically
    # For each category: Number of classes + 1 (for unknown)
    emb_sizes = []
    for i, col in enumerate(cat_cols):
        unique_classes = len(cat_encoder.categories_[i])
        vocab_size = unique_classes + 2  # +1 zero-index shifting, +1 padding margin
        # Empiric scaling rule of thumb for embeddings: min(50, vocab_size // 2)
        emb_dim = min(50, max(2, vocab_size // 2))
        emb_sizes.append((vocab_size, emb_dim))
        
    preprocessors = {
        'num_scaler': num_scaler,
        'cat_encoder': cat_encoder,
        'embedding_sizes': emb_sizes,
        'numeric_dim': len(numeric_cols)
    }
        
    return train_dataset, test_dataset, preprocessors

# ─── Train Loop ─────────────────────────────────────────────────────────────

def train_model(model_mode="full"):
    train_dataset, test_dataset, preps = prepare_data()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"\\n🚀 Using Device: {device}")
    
    # Initialize Architecture
    model = CategoricalEmbeddingNet(
        embedding_sizes=preps['embedding_sizes'],
        num_numeric=preps['numeric_dim'],
        mode=model_mode
    ).to(device)
    
    # Balancing minority class penalties natively
    class_ratio = (train_dataset.labels == 0).sum().item() / max(1, (train_dataset.labels == 1).sum().item())
    pos_weight = torch.tensor([class_ratio], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    print(f"\\n🔥 Starting Training (Mode: {model_mode})")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for cat_x, num_x, labels in train_loader:
            cat_x, num_x, labels = cat_x.to(device), num_x.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(cat_x, num_x)
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
            for cat_x, num_x, labels in val_loader:
                cat_x, num_x, labels = cat_x.to(device), num_x.to(device), labels.to(device)
                logits = model(cat_x, num_x)
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
    
    model_save_path = os.path.join(MODEL_DIR, f"dl_model_{model_mode}.pth")
    torch.save({'model_state_dict': model.state_dict()}, model_save_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.title(f"Learning Curves ({model_mode.capitalize()} Mode)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"dl_{model_mode}_learning_curves.png"))
    plt.close()
    
    return model

if __name__ == "__main__":
    train_model(model_mode="full")
