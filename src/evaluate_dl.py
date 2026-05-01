"""
evaluate_dl.py
==============
Rigorous evaluation and ablation studies for Phase 2 DL architectures on Tabular data.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.train_dl import train_model, prepare_data, BATCH_SIZE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_single_model(model, test_loader, device):
    """Evaluate a PyTorch model and return standard metrics."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    val_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for cat_x, num_x, labels in test_loader:
            cat_x, num_x, labels = cat_x.to(device), num_x.to(device), labels.to(device)
            logits = model(cat_x, num_x)
            
            loss = criterion(logits, labels)
            val_loss += loss.item() * cat_x.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / len(test_loader.dataset)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    metrics = {
        "Loss": round(avg_loss, 4),
        "Accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "Precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "Recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "AUC": round(roc_auc_score(all_labels, all_probs), 4)
    }
    
    cm = confusion_matrix(all_labels, all_preds)
    return metrics, cm

def run_ablation_study():
    """
    Train/Load the 3 variations (Ablation) and produce the comparison table.
    """
    print("=" * 60)
    print("  Deep Learning Technical Validation — Ablation Study")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    _, test_dataset, _ = prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    modes = ["numeric_only", "categorical_only", "full"]
    ablation_results = []
    confusion_matrices = {}
    
    for mode in modes:
        print(f"\\n⚙️ Preparing Model: {mode}")
        model = train_model(model_mode=mode)
        
        print(f"📊 Evaluating {mode}...")
        metrics, cm = evaluate_single_model(model, test_loader, device)
        metrics["Mode"] = mode
        ablation_results.append(metrics)
        confusion_matrices[mode] = cm
        
    df_results = pd.DataFrame(ablation_results)
    cols = ["Mode", "Loss", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    df_results = df_results[cols]
    
    print("\\n\\n" + "=" * 60)
    print("           ABLATION STUDY RESULTS (PHASE 2) ")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60)
    
    df_results.to_csv(os.path.join(OUTPUT_DIR, "ablation_table.csv"), index=False)
    
    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, mode in enumerate(modes):
        sns.heatmap(confusion_matrices[mode], annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['LOST', 'WON'], yticklabels=['LOST', 'WON'])
        axes[i].set_title(f"Confusion Matrix: {mode}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dl_ablation_confusion_matrices.png"))
    plt.close()
    
    print(f"✅ Ablation table & confusion matrices saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_ablation_study()
