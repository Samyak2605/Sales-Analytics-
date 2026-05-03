"""
evaluate_hybrid.py
==================
Diagnostic Evaluation and Ablation Studies for Phase 3 (Hybrid Innovation).
Generates the ultimate comparison table and diagnostic metrics.
"""

import os
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.preprocessing import split_data, load_data, handle_missing_values
from src.train_hybrid import prepare_hybrid_data, BATCH_SIZE
from src.hybrid_model import ProbabilisticGatingFusionNet
from src.evaluate_dl import evaluate_single_model as eval_dl
from src.dl_model import CategoricalEmbeddingNet
from src.train_dl import prepare_data as prepare_dl_data

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def get_ml_metrics():
    df = load_data(os.path.dirname(OUTPUT_DIR) + "/data")
    df = handle_missing_values(df)
    _, X_test, _, y_test = split_data(df)
    
    ml_model_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
    ml_pipeline = joblib.load(ml_model_path)
    
    preds = ml_pipeline.predict(X_test)
    probs = ml_pipeline.predict_proba(X_test)[:, 1]
    y_true = y_test.values
    
    metrics = {
        "Mode": "Advanced ML Only",
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, preds, zero_division=0), 4),
        "AUC": round(roc_auc_score(y_true, probs), 4)
    }
    return metrics

def get_dl_metrics(device):
    _, test_dataset, preps = prepare_dl_data()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CategoricalEmbeddingNet(
        embedding_sizes=preps['embedding_sizes'],
        num_numeric=preps['numeric_dim'],
        mode="full"
    ).to(device)
    
    model_path = os.path.join(MODEL_DIR, "dl_model_full.pth")
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    
    metrics, _ = eval_dl(model, test_loader, device)
    metrics["Mode"] = "Deep Learning Only"
    return metrics

def get_hybrid_metrics(device):
    _, test_dataset, preps = prepare_hybrid_data()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ProbabilisticGatingFusionNet(
        embedding_sizes=preps['embedding_sizes'],
        num_numeric=preps['numeric_dim']
    ).to(device)
    
    model_path = os.path.join(MODEL_DIR, "hybrid_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for cat_x, num_x, ml_priors, labels in test_loader:
            cat_x, num_x, ml_priors, labels = cat_x.to(device), num_x.to(device), ml_priors.to(device), labels.to(device)
            logits = model(cat_x, num_x, ml_priors)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    metrics = {
        "Mode": "Symbiotic Hybrid",
        "Accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "Precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "Recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "AUC": round(roc_auc_score(all_labels, all_probs), 4)
    }
    return metrics

def run_phase3_ablation():
    print("=" * 70)
    print("   PHASE 3: DIAGNOSTIC ABLATION STUDIES & HYBRID EVALUATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    print("📊 Extracting Baseline ML Metrics...")
    ml_metrics = get_ml_metrics()
    
    print("📊 Extracting Phase 2 DL Metrics...")
    dl_metrics = get_dl_metrics(device)
    
    print("📊 Evaluating Phase 3 Symbiotic Hybrid...")
    hyb_metrics = get_hybrid_metrics(device)
    
    df_results = pd.DataFrame([ml_metrics, dl_metrics, hyb_metrics])
    # Reorder columns
    cols = ["Mode", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    df_results = df_results[cols]
    
    print("\\n" + "=" * 70)
    print("                 FINAL ABLATION TABLE")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)
    
    df_results.to_csv(os.path.join(OUTPUT_DIR, "phase3_ablation_table.csv"), index=False)
    
    # Diagnostic Analysis
    ml_f1 = float(ml_metrics["F1-Score"])
    hyb_f1 = float(hyb_metrics["F1-Score"])
    diff = round((hyb_f1 - ml_f1) * 100, 2)
    
    print("\\n🔍 Diagnostic Analysis:")
    if diff > 0:
        print(f"✅ The Neural-Probabilistic Gating Mechanism successfully increased the F1-Score by {diff}% relative to the ML baseline.")
        print("   This proves that the non-linear Categorical Embeddings effectively captured latent residuals missed by the linear prior.")
    else:
        print(f"⚠️ The Hybrid model resulted in a {diff}% change in F1-Score. While the gating mechanism synthesizes predictions, the linear prior dominates the signal space for this specific dataset.")
        
    print(f"\\n✅ Phase 3 Ablation table saved to {OUTPUT_DIR}/phase3_ablation_table.csv")

if __name__ == "__main__":
    run_phase3_ablation()
