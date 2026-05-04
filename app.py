import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import os

from src.hybrid_model import ProbabilisticGatingFusionNet
from src.dl_model import CategoricalEmbeddingNet
from src.train_dl import prepare_data as prepare_dl_data
from src.preprocessing import load_data, handle_missing_values

st.set_page_config(page_title="Sales Prediction Engine", layout="wide")
st.title("🚀 AI Sales Analytics: Symbiotic Hybrid Engine")

# 1. Load Caches
@st.cache_resource
def load_models():
    # Load ML
    ml_model = joblib.load("models/baseline_model.pkl")
    
    # Load Preps (we need the encoders and sizes)
    _, _, preps = prepare_dl_data()
    
    device = torch.device('cpu')
    
    # Load DL
    dl_model = CategoricalEmbeddingNet(
        embedding_sizes=preps['embedding_sizes'],
        num_numeric=preps['numeric_dim'],
        mode="full"
    ).to(device)
    dl_model.load_state_dict(torch.load("models/dl_model_full.pth", map_location=device)['model_state_dict'])
    dl_model.eval()
    
    # Load Hybrid
    hybrid_model = ProbabilisticGatingFusionNet(
        embedding_sizes=preps['embedding_sizes'],
        num_numeric=preps['numeric_dim']
    ).to(device)
    hybrid_model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=device)['model_state_dict'])
    hybrid_model.eval()
    
    return ml_model, dl_model, hybrid_model, preps

@st.cache_data
def get_dropdown_options():
    df = load_data("data")
    df = handle_missing_values(df)
    return {
        "sales_agent": df['sales_agent'].unique().tolist(),
        "product": df['product'].unique().tolist(),
        "sector": df['sector'].unique().tolist(),
        "office_location": df['office_location'].unique().tolist()
    }

try:
    ml_model, dl_model, hybrid_model, preps = load_models()
    options = get_dropdown_options()
except Exception as e:
    st.error(f"Error loading models (Did you train Phase 1, 2, and 3?): {e}")
    st.stop()

# 2. UI Inputs
st.sidebar.header("Configure Deal Parameters")
agent = st.sidebar.selectbox("Sales Agent", options["sales_agent"])
product = st.sidebar.selectbox("Product", options["product"])
sector = st.sidebar.selectbox("Sector", options["sector"])
location = st.sidebar.selectbox("Office Location", options["office_location"])

duration = st.sidebar.slider("Sales Cycle Duration (Days)", 1, 365, 30)
revenue = st.sidebar.number_input("Client Revenue (Millions)", 1.0, 10000.0, 100.0)
employees = st.sidebar.slider("Client Employees", 1, 50000, 1000)
year_est = st.sidebar.slider("Year Established", 1800, 2024, 2000)

if st.sidebar.button("Predict Deal Outcome"):
    # 3. Format Data
    input_df = pd.DataFrame([{
        "sales_agent": agent, "product": product, "sector": sector, "office_location": location,
        "sales_cycle_duration_days": duration, "revenue": revenue, "employees": employees, "year_established": year_est
    }])
    
    # ML Prediction
    ml_prob = ml_model.predict_proba(input_df)[0, 1]
    
    # DL/Hybrid Formatting
    cat_cols = ["sales_agent", "product", "sector", "office_location"]
    num_cols = ["sales_cycle_duration_days", "revenue", "employees", "year_established"]
    
    # Handle unknown categories safely
    try:
        cat_arr = preps['cat_encoder'].transform(input_df[cat_cols]) + 1
    except:
        # Fallback for unknown categories
        cat_arr = np.ones((1, len(cat_cols))) 
        
    num_arr = preps['num_scaler'].transform(input_df[num_cols])
    
    cat_t = torch.tensor(cat_arr, dtype=torch.long)
    num_t = torch.tensor(num_arr, dtype=torch.float32)
    ml_t = torch.tensor([[ml_prob]], dtype=torch.float32)
    
    with torch.no_grad():
        dl_logit = dl_model(cat_t, num_t)
        dl_prob = torch.sigmoid(dl_logit).item()
        
        hybrid_logit = hybrid_model(cat_t, num_t, ml_t)
        hybrid_prob = torch.sigmoid(hybrid_logit).item()
        
    # 4. Display Results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Phase 1: ML Prior", f"{ml_prob*100:.1f}%")
    col2.metric("Phase 2: DL Latent", f"{dl_prob*100:.1f}%")
    col3.metric("Phase 3: Symbiotic Hybrid", f"{hybrid_prob*100:.1f}%", 
                f"{(hybrid_prob - ml_prob)*100:.1f}% vs ML")
                
    st.write("---")
    st.markdown("""
    ### Interpretability
    The **Hybrid Model** uses a differential gating mechanism. It analyzes the raw ML probability against the DL latent embeddings to dynamically adjust its final confidence. Notice how the Hybrid Prediction synthesizes the two baseline signals depending on the specific product and agent combination.
    """)
