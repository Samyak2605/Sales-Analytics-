# Predictive Sales Analytics Engine

An end-to-end Machine Learning and Deep Learning pipeline designed to forecast business opportunity win probabilities (Won vs. Lost outcomes) using multi-relational CRM data (`sales_pipeline`, `accounts`, etc).

## System Architecture

The project leverages a three-phase approach, culminating in a state-of-the-art Differentiable Programming hybrid network.

### Phase 1: Advanced Machine Learning
Builds a highly interpretable and robust Baseline model via Scikit-Learn:
- **`src/preprocessing.py`**: Left-joins `sales_pipeline` datasets with `accounts`, extracts sales cycle time-series components, handles missing categorizations robustly, and partitions subsets natively preserving exact deal-class structures (stratified selection).
- **`src/feature_engineering.py`**: Constructs mathematically precise `StandardScaler` transformations and expands high-cardinal categorical bounds into wide-sparse binaries via `OneHotEncoder`.
- **`src/train.py` & `evaluate.py`**: Compiles linear predictors (Logistic Regression) offering extremely explainable business boundaries and explicit class probability intersections.

### Phase 2: Deep Learning Neural Pipelines
Re-evaluates the mathematical paradigm to utilize high-density Deep Tabular parameters mapping PyTorch dynamics:
- **`src/dl_model.py` (Tabular Embedding Net)**: Bypasses static OneHot arrays by dynamically loading high-cardinal constraints into `nn.Embedding(size, dims)` components inside PyTorch. This forces models to compute Latent Dimensional semantic structures effectively evaluating correlations. 
- **`src/train_dl.py` & `evaluate_dl.py`**: Trains the networks securely incorporating `BCEWithLogitsLoss` and computes categorical vs numerical ablation bounds natively.

### Phase 3: Symbiotic Hybrid Network
Achieves true Neuro-Symbolic integration instead of relying on naive ensembling:
- **`src/hybrid_model.py` (Probabilistic Gating Network)**: Operates a differentiable attention gate that dynamically weights the Phase 1 Logistic Regression explicit prior probability against the Phase 2 deep latent residuals based on input characteristics.
- **`src/train_hybrid.py` & `evaluate_hybrid.py`**: Trains the gating mechanism mapping explicitly over both architectures simultaneously, proving diagnostic performance boosts mathematically.

---

## Reproducibility & Extra Mile Deliverables
This repository comes fully equipped with "Turn-Key" evaluation environments:
- **`Dockerfile` & `docker-compose.yml`**: Full OS-independent container execution capabilities.
- **`setup.sh`**: One-click local bash script that sets up the pure python virtual environment and installs configurations safely.
- **`app.py` (Streamlit Web Dashboard)**: An interactive web application that allows users to instantly test deals and compare the isolated Phase 1, Phase 2, and Phase 3 Hybrid models interactively side-by-side.

## Commands

To perform Advanced Machine learning natively:
```bash
python src/train.py
```

To execute Deep Neural Ablation natively:
```bash
python src/evaluate_dl.py
```

To train and evaluate the Phase 3 Symbiotic Hybrid architecture:
```bash
python src/train_hybrid.py
python src/evaluate_hybrid.py
```

To run the interactive UI Web Dashboard:
```bash
streamlit run app.py
```
