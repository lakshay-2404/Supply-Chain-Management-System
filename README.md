
# Inventory Management Model

A deep learning–driven inventory optimization system that forecasts product-level demand, minimizes stockouts and overstock, and adapts rapidly to new businesses via transfer learning.

## Highlights
- Demand forecasting with a multilayer perceptron (MLP) and classic baselines (RF, Linear/Ridge/Lasso/ElasticNet, KNN) for comparative benchmarking.  
- Transfer learning to fine‑tune on company‑specific data, accelerating adaptation and improving accuracy on new catalogs and stores.  
- End‑to‑end workflow: data prep, EDA, model development, hyperparameter optimization, evaluation, and production‑ready checkpoints.  

## Objectives
- Build a deep‑learning demand forecaster as a reusable starting point for inventory optimization.  
- Train on historical sales, prices, stock levels, and related signals; compare against strong baseline models.  
- Enable fast customization to a target company via transfer learning and domain features.  

## Directory structure
```
inventory-management-model/
├─ checkpoint/
│  ├─ base_model_checkpoint.h5
│  └─ transfert_learning_model_checkpoint.h5
├─ data/
│  ├─ 2017PurchasePricesDec.csv
│  ├─ BegInvFinal1232016.csv
│  ├─ EndInvFinal1232016.csv
│  ├─ InvoicePurchases12312016.csv
│  ├─ PurchasesFinal12312016.csv
│  ├─ SalesFinal12312016.csv
│  ├─ preprocessing/
│  └─ transfertLearning/
├─ notebooks/
│  ├─ data_exploration.ipynb
│  ├─ data_preprocessing.ipynb
│  ├─ stock_management_model_development.ipynb
│  ├─ stock_management_model_evaluation.ipynb
│  ├─ reference_models_training.ipynb
│  └─ transfert_learning.ipynb
├─ src/
│  ├─ data_preprocessing_utils.py
│  ├─ model.py
│  └─ utils.py
├─ images/
│  ├─ performance_comparison.png
│  └─ comparaison_trues_predict_values.png
├─ requirements.txt
└─ README.md
```

## Steps to run

### 1) Create environment and install dependencies
- Using conda:
```
conda config --add channels conda-forge
conda create -n invenv --file requirements.txt
conda activate invenv
pip install tensorflow==2.4 prettytable ipykernel typing-extensions==3.7.4
```

- Or with plain pip (Python 3.8+):
```
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
pip install tensorflow==2.4 prettytable ipykernel typing-extensions==3.7.4
```

### 2) Prepare data
- Place raw CSVs in data/.  
- Run notebooks/data_preprocessing.ipynb to generate cleaned files into data/preprocessing/.  
- Optionally populate data/transfertLearning/ with target company data for TL.

### 3) Train baseline and MLP model
- Run notebooks/reference_models_training.ipynb to train/record baseline metrics.  
- Run notebooks/stock_management_model_development.ipynb for MLP design, hyperparameter search, and training; best weights saved to checkpoint/.  

### 4) Evaluate
- Open notebooks/stock_management_model_evaluation.ipynb to compute RMSE/MAE/MAPE/R2 and compare against baselines; export plots to images/.

### 5) Transfer learning (optional)
- Use notebooks/transfert_learning.ipynb to adapt the base model to a specific company; save best TL weights to checkpoint/.

### 6) (Optional) Serve
- Package the trained model as an API (e.g., FastAPI/Flask) or build a planner UI (e.g., Streamlit) using src/model.py and saved checkpoints.

## Workflow

### Part 1 — Base model
- Data: multi-store, multi-product sales, inventories, invoices, and attributes (e.g., 79 stores, ~6.9k SKUs, ~2 months).  
- Preprocessing (notebooks/data_preprocessing.ipynb): cleaning, missing value handling, outlier treatment, exports to data/preprocessing/.  
- EDA (notebooks/data_exploration.ipynb): time trends, seasonality, relationships, drivers.  
- Baselines (reference_models_training.ipynb): RF, Linear, Ridge, Lasso, ElasticNet, KNN.  
- MLP model (stock_management_model_development.ipynb):  
  - Data formatting; initial model; Bayesian HPO; retrain best config; save checkpoints.  
- Evaluation (stock_management_model_evaluation.ipynb):  
  - Load weights; predict; compute RMSE, MAE, MAPE, R2, RRSE, RAE; compare to baselines.

Example results (illustrative):  
- Strong correlation (≈0.92), lower RMSE/MAE vs. baselines, materially better RRSE/RAE; MAPE may remain high on sparse or low-volume SKUs.  

### Part 2 — Transfer learning case study
- Use target company data (data/transfertLearning/), e.g., Favorita.  
- Preprocess, split train/test, adapt the base model with company-specific layers/features.  
- Fine‑tune and evaluate: often large gains in correlation/R2 and reductions in MAE/RMSE thanks to domain adaptation.  
- Visualize actual vs. predicted demand curves for key families/SKUs.

## Going further
- Serve as an API for real‑time replenishment, reorder point tuning, and buy-plans.  
- Build a lightweight Flask/Streamlit UI for planners.  
- Add exogenous signals (promotions, holidays, weather, macro).  
- Automate continuous training with data drift checks and scheduled re‑fits.

## License
GPL‑3.0. See LICENSE.
