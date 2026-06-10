import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
from imblearn.metrics import geometric_mean_score
import time
import sys

# Configuration
PROCESSED_DIR = "processed_data"
RESULTS_DIR = "results"
RANDOM_STATE = 42
N_SPLITS = 3

def create_dirs():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def load_data():
    print("Loading preprocessed data...", flush=True)
    # Load RAW training data (not yet SMOTE-d) for correct CV
    X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train_raw.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train_raw.npy'))
    return X_train, y_train

def get_models():
    # 6 Models (No SVM)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1, solver='lbfgs'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        # KNN Removed due to excessive runtime on full dataset
        "Deep Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=RANDOM_STATE), 
        "XGBoost": XGBClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
    }
    return models

def calculate_metrics(y_true, y_pred, y_prob=None, n_classes=None):
    # Standard Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Imbalance-Aware Metrics
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    g_mean = geometric_mean_score(y_true, y_pred, average='macro') # macro for multiclass
    
    metrics = {
        "Accuracy": acc,
        "Precision (Macro)": prec_macro,
        "Precision (Weighted)": prec_weighted,
        "Recall (Macro)": rec_macro,
        "Recall (Weighted)": rec_weighted,
        "F1-score (Macro)": f1_macro,
        "F1-score (Weighted)": f1_weighted,
        "Balanced Accuracy": bal_acc,
        "MCC": mcc,
        "Cohen's Kappa": kappa,
        "G-Mean": g_mean
    }
    
    # ROC AUC (needs probabilities)
    if y_prob is not None:
        try:
            # Handle multiclass ROC AUC
            if n_classes > 2:
                auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                auc_weighted = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:
                # Binary case: y_prob usually shape (n_samples, 2), take col 1
                if y_prob.shape[1] == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob
                auc_macro = roc_auc_score(y_true, y_prob_pos)
                auc_weighted = auc_macro # Same for binary usually
                
            metrics["ROC-AUC (Macro)"] = auc_macro
            metrics["ROC-AUC (Weighted)"] = auc_weighted
        except Exception as e:
            print(f"ROC AUC failed: {e}")
            metrics["ROC-AUC (Macro)"] = 0.0
            metrics["ROC-AUC (Weighted)"] = 0.0
            
    return metrics

def train_and_evaluate():
    print("Loading Training data...", flush=True)
    X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train_raw.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train_raw.npy'))
    
    print("Loading Test data...", flush=True)
    X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
    
    n_classes = len(np.unique(y_train))
    print(f"Data loaded. Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
    
    models = get_models()
    
    results = [] 
    predictions_agg = {} 
    
    # Subsampling REMOVED as per request - using full sampled dataset
    # if len(X_train) > MAX_TRAIN_SAMPLES: ...

    # Apply SMOTE to Training
    print("Applying SMOTE to training data...", flush=True)
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"SMOTE complete. New Train Shape: {X_train_res.shape}", flush=True)
    except Exception as e:
        print(f"SMOTE failed: {e}. Using raw.", flush=True)
        X_train_res, y_train_res = X_train, y_train

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        start_time = time.time()
        
        # Train
        model.fit(X_train_res, y_train_res)
        
        # Predict on Train
        y_train_pred = model.predict(X_train_res)
        y_train_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_train_prob = model.predict_proba(X_train_res)
            except:
                pass
                
        # Calculate Train Metrics
        m_train = calculate_metrics(y_train_res, y_train_pred, y_train_prob, n_classes)
        m_train['Model'] = name
        m_train['Dataset'] = 'Train' # Distinguish Train vs Test
        m_train['Fold'] = 0
        results.append(m_train)

        # Save Model
        joblib.dump(model, os.path.join(RESULTS_DIR, f'{name.replace(" ", "_")}_model.pkl'))
        
        # Predict on Test
        y_test_pred = model.predict(X_test)
        y_test_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_test_prob = model.predict_proba(X_test)
            except:
                pass
        
        # Calculate Test Metrics
        m_test = calculate_metrics(y_test, y_test_pred, y_test_prob, n_classes)
        m_test['Model'] = name
        m_test['Dataset'] = 'Test'
        m_test['Fold'] = 0
        results.append(m_test)
        
        elapsed = time.time() - start_time
        print(f"  {name} completed in {elapsed:.2f}s", flush=True)
        
        # Store predictions
        predictions_agg[name] = {
            'Train': {
                'y_true': y_train_res,
                'y_pred': y_train_pred,
                'y_prob': y_train_prob
            },
            'Test': {
                'y_true': y_test,
                'y_pred': y_test_pred,
                'y_prob': y_test_prob
            }
        }
        
        # Save Partial Results
        pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'cv_results_detailed_partial.csv'), index=False)
        joblib.dump(predictions_agg, os.path.join(RESULTS_DIR, 'predictions_partial.pkl'))
        print(f"Partial results saved for {name}.", flush=True)

    # Save Final
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'cv_results_detailed.csv'), index=False)
    joblib.dump(predictions_agg, os.path.join(RESULTS_DIR, 'predictions.pkl'))
    print("Results Saved.", flush=True)

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'cv_results_detailed.csv'), index=False)
    
    # Calculate Mean +/- Std
    # Group by Model AND Dataset (Train/Test)
    summary = results_df.groupby(['Model', 'Dataset']).mean(numeric_only=True)
    summary_std = results_df.groupby(['Model', 'Dataset']).std(numeric_only=True)
    
    summary = summary.join(summary_std, lsuffix='_mean', rsuffix='_std')
    
    summary.to_csv(os.path.join(RESULTS_DIR, 'cv_results_summary.csv'))
    print("\nSummary Results Saved.", flush=True)
    
    # Save Predictions for Visualization (Using joblib for numpy arrays)
    joblib.dump(predictions_agg, os.path.join(RESULTS_DIR, 'predictions.pkl'))
    print("Predictions Saved.")

if __name__ == "__main__":
    create_dirs()
    train_and_evaluate()
