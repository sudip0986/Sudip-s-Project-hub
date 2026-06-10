import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Configuration
RESULTS_DIR = "results"
PROCESSED_DIR = "processed_data"
FIGURES_DIR = "figures"

def create_dirs():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

def load_data():
    path = os.path.join(RESULTS_DIR, 'predictions.pkl')
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, 'predictions_partial.pkl')
        print(f"Loading partial predictions from {path}")
        
    predictions = joblib.load(path)
    le = joblib.load(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))
    return predictions, le

def plot_confusion_matrices(predictions, le):
    classes = le.classes_
    n_models = len(predictions)
    
    # Generate TWO separate files: one for Train, one for Test
    # Each file contains a grid of all models
    
    # Determine grid size
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    for dataset in ['Train', 'Test']:
        fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
        axes = axes.flatten()
        
        for i, (name, sets) in enumerate(predictions.items()):
            data = sets[dataset]
            y_true = data['y_true']
            y_pred = data['y_pred']
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                        xticklabels=classes, yticklabels=classes)
            axes[i].set_title(f'{name} ({dataset})')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
            axes[i].tick_params(axis='x', rotation=45)
            
        # Hide unused
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'confusion_matrices_{dataset}.png'), dpi=300)
        plt.close()
    print("Saved Confusion Matrices (Train & Test separate).")

def plot_roc_curves(predictions, le):
    # Plot Macro-Average ROC for all models 
    # One plot for Train, One for Test
    
    classes = le.classes_
    n_classes = len(classes)
    
    for dataset in ['Train', 'Test']:
        plt.figure(figsize=(12, 8))
        
        for name, sets in predictions.items():
            data = sets[dataset]
            y_true = data['y_true']
            y_prob = data['y_prob']
            
            if y_prob is None:
                continue 
                
            # Binarize y_true
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Compute Macro ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # Calculate for each class
            for i in range(n_classes):
                if np.sum(y_true_bin[:, i]) > 0:
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Aggregate all FPRs
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                if i in fpr:
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
            plt.plot(fpr["macro"], tpr["macro"], label=f'{name} (area = {roc_auc["macro"]:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{dataset} set: Receiver Operating Characteristic (Macro-Average)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(FIGURES_DIR, f'roc_curves_{dataset}.png'), dpi=300)
        plt.close()
    print("Saved ROC Comparisons.")

def plot_model_comparison_bar(results_df):
    # Plot Weighted F1 and Accuracy
    # Expects columns like 'F1-score (Weighted)', 'Accuracy', 'Model', 'Dataset'
    
    metrics_to_plot = ['F1-score (Weighted)', 'Accuracy', 'Balanced Accuracy']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        # Hue by Dataset to show Train vs Test
        sns.barplot(data=results_df, x='Model', y=metric, hue='Dataset', palette='viridis', ci=None)
        plt.title(f'Model Comparision: {metric} (Train vs Test)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'bar_{metric.replace(" ","_").replace("(","").replace(")","")}.png'), dpi=300)
        plt.close()
    print("Saved Bar Comparisons.")

def plot_radar_chart(results_df):
    # Normalize metrics to 0-1 range for radar if scales differ drastically, but usually they are 0-1.
    # Group by Model AND Dataset
    summary = results_df.groupby(['Model', 'Dataset']).mean()
    
    # Select subset of metrics
    metrics = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-score (Weighted)', 'Balanced Accuracy', 'MCC']
    
    # Check if all exist
    available_metrics = [m for m in metrics if m in summary.columns]
    
    if len(available_metrics) < 3:
        return

    # Prepare data for Radar
    labels = available_metrics
    num_vars = len(labels)
    
    # Angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += [angles[0]] # Close the loop
    
    # Create one plot per Model to compare its Train vs Test
    models = results_df['Model'].unique()
    
    for model in models:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for dataset in ['Train', 'Test']:
            if (model, dataset) in summary.index:
                values = summary.loc[(model, dataset), available_metrics].tolist()
                values += [values[0]]
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'{dataset}')
                ax.fill(angles, values, alpha=0.1)
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f"Performance Radar: {model}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'radar_plot_{model.replace(" ","_")}.png'), dpi=300)
        plt.close()
    print("Saved Radar Charts.")

def plot_shap_analysis(results_df, le):
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping.")
        return

    # 1. Identify Best Model (by Test F1-Weighted)
    # Check if 'Dataset' column exists (it should now)
    if 'Dataset' in results_df.columns:
        best_row = results_df[results_df['Dataset'] == 'Test'].sort_values(by='F1-score (Weighted)', ascending=False).iloc[0]
    else:
        # Fallback if old results
        best_row = results_df.sort_values(by='F1-score (Weighted)', ascending=False).iloc[0]
        
    best_model_name = best_row['Model']
    print(f"Best Model for SHAP: {best_model_name}")
    
    # 2. Load Model
    model_path = os.path.join(RESULTS_DIR, f'{best_model_name.replace(" ", "_")}_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
        
    model = joblib.load(model_path)
    
    # 3. Load Sample Data (Test Set)
    X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
    # SHAP is slow, take a small background sample
    background_sample = X_test[:100] 
    eval_sample = X_test[:200]
    
    # Determine Explainer
    # Tree models
    tree_models = ['Random Forest', 'XGBoost', 'LightGBM', 'Decision Tree']
    if best_model_name in tree_models:
        # Check if shap_values is list (multiclass)
        # TreeExplainer for multiclass returns list of matrices
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(eval_sample)
        except:
             # Fallback
             try:
                 explainer = shap.Explainer(model)
                 shap_values = explainer(eval_sample).values
             except:
                 print("SHAP TreeExplainer failed.")
                 return
    else:
        # Linear for LR
        if "Logistic" in best_model_name or "Linear" in best_model_name:
             explainer = shap.LinearExplainer(model, background_sample)
             shap_values = explainer.shap_values(eval_sample)
        else:
             print("Using KernelExplainer for SHAP (might be slow)...")
             # MLP or KNN needs KernelExplainer
             # We use a summary function (predict_proba)
             # KernelExplainer needs a function that takes numpy array
             f = lambda x: model.predict_proba(x)
             explainer = shap.KernelExplainer(f, background_sample)
             shap_values = explainer.shap_values(eval_sample)

    # 4. Plots
    # Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, eval_sample, feature_names=[f'Feat_{i}' for i in range(X_test.shape[1])], show=False)
    plt.title(f'SHAP Summary: {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'), dpi=300)
    plt.close()
    
    # Bar Plot (Feature Importance)
    plt.figure()
    shap.summary_plot(shap_values, eval_sample, feature_names=[f'Feat_{i}' for i in range(X_test.shape[1])], plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance: {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_bar.png'), dpi=300)
    plt.close()
    print("Saved SHAP Plots.")

def main():
    create_dirs()
    
    # Load Predictions
    if os.path.exists(os.path.join(RESULTS_DIR, 'predictions.pkl')) or os.path.exists(os.path.join(RESULTS_DIR, 'predictions_partial.pkl')):
        predictions, le = load_data()
        plot_confusion_matrices(predictions, le)
        plot_roc_curves(predictions, le)
    else:
        print("Predictions file not found.")
        
    # Load Results CSV
    csv_path = os.path.join(RESULTS_DIR, 'cv_results_detailed.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(RESULTS_DIR, 'cv_results_detailed_partial.csv')

    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        plot_model_comparison_bar(results_df)
        plot_radar_chart(results_df)
        # Call SHAP analysis only if predictions exist (need label encoder for logic? No, passed le here)
        # Actually plot_shap_analysis needs le? No, but arguments matched
        plot_shap_analysis(results_df, le)
        
        # Save boxplots of folds
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x='Model', y='F1-score (Weighted)')
        plt.title('Cross-Validation Score Distribution (F1 Weighted)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'boxplot_cv_scores.png'), dpi=300)
        plt.close()
        
    else:
        print("Detailed results CSV not found.")

if __name__ == "__main__":
    main()
