import shap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Configuration
DATASET_PATH = r"c:/Users/Nahid/Desktop/2/CICIDS2017 (1).csv"
PROCESSED_DIR = "processed_data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
MODEL_PATH = os.path.join(RESULTS_DIR, "XGBoost_model.pkl")
SAMPLE_SIZE = 1000  # Number of samples for SHAP analysis (SHAP is slow on large data)

def get_feature_names():
    print(f"Reading feature names from {DATASET_PATH}...")
    # Read only the header
    df_header = pd.read_csv(DATASET_PATH, nrows=0)
    # Strip whitespace
    df_header.columns = df_header.columns.str.strip()
    
    # Drop target column 'Attack Type'
    if 'Attack Type' in df_header.columns:
        features = df_header.drop(columns=['Attack Type']).columns.tolist()
    else:
        # Fallback if column name is different or missing
        print("Warning: 'Attack Type' column not found in header. Using all columns except last.")
        features = df_header.columns[:-1].tolist()
        
    print(f"Found {len(features)} features: {features[:5]}...")
    return features

def main():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    # 1. Load Data
    print("Loading Test Data...")
    X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
    print(f"X_test shape: {X_test.shape}")

    # 2. Sample Data
    # SHAP TreeExplainer is relatively fast, but for 100k+ rows it takes time. 
    # 1000 samples is usually sufficient for global explanation.
    if X_test.shape[0] > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} instances from Test set...")
        # Use a fixed seed for reproducibility
        indices = np.random.RandomState(42).choice(X_test.shape[0], SAMPLE_SIZE, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test

    # 3. Load Model
    print(f"Loading XGBoost model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 4. Get Feature Names
    feature_names = get_feature_names()
    
    # Verify feature count matches
    if len(feature_names) != X_sample.shape[1]:
        print(f"Warning: Feature count mismatch! Model expected {X_sample.shape[1]}, found {len(feature_names)} names.")
        # Fallback to generic names
        feature_names = [f"Feature {i}" for i in range(X_sample.shape[1])]

    # 5. Compute SHAP Values
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multiclass structures (List or 3D Array)
    if isinstance(shap_values, list):
        print(f"SHAP values is a list of length {len(shap_values)}")
        # Use class 1 (usually first attack class) if available, else class 0
        target_class = 1 if len(shap_values) > 1 else 0
        shap_values_plot = shap_values[target_class]
        shap_values_summary = shap_values # Summary plot can take the list
    elif len(shap_values.shape) == 3:
        print(f"SHAP values is 3D: {shap_values.shape}")
        # Try to identify which dimension is Classes (usually the last one for XGBoost)
        if shap_values.shape[2] == 7:
            target_class = 1
            shap_values_plot = shap_values[:, :, target_class]
            # Summary plot expects list of arrays for multiclass summary
            shap_values_summary = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        else:
            # Fallback
            shap_values_plot = shap_values[0] if shap_values.shape[0] == 7 else shap_values[:, :, 0]
            shap_values_summary = shap_values
    else:
        print(f"SHAP values is 2D: {shap_values.shape}")
        shap_values_plot = shap_values
        shap_values_summary = shap_values

    print(f"X_sample shape: {X_sample.shape}")
    print("SHAP values prepared. Generating plots...")

    # 6. Generate Plots
    
    # Plot 1: Summary Plot (Dot) - Use single class for beeswarm impact
    plt.figure(figsize=(14, 10))
    # Note: Use shap_values_plot (2D) for dot plot to show feature impact for target class
    shap.summary_plot(shap_values_plot, X_sample, feature_names=feature_names, plot_type="dot", show=False)
    plt.title(f"SHAP Summary (Dot) - Class {target_class}", fontsize=20, pad=25)
    plt.gcf().subplots_adjust(left=0.3, bottom=0.15)
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_dot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_summary_dot.png")

    # Plot 2: Summary Plot (Bar) - Use multiclass for global importance
    plt.figure(figsize=(14, 10))
    # Note: Use shap_values_summary (List) for stacked bar plot showing importance across all classes
    shap.summary_plot(shap_values_summary, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar) - All Classes", fontsize=20, pad=25)
    # Adjust margins to prevent label cropping (especially the x-axis "mean(|SHAP value|)")
    plt.gcf().subplots_adjust(left=0.3, bottom=0.15)
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_summary_bar.png")

    # Plot 3: Dependence Plot
    # Calculate feature importance from the selected class slice
    vals = np.abs(shap_values_plot).mean(axis=0)
    top_feature_idx = np.argmax(vals)
    top_feature_name = feature_names[top_feature_idx]
    print(f"Top feature for dependence plot: {top_feature_name}")

    plt.figure(figsize=(12, 8))
    # Pass ONLY the 2D slice to dependence_plot
    shap.dependence_plot(top_feature_idx, shap_values_plot, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Dependence: {top_feature_name} (Class {target_class})", fontsize=18, pad=25)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(FIGURES_DIR, "shap_dependence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_dependence.png")
    
    print("SHAP analysis complete.")

if __name__ == "__main__":
    main()
