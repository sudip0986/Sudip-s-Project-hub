import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os
import joblib

# Configuration
DATASET_PATH = r"c:/Users/Nahid/Desktop/2/CICIDS2017 (1).csv"
PROCESSED_DIR = "processed_data"
FIGURES_DIR = "figures"
SAMPLE_FRACTION = 0.20
RANDOM_STATE = 42

def create_dirs():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

def load_and_sample_data(path, fraction):
    print(f"Loading dataset from {path}...")
    # Load only a sample if dataset is huge, but here we can load all then sample to ensure stratify if possible
    # For efficiency with large CSVs, we might want to skip rows, but to respect stratification we usually need at least class labels.
    # Given the requirements: "Use only 20% of dataset (Try to use it somewhat balanced not exactly while using the 20 % of the dataset that will be used for the models)"
    
    # Let's load the full dataset first as 700MB is manageable in memory for typical modern machines
    try:
        df = pd.read_csv(path)
        print(f"Original shape: {df.shape}")
        
        # Strip column names just in case
        df.columns = df.columns.str.strip()
        
        # Custom Sampling Strategy: Keep all minority attacks, downsample Normal Traffic to reach 20% total
        target_size = int(len(df) * fraction)
        print(f"Target sample size (approx {fraction*100}%): {target_size}")
        
        # Identify Normal and Attack classes
        # Assuming 'Normal Traffic' is the majority class label.
        # Let's count classes first
        counts = df['Attack Type'].value_counts()
        print("Original Class Counts:\n", counts)
        
        # We need to handle potential whitespace in labels if not cleaned yet
        normal_label = 'Normal Traffic' # Based on previous view_file output
        
        df_normal = df[df['Attack Type'] == normal_label]
        df_attack = df[df['Attack Type'] != normal_label]
        
        n_attacks = len(df_attack)
        print(f"Total Attack Samples: {n_attacks}")
        
        if n_attacks >= target_size:
            print("Warning: Attack samples alone exceed the target 20% size. Sampling stratified from all.")
            df_sampled, _ = train_test_split(df, train_size=target_size, stratify=df['Attack Type'], random_state=RANDOM_STATE)
        else:
            # Take all attacks
            # Fill remainder with Normal
            n_normal_needed = target_size - n_attacks
            print(f"Taking all {n_attacks} attacks and sampling {n_normal_needed} normal traffic.")
            
            if n_normal_needed > len(df_normal):
                 # Should not happen given the premise, but just in case
                 n_normal_needed = len(df_normal)
            
            df_normal_sampled = df_normal.sample(n=n_normal_needed, random_state=RANDOM_STATE)
            df_sampled = pd.concat([df_attack, df_normal_sampled])
            
            # Shuffle
            df_sampled = df_sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
            
        print(f"Sampled shape: {df_sampled.shape}")
        print("Sampled Class Counts:\n", df_sampled['Attack Type'].value_counts())
        return df_sampled
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    print("Cleaning data...")
    # Drop duplicates
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")
    
    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"Shape after removing NaNs/Infs: {df.shape}")
    
    return df

def plot_class_distribution(y, title, filename):
    plt.figure(figsize=(10, 6))
    counts = y.value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close()
    print(f"Saved distribution plot to {os.path.join(FIGURES_DIR, filename)}")

def preprocess_and_balance(df):
    target_col = 'Attack Type'
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the encoder mapping for later reference
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Class Mapping: {mapping}")
    
    # Split into Train/Test (Test set should remain 'pure' - no SMOTE)
    # We will use Cross-Validation on Train, but keeping a hold-out test set is good practice.
    # However, the prompt asks for Cross Validation (k=3). We can do CV on the whole processed set or just train.
    # Standard practice: Split Train/Test. Apply SMOTE only on Train. Evaluate on Test.
    # The prompt also says "Use Stratified K-Fold Cross Validation (k=3)".
    # To support the "Model Training" phase requirements, we will prepare X and y that are cleaned.
    # balancing (SMOTE) is usually done *inside* the Cross-Validation loop to prevent data leakage.
    # BUT, Requirement 1 says: "Apply class balancing techniques: SMOTE... Plot class distribution before and after balancing".
    # This implies doing it globally or establishing a balanced Dataset for training.
    # Given "2. Cross Validation... Ensure no data leakage", strict CV requires SMOTE inside the fold.
    # I will split 80/20 Train/Test.
    # I will apply SMOTE to the *Training* split only, and save THAT as the training set to be used for CV training loops?
    # Actually, if I perform SMOTE *before* CV splits, I leak synthetic data into validation folds. 
    # Best Practice: Prepare cleaned X, y. In `model_training.py`, during CV, we apply SMOTE to the training fold.
    # HOWEVER, the prompt lists "Data Preprocessing" as a distinct Phase 1 with SMOTE and visual requirements.
    # I will perform a split: X_train, X_test, y_train, y_test.
    # I will apply SMOTE to X_train, y_train to create X_train_balanced, y_train_balanced.
    # The models can then use X_train_balanced for training (or CV on it, though CV on already upsampled data is debatable, usually one pipelines SMOTE).
    # Let's save both the raw split and the balanced version of the training set.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE)
    
    # Scale Features (Fit on Train, Transform Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Keep for final hold-out if needed
    
    # Plot Before
    plot_class_distribution(pd.Series(le.inverse_transform(y_train)), "Class Distribution (Before SMOTE)", "class_dist_before.png")
    
    # Apply SMOTE to Training Data
    print("Applying SMOTE...")
    # Strategy: 'not majority' resamples all classes except the majority class
    # or 'auto'. 'auto' is equivalent to 'not majority'.
    # We might want to limit the upsampling if some classes are tiny to avoid massive synthetic noise, but 'auto' is standard request.
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        print(f"SMOTE applied. New training shape: {X_train_res.shape}")
        
        # Plot After
        plot_class_distribution(pd.Series(le.inverse_transform(y_train_res)), "Class Distribution (After SMOTE)", "class_dist_after.png")
        
    except Exception as e:
        print(f"SMOTE failed (possibly k_neighbors error for extremely rare classes): {e}")
        # Fallback: Just use scaled data
        X_train_res, y_train_res = X_train_scaled, y_train

    # Save Artifacts
    print("Saving processed data...")
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))
    
    # Save arrays
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train_res)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train_res)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    
    # Also save the original imbalanced training set in case we want to use pipeline SMOTE in CV
    np.save(os.path.join(PROCESSED_DIR, 'X_train_raw.npy'), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, 'y_train_raw.npy'), y_train)

    print("Data preprocessing complete.")

if __name__ == "__main__":
    create_dirs()
    df = load_and_sample_data(DATASET_PATH, SAMPLE_FRACTION)
    if df is not None:
        df = clean_data(df)
        preprocess_and_balance(df)
