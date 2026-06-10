import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import os
import matplotlib.pyplot as plt

# Configuration
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

def check_significance():
    path = os.path.join(RESULTS_DIR, 'cv_results_detailed.csv')
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, 'cv_results_detailed_partial.csv')
        
    if not os.path.exists(path):
        print("No results file found.")
        return

    df = pd.read_csv(path)
    
    # Filter for Test set results only
    if 'Dataset' in df.columns:
        df = df[df['Dataset'] == 'Test']
    
    # Pivot to get: Index=Fold, Columns=Model, Values=Metric
    # We perform Friedman on F1-score (Weighted) usually
    metric = 'F1-score (Weighted)'
    
    pivot_df = df.pivot(index='Fold', columns='Model', values=metric)
    print(f"Data for Statistical Test ({metric}):\n", pivot_df)
    
    # Friedman Test
    # Tests if there is a difference between models across folds
    stat, p = stats.friedmanchisquare(*[pivot_df[col].values for col in pivot_df.columns])
    print(f"\nFriedman Test: Statistic={stat:.4f}, p-value={p:.4e}")
    
    results_txt = f"Friedman Test for {metric}:\nStatistic={stat:.4f}, p-value={p:.4e}\n\n"
    
    alpha = 0.05
    if p < alpha:
        print("Significant difference found (p < 0.05). Proceeding to Post-hoc tests.")
        results_txt += "Significant difference found. Post-hoc tests (Nemenyi):\n"
        
        # Nemenyi Test
        # Takes the pivoted data directly
        nemenyi = sp.posthoc_nemenyi_friedman(pivot_df)
        print("\nNemenyi Post-hoc p-values:\n", nemenyi)
        results_txt += str(nemenyi) + "\n\n"
        
        # Save Nemenyi heatmap
        plt.figure(figsize=(10, 8))
        import seaborn as sns
        sns.heatmap(nemenyi, annot=True, cmap='RdYlGn_r', vmin=0, vmax=0.05)
        plt.title('Nemenyi Test P-values (Green = Significant Difference)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'critical_difference_heatmap.png'), dpi=300)
        plt.close()
        
        # Wilcoxon Signed-Rank Test (Pairwise) - comparing best vs others
        # Find best model first (highest mean)
        means = pivot_df.mean()
        best_model = means.idxmax()
        results_txt += f"Best Model based on Mean {metric}: {best_model}\n"
        results_txt += "Pairwise Wilcoxon vs Best Model:\n"
        
        for model in pivot_df.columns:
            if model == best_model:
                continue
            w_stat, w_p = stats.wilcoxon(pivot_df[best_model], pivot_df[model])
            res_str = f"{best_model} vs {model}: p={w_p:.4e}"
            print(res_str)
            results_txt += res_str + "\n"
            
    else:
        print("No significant difference found.")
        results_txt += "No significant difference found per Friedman test."

    with open(os.path.join(RESULTS_DIR, 'statistical_results.txt'), 'w') as f:
        f.write(results_txt)
    print("Statistical results saved.")

if __name__ == "__main__":
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    check_significance()
