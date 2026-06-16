import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import SAVE_DIR

sns.set_theme(style="whitegrid", context="paper")

def combine_and_plot():
    f1 = os.path.join(SAVE_DIR, "bilstm_metrics.csv")
    f2 = os.path.join(SAVE_DIR, "cnn_bilstm_attn_metrics.csv")
    
    dfs = []
    if os.path.exists(f1): dfs.append(pd.read_csv(f1))
    if os.path.exists(f2): dfs.append(pd.read_csv(f2))
    
    if not dfs:
        print("No metrics found. Please train models first.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(SAVE_DIR, "model_comparison.csv"), index=False)
    
    # Dynamically select the best model's artifacts to power the dashboard (Step 11 & 9 logic)
    best_model = df.loc[df['Accuracy'].idxmax()]['Model']
    best_prefix = "bilstm" if best_model == "BiLSTM" else "cnn_bilstm_attention"
    
    artifacts = [
        ("accuracy_plot.png", "accuracy_plot.png"),
        ("loss_plot.png", "loss_plot.png"),
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("classification_report.txt", "classification_report.txt"),
        ("predictions.csv", "predictions.csv")
    ]
    
    for src_suf, dst in artifacts:
        src = os.path.join(SAVE_DIR, f"{best_prefix}_{src_suf}")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SAVE_DIR, dst))
            
    # Generate Comparison Plot
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Performance Comparison: Traditional vs. Hybrid Architecture", fontweight="bold")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "model_comparison_plot.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    combine_and_plot()