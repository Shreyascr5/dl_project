import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve)
from src.config import SAVE_DIR, CLASSES

sns.set_theme(style="whitegrid")

def evaluate_and_save(model_name, model, X_test, y_test, history=None, train_time=0):
    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    
    # Save Report
    report = classification_report(y_test, preds, target_names=CLASSES, zero_division=0)
    with open(os.path.join(SAVE_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(report)
        
    # Save Predictions CSV
    pd.DataFrame({"Actual": y_test, "Predicted": preds, "Confidence": np.max(probs, axis=1)}).to_csv(
        os.path.join(SAVE_DIR, f"{model_name}_predictions.csv"), index=False)

    # Plot Accuracy & Loss
    if history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history["accuracy"], label="Train")
        ax1.plot(history.history["val_accuracy"], label="Val")
        ax1.set_title(f"{model_name} - Accuracy")
        ax1.legend()
        
        ax2.plot(history.history["loss"], label="Train")
        ax2.plot(history.history["val_loss"], label="Val")
        ax2.set_title(f"{model_name} - Loss")
        ax2.legend()
        plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_history.png"), dpi=200)
        plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_cm.png"), dpi=200)
    plt.close()
    
    # ROC & PR Curves
    pos_probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pos_probs)
    pr, rc, _ = precision_recall_curve(y_test, pos_probs)
    
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_title("ROC Curve")
    ax3.legend()
    
    ax4.plot(rc, pr, color='green', lw=2, label=f'PR AUC = {auc(rc, pr):.3f}')
    ax4.set_title("Precision-Recall Curve")
    ax4.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_roc_pr.png"), dpi=200)
    plt.close()

    # Append to comparison CSV
    metrics_file = os.path.join(SAVE_DIR, "model_comparison.csv")
    df_new = pd.DataFrame([{"Model": model_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "Train Time (s)": round(train_time, 2)}])
    if os.path.exists(metrics_file):
        df_existing = pd.read_csv(metrics_file)
        df_existing = df_existing[df_existing["Model"] != model_name]
        df_new = pd.concat([df_existing, df_new], ignore_index=True)
    df_new.to_csv(metrics_file, index=False)