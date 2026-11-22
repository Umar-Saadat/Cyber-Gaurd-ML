# evaluate.py - Model evaluation metrics, charts, and report generation
import os
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for generating figures safely
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.utils import RESULTS_DIR

def evaluate_model(pipeline, X_test, y_test, model_name, level_name="level1", class_names=None, train_acc=None):
    """
    Evaluates a trained model on test data.
    Computes and prints training accuracy, testing accuracy, classification report, and confusion matrix.
    Generates and saves confusion matrix heatmap.
    """
    y_pred = pipeline.predict(X_test)
    
    is_binary = (level_name == "level1")
    average_mode = "binary" if is_binary else "weighted"
    
    acc = accuracy_score(y_test, y_pred)
    
    try:
        prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
        rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)
    except Exception:
        prec, rec, f1 = 0.0, 0.0, 0.0

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, zero_division=0)
    
    # Determine class labels for plotting
    if class_names is None:
        if is_binary:
            class_names = ["BENIGN", "ATTACK"]
        else:
            class_names = sorted(list(set(y_test).union(set(y_pred))))
            
    # Print comprehensive evaluation results to terminal
    print("\n" + "=" * 65)
    print(f"  🛡️  MODEL EVALUATION: {model_name} [{level_name.upper()}]")
    print("=" * 65)
    if train_acc is not None:
        print(f"  🔹 Training Accuracy: {train_acc * 100:.2f}% ({train_acc})")
    print(f"  🔹 Testing Accuracy:  {acc * 100:.2f}% ({acc})")
    print(f"  🔹 Precision ({average_mode}): {prec:.4f}")
    print(f"  🔹 Recall ({average_mode}):    {rec:.4f}")
    print(f"  🔹 F1-Score ({average_mode}):  {f1:.4f}")
    print("\n--- CLASSIFICATION REPORT ---")
    print(report_str)
    print("--- CONFUSION MATRIX ---")
    print(cm)
    print("=" * 65 + "\n")
            
    # Generate and save confusion matrix figure
    cm_path = save_confusion_matrix_plot(cm, class_names, model_name, level_name)
    
    metrics = {
        "Model": model_name,
        "Level": level_name,
        "Accuracy": round(acc, 4),
        "Training_Accuracy": round(train_acc, 4) if train_acc is not None else round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Confusion_Matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
        "Report_Dict": report_dict,
        "Report_Text": report_str,
        "CM_Plot_Path": cm_path
    }
    
    return metrics

def save_confusion_matrix_plot(cm, labels, model_name, level_name):
    """
    Plots and saves a heatmap for confusion matrix.
    """
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues" if level_name == "level1" else "Greens",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {model_name} ({level_name.upper()})", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    fname = f"cm_{level_name}_{clean_name}.png"
    filepath = os.path.join(RESULTS_DIR, fname)
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def evaluate_regression_model(pipeline, X_test, y_test, model_name, target_col="Flow Duration"):
    """
    Evaluates a trained regression model on test data using MAE, MSE, RMSE, and R2.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 65)
    print(f"  🛡️  REGRESSION EVALUATION: {model_name} [Target: {target_col}]")
    print("=" * 65)
    print(f"  🔹 R² Score: {r2:.4f}")
    print(f"  🔹 MAE:      {mae:.2f}")
    print(f"  🔹 MSE:      {mse:.2f}")
    print(f"  🔹 RMSE:     {rmse:.2f}")
    print("=" * 65 + "\n")
    
    # Save diagnostic regression scatter plot
    plot_path = save_regression_plot(y_test, y_pred, model_name, target_col)
    
    metrics = {
        "Model": model_name,
        "Level": "regression",
        "Target_Column": target_col,
        "R2-Score": round(r2, 4),
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "Plot_Path": plot_path
    }
    return metrics

def save_regression_plot(y_true, y_pred, model_name, target_col):
    """
    Saves a scatter plot of Actual vs Predicted continuous values.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, color="#00f5d4", edgecolors="none", s=25)
    
    # Diagonal reference line
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], color="#ff0055", linestyle="--", linewidth=1.5, label="Ideal")
    
    plt.title(f"Actual vs Predicted: {target_col} ({model_name})")
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    fname = f"reg_actual_vs_pred_{clean_name}.png"
    filepath = os.path.join(RESULTS_DIR, fname)
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def save_results_to_csv(results_list, level_name="level1"):
    """
    Appends or creates results CSV file in results/ directory.
    """
    filename = f"results_{level_name}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    df_new = pd.DataFrame(results_list)
    # Remove complex objects before saving to CSV
    cols_to_drop = [c for c in ["Confusion_Matrix", "Report_Dict", "Report_Text"] if c in df_new.columns]
    df_clean = df_new.drop(columns=cols_to_drop)
    
    if os.path.exists(filepath):
        try:
            df_old = pd.read_csv(filepath)
            df_combined = pd.concat([df_old, df_clean], ignore_index=True).drop_duplicates(subset=["Model"], keep="last")
        except Exception:
            df_combined = df_clean
    else:
        df_combined = df_clean
        
    df_combined.to_csv(filepath, index=False)
    return filepath, df_combined

def save_comparison_chart(df_results, level_name="level1"):
    """
    Generates a comparative bar chart of trained models.
    """
    if df_results.empty or "Model" not in df_results.columns:
        return None
        
    plt.figure(figsize=(8, 4.5))
    if level_name == "regression":
        if "R2-Score" in df_results.columns:
            sns.barplot(data=df_results, x="Model", y="R2-Score", palette="crest")
            plt.ylabel("R² Score")
            plt.title("Regression Model Comparison: R² Score")
    else:
        if "Accuracy" in df_results.columns:
            sns.barplot(data=df_results, x="Model", y="Accuracy", palette="viridis")
            plt.ylabel("Accuracy Score")
            plt.title(f"Model Accuracy Comparison ({level_name.upper()})")
            plt.ylim(0, 1.05)
            
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    
    fname = f"comparison_{level_name}.png"
    filepath = os.path.join(RESULTS_DIR, fname)
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath
