# server.py - Modern Flask Backend API for CyberGuard ML
import os
import time
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file

from src.utils import BASE_DIR, DATASET_DIR, MODELS_DIR, RESULTS_DIR, clean_column_names, get_dataset_files
from src.data_loader import scan_dataset_folder, load_data, load_balanced_training_data, get_attack_samples
from src.preprocessing import clean_dataset, prepare_level1_data, prepare_level2_data, prepare_regression_data
from src.train import (
    AVAILABLE_MODELS,
    AVAILABLE_REGRESSION_MODELS,
    train_single_model,
    save_trained_model,
    load_saved_model,
    list_saved_models
)
from src.evaluate import (
    evaluate_model,
    evaluate_regression_model,
    save_results_to_csv,
    save_comparison_chart
)
from src.predict import predict_from_dataframe, predict_single_instance

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# ----------------- HOME ROUTE -----------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------- SYSTEM STATUS API -----------------
@app.route("/api/status")
def get_status():
    dataset_info = scan_dataset_folder()
    l1_models = list_saved_models("level1")
    l2_models = list_saved_models("level2")
    reg_models = list_saved_models("regression")
    
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(('.png', '.csv'))] if os.path.exists(RESULTS_DIR) else []
    
    return jsonify({
        "status": "online",
        "dataset_files": dataset_info["file_count"],
        "dataset_details": dataset_info["files"],
        "models_count": {
            "level1": len(l1_models),
            "level2": len(l2_models),
            "regression": len(reg_models),
            "total": len(l1_models) + len(l2_models) + len(reg_models)
        },
        "results_files_count": len(result_files)
    })

# ----------------- DATASET API -----------------
@app.route("/api/dataset/preview")
def dataset_preview():
    try:
        max_rows = int(request.args.get("rows", 100))
        df = load_data(max_rows=max_rows)
        if df is None or df.empty:
            return jsonify({"error": "No CSV files found in dataset directory"}), 404
            
        columns = list(df.columns)
        
        # Check label distribution
        label_col = next((c for c in columns if "label" in c.lower()), None)
        label_counts = {}
        if label_col:
            raw_counts = df[label_col].value_counts().to_dict()
            label_counts = {str(k).strip(): int(v) for k, v in raw_counts.items()}
            
        # Replace NaN / Inf for JSON serialization
        preview_df = df.head(50).replace([np.inf, -np.inf], "Inf").fillna("N/A")
        records = preview_df.to_dict(orient="records")
        
        return jsonify({
            "success": True,
            "total_columns": len(columns),
            "columns": columns,
            "sample_rows_count": len(df),
            "label_column": label_col,
            "label_distribution": label_counts,
            "records": records
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------- PREPROCESSING STATS API -----------------
@app.route("/api/preprocessing/stats")
def preprocessing_stats():
    try:
        sample_rows = int(request.args.get("sample", 1000))
        df = load_data(max_rows=sample_rows)
        if df is None or df.empty:
            return jsonify({"error": "No dataset found"}), 404
            
        cleaned_df, report = clean_dataset(df)
        columns_sample = list(cleaned_df.columns)[:25]
        
        return jsonify({
            "success": True,
            "report": report,
            "sample_features": columns_sample,
            "total_features": len(cleaned_df.columns)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------- MODELS LIST API -----------------
@app.route("/api/models")
def get_models():
    l1 = list_saved_models("level1")
    l2 = list_saved_models("level2")
    reg = list_saved_models("regression")
    
    # Available model architectures for training
    supported = {
        "level1": list(AVAILABLE_MODELS.keys()),
        "level2": list(AVAILABLE_MODELS.keys()),
        "regression": list(AVAILABLE_REGRESSION_MODELS.keys())
    }
    
    return jsonify({
        "saved": {
            "level1": l1,
            "level2": l2,
            "regression": reg
        },
        "available_architectures": supported
    })

# ----------------- ATTACK SAMPLES API -----------------
@app.route("/api/attack-samples")
def get_attack_samples_api():
    try:
        samples_data = get_attack_samples()
        return jsonify(samples_data)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

# ----------------- TRAINING API -----------------
@app.route("/api/train", methods=["POST"])
def run_training():
    try:
        data = request.get_json() or {}
        level = data.get("level", "level1")
        model_name = data.get("model_name", "Random Forest")
        sample_size = int(data.get("sample_size", 5000))
        test_size = float(data.get("test_size", 0.2))
        target_reg_col = data.get("target_col", "Flow Duration")
        
        # 1. Load data from ALL available CSV files in the dataset folder, handling class imbalance
        raw_df = load_balanced_training_data(sample_size=sample_size, level=level)
        if raw_df is None or raw_df.empty:
            return jsonify({"error": "No dataset available for training"}), 400
            
        # 2. Clean data (inf handling, na dropping, duplicates, leaky columns)
        cleaned_df, clean_report = clean_dataset(raw_df)
        
        # 3. Prepare split based on level
        if level == "regression":
            X_train, X_test, y_train, y_test, feature_names, used_target = prepare_regression_data(
                cleaned_df, target_col=target_reg_col, sample_size=sample_size, test_size=test_size
            )
            classes = None
        elif level == "level1":
            X_train, X_test, y_train, y_test, feature_names = prepare_level1_data(
                cleaned_df, sample_size=sample_size, test_size=test_size
            )
            classes = ["BENIGN", "ATTACK"]
            used_target = "Label"
        else: # level2
            X_train, X_test, y_train, y_test, feature_names, classes = prepare_level2_data(
                cleaned_df, sample_size=sample_size, test_size=test_size
            )
            used_target = "Label"
            if not classes or len(classes) < 2:
                return jsonify({"error": "Dataset does not contain enough attack subtypes to train Level 2."}), 400
                
        # 4. Train pipeline (imputer + scaler + model)
        pipeline, duration, train_acc = train_single_model(model_name, X_train, y_train)
        
        # 5. Evaluate and save metrics
        if level == "regression":
            eval_metrics = evaluate_regression_model(pipeline, X_test, y_test, model_name, target_col=used_target)
            eval_metrics["Training_Time_s"] = duration
            saved_path, meta = save_trained_model(
                pipeline, model_name, "regression", feature_names, target_col=used_target
            )
        else:
            eval_metrics = evaluate_model(
                pipeline, X_test, y_test, model_name, level_name=level, class_names=classes, train_acc=train_acc
            )
            eval_metrics["Training_Time_s"] = duration
            saved_path, meta = save_trained_model(
                pipeline, model_name, level, feature_names, class_names=classes, train_acc=train_acc
            )
            
        # 6. Save results to CSV & plot
        save_results_to_csv([eval_metrics], level_name=level)
        save_comparison_chart(pd.DataFrame([eval_metrics]), level_name=level)
        
        return jsonify({
            "success": True,
            "level": level,
            "model_name": model_name,
            "duration": duration,
            "metrics": eval_metrics,
            "saved_model": os.path.basename(saved_path),
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "features_used": len(feature_names),
            "classes_learned": classes if classes else ["Continuous Flow Metric"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ----------------- EVALUATION RESULTS API -----------------
@app.route("/api/results")
def get_results():
    data = {}
    for level in ["level1", "level2", "regression"]:
        csv_file = os.path.join(RESULTS_DIR, f"results_{level}.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                data[level] = df.to_dict(orient="records")
            except Exception:
                data[level] = []
        else:
            data[level] = []
            
    # List generated plot files
    plots = []
    if os.path.exists(RESULTS_DIR):
        for f in os.listdir(RESULTS_DIR):
            if f.endswith(".png"):
                plots.append({
                    "name": f,
                    "url": f"/results/{f}",
                    "type": "confusion_matrix" if f.startswith("cm_") else ("regression" if "reg_" in f or "regression" in f else "comparison")
                })
                
    return jsonify({
        "success": True,
        "metrics_tables": data,
        "plots": plots
    })

# ----------------- SERVE RESULT PLOTS -----------------
@app.route("/results/<filename>")
def serve_result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)

# ----------------- PREDICTION APIs -----------------
@app.route("/api/predict/single", methods=["POST"])
def predict_single():
    try:
        data = request.get_json() or {}
        feature_dict = data.get("features", {})
        l1_file = data.get("l1_model")
        l2_file = data.get("l2_model")
        reg_file = data.get("reg_model")
        
        # Load packages
        l1_package = load_saved_model(l1_file) if l1_file else None
        if not l1_package:
            all_l1 = list_saved_models("level1")
            if all_l1:
                l1_package = load_saved_model(all_l1[0]["filename"])
            else:
                return jsonify({"error": "No Level 1 model trained or selected. Please train a model first."}), 400
                
        l2_package = load_saved_model(l2_file) if l2_file else None
        if not l2_package:
            all_l2 = list_saved_models("level2")
            if all_l2:
                l2_package = load_saved_model(all_l2[0]["filename"])
                
        reg_package = load_saved_model(reg_file) if reg_file else None
        if not reg_package:
            all_reg = list_saved_models("regression")
            if all_reg:
                reg_package = load_saved_model(all_reg[0]["filename"])
        
        result = predict_single_instance(feature_dict, l1_package, l2_package, reg_package)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
            
        l1_file = request.form.get("l1_model")
        l2_file = request.form.get("l2_model")
        reg_file = request.form.get("reg_model")
        
        # Load packages
        l1_package = load_saved_model(l1_file) if l1_file else None
        if not l1_package:
            all_l1 = list_saved_models("level1")
            if all_l1:
                l1_package = load_saved_model(all_l1[0]["filename"])
            else:
                return jsonify({"error": "No Level 1 model trained or selected."}), 400
                
        l2_package = load_saved_model(l2_file) if l2_file else None
        if not l2_package:
            all_l2 = list_saved_models("level2")
            if all_l2:
                l2_package = load_saved_model(all_l2[0]["filename"])
                
        reg_package = load_saved_model(reg_file) if reg_file else None
        
        df_upload = pd.read_csv(file, nrows=1000)
        df_upload = clean_column_names(df_upload)
        
        pred_df = predict_from_dataframe(df_upload, l1_package, l2_package, reg_package)
        
        # Save batch file for download
        timestamp = int(time.time())
        batch_filename = f"predictions_batch_{timestamp}.csv"
        batch_filepath = os.path.join(RESULTS_DIR, batch_filename)
        pred_df.to_csv(batch_filepath, index=False)
        
        # Statistics
        total_rows = len(pred_df)
        benign_count = int((pred_df["Level_1_Prediction"] == "BENIGN").sum())
        attack_count = int((pred_df["Level_1_Prediction"] == "ATTACK").sum())
        avg_conf = round(float(pred_df["Attack_Confidence_%"].mean()), 1)
        
        attack_distribution = pred_df["Predicted_Attack_Type"].value_counts().to_dict()
        
        target_name = (reg_package["metadata"].get("target_col") or "Flow Duration") if reg_package else "Flow Duration"
        pred_reg_col = f"Predicted_{target_name}"
        avg_reg = round(float(pred_df[pred_reg_col].mean()), 2) if pred_reg_col in pred_df.columns else None
        
        # Preview rows
        display_cols = ["Level_1_Prediction", "Level_1_Confidence_%", "Predicted_Attack_Type", "Attack_Confidence_%"]
        if pred_reg_col in pred_df.columns:
            display_cols.append(pred_reg_col)
            
        preview_records = pred_df[display_cols].head(30).to_dict(orient="records")
        
        return jsonify({
            "success": True,
            "total_rows": total_rows,
            "benign_count": benign_count,
            "attack_count": attack_count,
            "avg_confidence": avg_conf,
            "avg_regression": avg_reg,
            "regression_target": target_name,
            "attack_distribution": {str(k): int(v) for k, v in attack_distribution.items()},
            "preview": preview_records,
            "download_filename": batch_filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/download/<filename>")
def download_batch_results(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🛡️  CyberGuard ML - Modern SOC Dashboard Server")
    print("  Access dashboard at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
