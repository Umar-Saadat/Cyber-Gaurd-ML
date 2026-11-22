import streamlit as st
import pandas as pd
import time
from src.data_loader import scan_dataset_folder, load_data
from src.preprocessing import clean_dataset, prepare_level1_data, prepare_level2_data, prepare_regression_data
from src.train import AVAILABLE_MODELS, AVAILABLE_REGRESSION_MODELS, train_single_model, save_trained_model
from src.evaluate import evaluate_model, evaluate_regression_model, save_results_to_csv, save_comparison_chart
from src.utils import find_label_column

st.set_page_config(page_title="Model Training - CyberGuard ML", page_icon="⚙️", layout="wide")

st.title("⚙️ Model Training")
st.markdown("Train and evaluate scikit-learn machine learning algorithms on the network intrusion dataset.")

dataset_info = scan_dataset_folder()
if dataset_info["file_count"] == 0:
    st.error("No dataset CSV files found. Please place CSVs into the `dataset/` folder.")
    st.stop()

# 1. Training Configuration Form
st.subheader("1. Configuration Settings")

col_settings_1, col_settings_2 = st.columns(2)

with col_settings_1:
    task_level = st.radio(
        "Select Model Type / Classification Level:",
        ("Level 1: Binary Classification (BENIGN vs ATTACK)",
         "Level 2: Multiclass Attack Classification",
         "Regression: Continuous Flow Metric Prediction")
    )
    is_level_1 = "Level 1" in task_level
    is_regression = "Regression" in task_level
    
    if is_regression:
        level_tag = "regression"
        target_reg_col = st.selectbox(
            "Select Numerical Target to Predict:",
            ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Total Length of Fwd Packets", "Packet Length Mean"],
            help="Flow Duration represents how long the network connection lasts (in microseconds), a critical metric for detecting prolonged attack sessions vs rapid scans."
        )
    elif is_level_1:
        level_tag = "level1"
        target_reg_col = None
    else:
        level_tag = "level2"
        target_reg_col = None

    sample_size = st.select_slider(
        "Training Sample Size (number of network flow rows):",
        options=[1000, 2500, 5000, 10000, 20000, 50000],
        value=5000,
        help="Keeping sample size manageable ensures fast training and prevents computer freezing/out-of-memory errors."
    )

with col_settings_2:
    test_size_percent = st.slider("Test Split Size (%):", min_value=10, max_value=40, value=20, step=5)
    test_size = test_size_percent / 100.0
    
    if is_regression:
        available_algo_options = list(AVAILABLE_REGRESSION_MODELS.keys())
        default_algos = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
    else:
        available_algo_options = list(AVAILABLE_MODELS.keys())
        default_algos = ["Logistic Regression", "Decision Tree", "Random Forest"]
        
    selected_models = st.multiselect(
        "Choose Algorithms to Train:",
        options=available_algo_options,
        default=default_algos
    )

st.divider()

# 2. Dataset Selection
st.subheader("2. Dataset Source")
file_names = [f["name"] for f in dataset_info["files"]]
selected_files = st.multiselect(
    "Select CSV files to include:",
    options=file_names,
    default=file_names
)

if not selected_files:
    st.warning("Please select at least one dataset file.")
    st.stop()

if not selected_models:
    st.warning("Please select at least one algorithm to train.")
    st.stop()

# 3. Training Execution Button
train_button = st.button("🚀 Start Model Training", type="primary")

if train_button:
    # Get selected file paths
    file_path_map = {f["name"]: f["path"] for f in dataset_info["files"]}
    target_paths = [file_path_map[fname] for fname in selected_files]
    
    status_box = st.status("Executing training workflow...", expanded=True)
    
    with status_box:
        st.write("📥 Loading dataset records...")
        raw_df = load_data(file_paths=target_paths, max_rows=sample_size * 2)
        
        if raw_df is None or raw_df.empty:
            st.error("Failed to load any records from selected dataset files.")
            st.stop()
            
        st.write(f"🧹 Preprocessing and cleaning {len(raw_df):,} raw records...")
        cleaned_df, clean_stats = clean_dataset(raw_df)
        st.write(f"✓ Cleaned dataset has {len(cleaned_df):,} records and {clean_stats['duplicates_removed']} duplicates removed.")
        
        # Prepare Level 1, Level 2, or Regression splits
        try:
            if is_regression:
                st.write(f"📊 Preparing Regression dataset (predicting '{target_reg_col}')...")
                X_train, X_test, y_train, y_test, feature_names, used_target = prepare_regression_data(
                    cleaned_df, target_col=target_reg_col, sample_size=sample_size, test_size=test_size
                )
                classes = None
                st.write(f"Features: {len(feature_names)} | Training Rows: {len(X_train):,} | Target: {used_target}")
            elif is_level_1:
                st.write("📊 Preparing Level 1 (BENIGN vs ATTACK) training and testing sets...")
                X_train, X_test, y_train, y_test, feature_names = prepare_level1_data(
                    cleaned_df, sample_size=sample_size, test_size=test_size
                )
                classes = ["BENIGN", "ATTACK"]
                
                # Check class representation
                train_counts = y_train.value_counts().to_dict()
                st.write(f"Class counts in training split: {train_counts}")
                if len(train_counts) < 2:
                    st.warning("⚠️ Warning: The loaded dataset file currently only contains BENIGN traffic. Attack records (class 1) are not present in this file. The model will train on single class, or you can add Friday/Wednesday CICIDS2017 files containing attacks.")
            else:
                st.write("📊 Preparing Level 2 (Attack Subtypes) training and testing sets...")
                X_train, X_test, y_train, y_test, feature_names, classes = prepare_level2_data(
                    cleaned_df, sample_size=sample_size, test_size=test_size
                )
                st.write(f"Attack classes present: {classes}")
        except Exception as prep_err:
            st.error(f"Data Preparation Error: {str(prep_err)}")
            status_box.update(label="Training halted due to data constraint", state="error")
            st.stop()
            
        # Training Loop
        results_list = []
        progress_bar = st.progress(0)
        
        for idx, model_name in enumerate(selected_models):
            st.write(f"⏳ Training **{model_name}**...")
            try:
                pipeline, duration = train_single_model(model_name, X_train, y_train)
                
                if is_regression:
                    eval_metrics = evaluate_regression_model(
                        pipeline, X_test, y_test, model_name, target_col=target_reg_col
                    )
                    eval_metrics["Training_Time"] = duration
                    saved_path, meta = save_trained_model(
                        pipeline, model_name, level_tag, feature_names, target_col=target_reg_col
                    )
                    eval_metrics["Saved_Path"] = saved_path
                    results_list.append(eval_metrics)
                    st.write(f"✅ {model_name} trained in {duration}s | R²: {eval_metrics['R2-Score']} | MAE: {eval_metrics['MAE']}")
                else:
                    eval_metrics = evaluate_model(
                        pipeline, X_test, y_test, model_name, level_name=level_tag, class_names=classes
                    )
                    eval_metrics["Training_Time"] = duration
                    saved_path, meta = save_trained_model(
                        pipeline, model_name, level_tag, feature_names, classes
                    )
                    eval_metrics["Saved_Path"] = saved_path
                    results_list.append(eval_metrics)
                    st.write(f"✅ {model_name} trained in {duration}s | Accuracy: {eval_metrics['Accuracy']*100:.2f}% | F1: {eval_metrics['F1-Score']:.4f}")
            except Exception as train_err:
                st.error(f"❌ Failed to train {model_name}: {str(train_err)}")
                
            progress_bar.progress((idx + 1) / len(selected_models))
            
        status_box.update(label="Training completed successfully!", state="complete")

    # If models trained, save results and show table
    if results_list:
        csv_path, df_res = save_results_to_csv(results_list, level_name=level_tag)
        save_comparison_chart(df_res, level_name=level_tag)
        
        st.success(f"🎉 Successfully trained and saved {len(results_list)} models! Results saved to `{csv_path}`.")
        
        st.subheader("Training Results Table")
        if is_regression:
            st.dataframe(df_res[["Model", "Target_Column", "R2-Score", "MAE", "RMSE", "MSE", "Training_Time_s"]], use_container_width=True)
        else:
            st.dataframe(df_res[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Training_Time_s"]], use_container_width=True)
        st.info("👉 Head over to the **Model Evaluation** page in the sidebar to view detailed comparison charts and diagnostic plots!")
