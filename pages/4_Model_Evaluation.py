import streamlit as st
import os
import glob
import pandas as pd
from PIL import Image
from src.utils import RESULTS_DIR, MODELS_DIR

st.set_page_config(page_title="Model Evaluation - CyberGuard ML", page_icon="📈", layout="wide")

st.title("📈 Model Evaluation & Performance")
st.markdown("Examine accuracy, precision, recall, F1-score, confusion matrix heatmaps, and classification metrics.")

# Select Level to View
level_option = st.radio("Select Model Evaluation View:", ["Level 1 (Binary)", "Level 2 (Multiclass)", "Regression (Continuous)"], horizontal=True)

if "Regression" in level_option:
    level_tag = "regression"
    is_regression = True
elif "Level 1" in level_option:
    level_tag = "level1"
    is_regression = False
else:
    level_tag = "level2"
    is_regression = False

results_csv = os.path.join(RESULTS_DIR, f"results_{level_tag}.csv")
comparison_img = os.path.join(RESULTS_DIR, f"comparison_{level_tag}.png")

if not os.path.exists(results_csv):
    st.warning(f"⚠️ No training results found for {level_option}. Please train one or more models in the **Model Training** page first!")
    st.stop()

# Load Results CSV
results_df = pd.read_csv(results_csv)

st.subheader(f"📊 Metrics Comparison Table ({level_option})")

if is_regression:
    sort_metric = st.selectbox("Sort models by:", ["R2-Score", "MAE", "RMSE", "MSE", "Training_Time_s"], index=0)
    ascending = False if sort_metric == "R2-Score" else True
    sorted_df = results_df.sort_values(by=sort_metric, ascending=ascending).reset_index(drop=True)
    st.dataframe(sorted_df, use_container_width=True)
    
    best_model = sorted_df.iloc[0]
    st.success(f"🏆 **Top Performer ({sort_metric}):** `{best_model['Model']}` with **{best_model[sort_metric]}**")
    
    st.divider()
    
    # Comparison Chart
    st.subheader("📊 Cross-Model Performance Chart")
    if os.path.exists(comparison_img):
        img = Image.open(comparison_img)
        st.image(img, caption="Regression Model Comparison (R² Score)", use_container_width=True)
        st.caption("Explanation: The R² (Coefficient of Determination) score measures how well the regression model explains the variance of the continuous target (1.0 = perfect prediction).")
        
    st.divider()
    
    # Actual vs Predicted Plots
    st.subheader("🎯 Actual vs. Predicted Plots")
    reg_plots = glob.glob(os.path.join(RESULTS_DIR, "reg_actual_vs_pred_*.png"))
    if reg_plots:
        cols = st.columns(min(len(reg_plots), 2))
        for i, r_plot in enumerate(reg_plots):
            m_name = os.path.basename(r_plot).replace("reg_actual_vs_pred_", "").replace(".png", "").replace("_", " ")
            with cols[i % len(cols)]:
                st.markdown(f"**{m_name}**")
                im = Image.open(r_plot)
                st.image(im, use_container_width=True)
                st.caption("Points closer to the red dashed line (y = x) indicate more accurate regression predictions.")
    else:
        st.info("No Actual vs Predicted plots found.")

else:
    # Allow sorting by metric for classification
    sort_metric = st.selectbox("Sort models by:", ["F1-Score", "Accuracy", "Precision", "Recall", "Training_Time_s"], index=0)
    sorted_df = results_df.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)
    st.dataframe(sorted_df, use_container_width=True)

    # Highlight Best Model
    best_model = sorted_df.iloc[0]
    st.success(f"🏆 **Top Performer ({sort_metric}):** `{best_model['Model']}` with **{best_model[sort_metric]}**")

    st.divider()

    # Comparison Chart
    st.subheader("📊 Cross-Model Performance Chart")
    if os.path.exists(comparison_img):
        img = Image.open(comparison_img)
        st.image(img, caption=f"Model Comparison across Metrics ({level_option})", use_container_width=True)
        st.caption("Explanation: The bar chart contrasts the overall accuracy, precision (attack detection certainty), recall (attack capture rate), and harmonic F1 score across all tested algorithms.")
    else:
        st.info("No comparison chart available.")

    st.divider()

    # Confusion Matrices
    st.subheader("🧩 Confusion Matrix Heatmaps")
    cm_files = glob.glob(os.path.join(RESULTS_DIR, f"cm_{level_tag}_*.png"))

    if cm_files:
        cols = st.columns(min(len(cm_files), 2))
        for i, cm_file in enumerate(cm_files):
            model_name = os.path.basename(cm_file).replace(f"cm_{level_tag}_", "").replace(".png", "").replace("_", " ")
            with cols[i % len(cols)]:
                st.markdown(f"**{model_name}**")
                im = Image.open(cm_file)
                st.image(im, use_container_width=True)
                st.caption("Shows True Negatives, False Positives (Type I error), False Negatives (Type II error), and True Positives.")
    else:
        st.info("No confusion matrix images found for this level.")
