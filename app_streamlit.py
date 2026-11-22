import streamlit as st
import os
from src.data_loader import scan_dataset_folder
from src.train import list_saved_models

# Page Configuration
st.set_page_config(
    page_title="CyberGuard ML - Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Navy / Green Theme)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2d6a4f;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1b4332;
    }
    .guide-box {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 16px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛡️ CyberGuard ML")
st.sidebar.markdown("**Network Intrusion Detection System**")
st.sidebar.caption("BS AI Semester Project")
st.sidebar.divider()
st.sidebar.info("💡 Navigate using the pages in the sidebar to explore data, train models, view metrics, and run predictions.")

# Main Home Page Content
st.markdown('<div class="main-title">🛡️ CyberGuard ML: Network Intrusion Detection & Attack Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Two-Level Machine Learning Architecture for Network Security and Threat Analysis (CICIDS2017)</div>', unsafe_allow_html=True)

# Scan Dataset and Saved Models
dataset_info = scan_dataset_folder()
l1_models = list_saved_models("level1")
l2_models = list_saved_models("level2")
reg_models = list_saved_models("regression")

# Summary Metrics in Columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Dataset Files</div>
        <div class="metric-value">{dataset_info['file_count']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status_text = "Ready" if dataset_info['file_count'] > 0 else "Missing"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Dataset Status</div>
        <div class="metric-value">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Level 1 Saved</div>
        <div class="metric-value">{len(l1_models)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Level 2 Saved</div>
        <div class="metric-value">{len(l2_models)}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Regression Saved</div>
        <div class="metric-value">{len(reg_models)}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.divider()

# System Architecture Overview
st.subheader("🎯 Integrated Machine Learning Architecture")
c_arch1, c_arch2, c_arch3 = st.columns(3)

with c_arch1:
    st.markdown("""
    ### 🟢 Level 1: Binary Detection
    * **Objective:** Distinguish between **BENIGN** (Normal) traffic and **ATTACK** (Suspicious) traffic.
    * **Confidence:** Output detection confidence (1–100%).
    * **Algorithms:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM.
    """)

with c_arch2:
    st.markdown("""
    ### 🔴 Level 2: Attack Type
    * **Objective:** Identify the specific attack vector:
      * *DDoS, PortScan, Patator, Web Attack, etc.*
    * **Confidence Score:** Derived from model `predict_proba()` (1–100%).
    * **Condition:** Triggered for flagged attack traffic.
    """)

with c_arch3:
    st.markdown("""
    ### 📈 Level 3: Flow Regression
    * **Objective:** Predict continuous network flow parameters (e.g. *Flow Duration*, *Flow Bytes/s*).
    * **Utility:** Quantify flow intensity & duration profile to recognize connection anomalies.
    * **Algorithms:** Linear Regression, Ridge, Decision Tree, Random Forest Regressor.
    """)

st.write("")

# Files Status Table
st.subheader("📁 Dataset Files Status")
if dataset_info['file_count'] > 0:
    for f in dataset_info['files']:
        st.write(f"- 📄 **{f['name']}** — Size: `{f['size_mb']} MB` (Path: `{f['path']}`)")
else:
    st.warning("⚠️ No CSV files found in the `dataset/` folder. Please place CICIDS2017 CSV files into the `dataset/` directory.")

st.markdown("""
<div class="guide-box">
    <h4>🚀 Quick Start Navigation Guide</h4>
    <ol>
        <li><strong>Dataset Explorer:</strong> Preview dataset columns, shapes, and class distributions.</li>
        <li><strong>Preprocessing:</strong> Clean null/infinite values, scale numerical features, and inspect train/test splits.</li>
        <li><strong>Model Training:</strong> Train Level 1 (Binary) and Level 2 (Multiclass) classifiers with customizable sample sizes.</li>
        <li><strong>Model Evaluation:</strong> View Accuracy, Precision, Recall, F1 scores, Confusion Matrices, and comparison charts.</li>
        <li><strong>Predictions:</strong> Upload your own CSV network traffic file or test manual flow features live!</li>
        <li><strong>About Project:</strong> Comprehensive documentation of methodology, dataset attributes, and limitations.</li>
    </ol>
</div>
""", unsafe_allow_html=True)
