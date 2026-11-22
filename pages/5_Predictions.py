import streamlit as st
import pandas as pd
import numpy as np
import os
from src.train import list_saved_models, load_saved_model
from src.predict import predict_from_dataframe, predict_single_instance
from src.utils import clean_column_names

st.set_page_config(page_title="Predictions - CyberGuard ML", page_icon="🎯", layout="wide")

st.title("🎯 Live Intrusion Prediction")
st.markdown("Test network traffic using trained Level 1 (Binary) and Level 2 (Attack Classification) models.")

# 1. Load Available Models
l1_models = list_saved_models("level1")
l2_models = list_saved_models("level2")
reg_models = list_saved_models("regression")

if not l1_models:
    st.warning("⚠️ No Level 1 trained models found! Please train at least one model in the **Model Training** page before making predictions.")
    st.stop()

# Model Selection Dropdowns
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    l1_choices = {m["filename"]: m["filepath"] for m in l1_models}
    selected_l1_file = st.selectbox("1. Level 1 (Binary Model):", list(l1_choices.keys()))
    l1_package = load_saved_model(l1_choices[selected_l1_file])

with col_m2:
    l2_package = None
    if l2_models:
        l2_choices = {"None (Level 1 Only)": None}
        l2_choices.update({m["filename"]: m["filepath"] for m in l2_models})
        selected_l2_file = st.selectbox("2. Level 2 (Attack Type Model):", list(l2_choices.keys()))
        if l2_choices[selected_l2_file]:
            l2_package = load_saved_model(l2_choices[selected_l2_file])
    else:
        st.info("ℹ️ No Level 2 model saved.")

with col_m3:
    reg_package = None
    if reg_models:
        reg_choices = {"None (No Regression)": None}
        reg_choices.update({m["filename"]: m["filepath"] for m in reg_models})
        selected_reg_file = st.selectbox("3. Regression Model (Continuous Metric):", list(reg_choices.keys()))
        if reg_choices[selected_reg_file]:
            reg_package = load_saved_model(reg_choices[selected_reg_file])
    else:
        st.info("ℹ️ No Regression model saved.")

st.divider()

# Input Method Tabs
method_tab1, method_tab2 = st.tabs(["📁 Method 1: Batch CSV Upload", "✍️ Method 2: Manual Feature Input (Demo)"])

# Method 1: CSV Upload
with method_tab1:
    st.subheader("Upload Network Traffic CSV")
    st.markdown("Upload a CSV file containing network flow records (such as capture records from Wireshark / CICFlowMeter).")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, nrows=1000)
            df_upload = clean_column_names(df_upload)
            st.write(f"✓ Uploaded **{len(df_upload)}** rows and **{len(df_upload.columns)}** columns.")
            
            with st.expander("Preview Uploaded Data"):
                st.dataframe(df_upload.head(5), use_container_width=True)
                
            if st.button("Run Batch Prediction on Uploaded Data", type="primary"):
                with st.spinner("Classifying network flows and predicting continuous metrics..."):
                    pred_df = predict_from_dataframe(df_upload, l1_package, l2_package, reg_package)
                    
                    st.success("✅ Prediction complete!")
                    
                    # Prediction summary
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1:
                        st.markdown("**Binary Distribution:**")
                        st.dataframe(pred_df["Level_1_Prediction"].value_counts())
                    with col_p2:
                        st.markdown("**Attack Type Distribution:**")
                        st.dataframe(pred_df["Predicted_Attack_Type"].value_counts())
                    with col_p3:
                        st.markdown("**Mean Confidence Score:**")
                        mean_conf = pred_df["Attack_Confidence_%"].mean()
                        st.metric("Avg Confidence", f"{mean_conf:.1f}%")
                        
                    st.subheader("Predictions Data Table")
                    display_cols = ["Level_1_Prediction", "Predicted_Attack_Type", "Attack_Confidence_%"]
                    if reg_package:
                        reg_target = reg_package["metadata"].get("target_col", "Flow Duration")
                        if f"Predicted_{reg_target}" in pred_df.columns:
                            display_cols.append(f"Predicted_{reg_target}")
                    display_cols += [c for c in pred_df.columns if c not in display_cols][:5]
                    
                    st.dataframe(pred_df[display_cols].head(50), use_container_width=True)
                    
                    # Download button
                    csv_bytes = pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_bytes,
                        file_name="cyberguard_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {str(e)}")

# Method 2: Manual Feature Input
with method_tab2:
    st.subheader("Interactive Single Flow Demonstration")
    st.caption("Enter key network traffic metrics to simulate a real-time intrusion test.")
    
    key_features = [
        ("Destination Port", 80, "Destination TCP/UDP Port (e.g. 80 for HTTP, 443 for HTTPS, 21 for FTP)"),
        ("Flow Duration", 100000, "Duration of the flow in microseconds"),
        ("Total Fwd Packets", 10, "Total packets sent in forward direction"),
        ("Total Backward Packets", 8, "Total packets sent in backward direction"),
        ("Total Length of Fwd Packets", 500, "Total size of packet payloads in forward direction (bytes)"),
        ("Total Length of Bwd Packets", 800, "Total size of packet payloads in backward direction (bytes)"),
        ("Flow Bytes/s", 13000.0, "Flow throughput in bytes per second"),
        ("Flow Packets/s", 180.0, "Flow rate in packets per second"),
        ("Average Packet Size", 72.0, "Average packet length across connection"),
        ("SYN Flag Count", 1, "Number of packets with SYN flag set (connection establishment)")
    ]
    
    form_inputs = {}
    col_f1, col_f2 = st.columns(2)
    
    for i, (feat_name, def_val, help_txt) in enumerate(key_features):
        col_target = col_f1 if (i % 2 == 0) else col_f2
        with col_target:
            if isinstance(def_val, float):
                form_inputs[feat_name] = st.number_input(feat_name, value=float(def_val), help=help_txt)
            else:
                form_inputs[feat_name] = st.number_input(feat_name, value=int(def_val), help=help_txt)
                
    st.info("💡 Note: Features not entered manually above will be filled using the model's fitted median values.")
    
    if st.button("Classify Single Flow", type="primary"):
        with st.spinner("Evaluating connection characteristics..."):
            pred_result = predict_single_instance(form_inputs, l1_package, l2_package, reg_package)
            
            c_res1, c_res2, c_res3 = st.columns(3)
            with c_res1:
                if pred_result["level1"] == "BENIGN":
                    st.success("### 🟢 Classification: BENIGN\nNormal, benign network flow.")
                else:
                    st.error("### 🔴 Classification: ATTACK\nSuspicious malicious traffic detected!")
                    
            with c_res2:
                st.markdown(f"### 🛡️ Predicted Attack Type:\n**{pred_result['attack_type']}**")
                conf_val = float(pred_result['confidence'])
                st.markdown(f"**Confidence Score:** `{conf_val}%`")
                # Visual confidence bar
                st.progress(min(1.0, max(0.01, conf_val / 100.0)))
                
            with c_res3:
                if reg_package and pred_result["regression_val"] is not None:
                    target_n = pred_result["regression_target"]
                    val = pred_result["regression_val"]
                    st.markdown(f"### 📈 Predicted {target_n}:\n**{val:,.2f}**")
                    st.caption("Estimated continuous network metric using the trained regression model.")
                else:
                    st.info("### 📈 Regression Metric:\n*No regression model selected.*")
