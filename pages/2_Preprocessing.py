import streamlit as st
import pandas as pd
from src.data_loader import scan_dataset_folder, load_data
from src.preprocessing import clean_dataset, prepare_level1_data
from src.utils import find_label_column

st.set_page_config(page_title="Data Preprocessing - CyberGuard ML", page_icon="🧹", layout="wide")

st.title("🧹 Data Preprocessing & Cleaning")
st.markdown("Understand how raw network flow statistics are validated, cleaned of anomalies, and converted into machine-learning ready feature sets.")

# Explanation of Preprocessing Pipeline
with st.expander("ℹ️ How the Preprocessing Pipeline Works (Beginner Guide)", expanded=True):
    st.markdown("""
    1. **Column Name Standardization:** Strips leading/trailing whitespaces (e.g., `' Flow Duration'` becomes `'Flow Duration'`).
    2. **Handling Infinite Values:** Network flow calculations (like `Flow Bytes/s`) frequently produce `+Inf` or `-Inf` due to zero-duration packet bursts. These are safely converted to `NaN`.
    3. **Missing Value & Duplicate Cleaning:** Rows with missing attributes or duplicated network connections are removed to prevent bias and noise.
    4. **Feature Selection:** Target labels (`Label`) and zero-variance constant features are excluded to prevent data leakage and mathematical errors.
    5. **Train / Test Splitting:** Training data is partitioned into separate train (e.g., 80%) and test (e.g., 20%) sets.
    6. **Scaling & Imputation:** Scalers (`StandardScaler`) and imputers (`SimpleImputer`) are fit **only on the training split** and applied consistently to the test split to avoid data leakage.
    """)

# Select data to inspect cleaning
dataset_info = scan_dataset_folder()
if dataset_info["file_count"] == 0:
    st.error("No dataset CSV files found. Please place CSVs into the `dataset/` folder.")
    st.stop()

file_options = {f["name"]: f["path"] for f in dataset_info["files"]}
selected_file_name = st.selectbox("Select CSV file for preprocessing analysis:", list(file_options.keys()))
selected_file_path = file_options[selected_file_name]

sample_size = st.number_input("Number of rows to load for preprocessing inspection:", min_value=500, max_value=50000, value=5000, step=500)

if st.button("Run Preprocessing Analysis"):
    with st.spinner("Executing cleaning pipeline..."):
        # 1. Load raw
        raw_df = load_data(file_paths=[selected_file_path], max_rows=sample_size)
        
        # 2. Clean
        cleaned_df, summary = clean_dataset(raw_df)
        
        # Store in session state for inspection
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["clean_summary"] = summary
        st.session_state["raw_shape"] = raw_df.shape
        st.session_state["clean_shape"] = cleaned_df.shape

if "clean_summary" in st.session_state:
    summary = st.session_state["clean_summary"]
    raw_shape = st.session_state["raw_shape"]
    clean_shape = st.session_state["clean_shape"]
    
    st.subheader("Cleaning Results Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Rows", f"{raw_shape[0]:,}")
    c2.metric("Cleaned Rows", f"{clean_shape[0]:,}")
    c3.metric("Duplicates Dropped", f"{summary['duplicates_removed']:,}")
    c4.metric("Invalid/Missing Handled", f"{summary['missing_values_handled']:,}")
    
    st.divider()
    
    st.subheader("Before & After Comparison")
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Original Dimensions:**")
        st.write(f"- Rows: `{raw_shape[0]}`")
        st.write(f"- Columns: `{raw_shape[1]}`")
    with col_after:
        st.markdown("**Cleaned Dimensions:**")
        st.write(f"- Rows: `{clean_shape[0]}`")
        st.write(f"- Columns: `{clean_shape[1]}`")
        
    st.success("✅ Preprocessing pipeline verified! Ready to be utilized in model training.")
