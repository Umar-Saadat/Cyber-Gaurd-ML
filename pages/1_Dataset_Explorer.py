import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import scan_dataset_folder, load_data, get_dataset_summary
from src.utils import find_label_column

st.set_page_config(page_title="Dataset Explorer - CyberGuard ML", page_icon="📊", layout="wide")

st.title("📊 Dataset Explorer")
st.markdown("Inspect raw network traffic records, column definitions, label frequencies, and data quality.")

# 1. Scan for CSV files
dataset_info = scan_dataset_folder()

if dataset_info["file_count"] == 0:
    st.error("⚠️ No CSV dataset files found in `dataset/` or current folder!")
    st.info("Please copy your CICIDS2017 CSV files (e.g., `DATA.csv`) into the `dataset/` folder.")
    st.stop()

# Selection of file to explore
file_options = {f["name"]: f["path"] for f in dataset_info["files"]}
selected_file_name = st.selectbox("Select CSV file to inspect:", list(file_options.keys()))
selected_file_path = file_options[selected_file_name]

# Choose rows to load for preview
col_preview_settings = st.columns(3)
with col_preview_settings[0]:
    preview_rows = st.number_input("Number of rows to load for preview:", min_value=100, max_value=100000, value=5000, step=1000)

with st.spinner(f"Loading preview of {preview_rows} rows from {selected_file_name}..."):
    try:
        df_preview = load_data(file_paths=[selected_file_path], max_rows=preview_rows)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

if df_preview is None or df_preview.empty:
    st.warning("Could not read any records from the selected file.")
    st.stop()

# Summary Metrics
summary = get_dataset_summary(df_preview)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows Loaded", f"{summary['total_rows']:,}")
c2.metric("Total Columns", summary['total_columns'])
c3.metric("Memory Usage", f"{summary['memory_mb']} MB")
c4.metric("Missing Values", f"{summary['missing_values_count']:,}")

st.divider()

# Tabbed Explorer
tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🏷️ Label Distribution", "🔍 Columns & Types"])

with tab1:
    st.subheader("Data Table Preview")
    num_display = st.slider("Rows to display:", 5, 50, 10)
    st.dataframe(df_preview.head(num_display), use_container_width=True)

with tab2:
    st.subheader("Label Frequencies")
    label_col = summary.get("label_column")
    if label_col:
        counts_df = pd.DataFrame(
            list(summary["label_counts"].items()),
            columns=["Traffic Class", "Count"]
        )
        counts_df["Percentage"] = (counts_df["Count"] / counts_df["Count"].sum() * 100).round(2)
        
        c_chart, c_table = st.columns([3, 2])
        
        with c_chart:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=counts_df, x="Traffic Class", y="Count", ax=ax, palette="crest")
            ax.set_title(f"Class Distribution in Loaded Sample ({selected_file_name})")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with c_table:
            st.dataframe(counts_df, use_container_width=True)
            
        if len(counts_df) == 1 and counts_df["Traffic Class"].iloc[0] == "BENIGN":
            st.info("ℹ️ Note: This particular CSV file contains only **BENIGN** normal traffic (a single day/session capture from the CICIDS2017 dataset). If you have other CICIDS2017 attack files (such as Friday/Wednesday attack captures), place them in the `dataset/` folder so the system can combine and train on both!")
    else:
        st.warning("Target column ('Label') was not found in this CSV.")

with tab3:
    st.subheader("Column Metadata & Missing Values")
    col_info = []
    for col in df_preview.columns:
        col_info.append({
            "Column Name": col,
            "Data Type": str(df_preview[col].dtype),
            "Missing Values": int(df_preview[col].isnull().sum()),
            "Sample Value": str(df_preview[col].iloc[0]) if len(df_preview) > 0 else "N/A"
        })
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)
