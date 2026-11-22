# utils.py - Helper functions and utility constants
import os
import glob
import re
import pandas as pd
import numpy as np

# Default directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required folders exist
for folder in [DATASET_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Mapping between UI element IDs and dataset feature column names
UI_FEATURE_MAPPING = {
    "feat-port": "Destination Port",
    "feat-duration": "Flow Duration",
    "feat-fwd-pkts": "Total Fwd Packets",
    "feat-bwd-pkts": "Total Backward Packets",
    "feat-fwd-len": "Total Length of Fwd Packets",
    "feat-bwd-len": "Total Length of Bwd Packets",
    "feat-bytes-sec": "Flow Bytes/s",
    "feat-pkts-sec": "Flow Packets/s",
    "feat-syn-count": "SYN Flag Count"
}

def get_dataset_files():
    """
    Search for all CSV files in the dataset folder and the project root folder.
    Returns a sorted list of CSV file paths without duplicates.
    """
    files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    dataset_basenames = {os.path.basename(f).lower() for f in files}
    
    # Also check if CSV exists in base directory and include if not already in dataset
    root_csvs = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    for rf in root_csvs:
        if os.path.basename(rf).lower() not in dataset_basenames:
            files.append(rf)
            
    files.sort()
    return files

def clean_column_names(df):
    """
    Strip leading and trailing spaces from column names.
    """
    df.columns = [col.strip() for col in df.columns]
    return df

def find_label_column(columns):
    """
    Find the label/target column name among the columns.
    Typically named 'Label' or contains 'label'.
    """
    for col in columns:
        if col.strip().lower() == "label":
            return col
    for col in columns:
        if "label" in col.strip().lower():
            return col
    return None

def normalize_attack_label(label):
    """
    Normalizes attack labels across different CICIDS2017 files to standard canonical names:
    - Normalizes character encoding artifacts (ï¿½, \ufffd, etc.).
    - Standardizes uppercase/lowercase and whitespace.
    - Preserves specific attack families (DDoS, PortScan, Bot, Infiltration,
      Web Attack variants, Patator variants, DoS variants, Heartbleed).
    """
    if pd.isna(label):
        return "BENIGN"
        
    s = str(label).strip()
    if s.upper() == "BENIGN":
        return "BENIGN"
        
    # Replace character encoding artifacts
    s = s.replace("ï¿½", "-").replace("\ufffd", "-")
    # Clean multiple spaces or dashes
    s = re.sub(r'\s+', ' ', s).strip()
    
    s_lower = s.lower()
    if "sql injection" in s_lower:
        return "Web Attack - Sql Injection"
    elif "xss" in s_lower:
        return "Web Attack - XSS"
    elif "brute force" in s_lower:
        return "Web Attack - Brute Force"
    elif "slowloris" in s_lower:
        return "DoS slowloris"
    elif "slowhttptest" in s_lower:
        return "DoS Slowhttptest"
    elif "goldeneye" in s_lower:
        return "DoS GoldenEye"
    elif "hulk" in s_lower:
        return "DoS Hulk"
    elif "heartbleed" in s_lower:
        return "Heartbleed"
    elif "ftp" in s_lower and "patator" in s_lower:
        return "FTP-Patator"
    elif "ssh" in s_lower and "patator" in s_lower:
        return "SSH-Patator"
    elif "portscan" in s_lower:
        return "PortScan"
    elif "ddos" in s_lower:
        return "DDoS"
    elif "bot" in s_lower:
        return "Bot"
    elif "infiltration" in s_lower:
        return "Infiltration"
        
    return s
