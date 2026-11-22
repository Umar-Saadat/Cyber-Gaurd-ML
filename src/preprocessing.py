# preprocessing.py - Data cleaning, feature handling, and train/test splitting
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import find_label_column, normalize_attack_label

# Leaky, duplicate, or irrelevant column candidates to remove
IRRELEVANT_COLUMNS = [
    "Fwd Header Length.1", # Exact duplicate of 'Fwd Header Length' in CICIDS2017
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Source Port"
]

def clean_dataset(df):
    """
    Performs standard cleaning on raw dataframe:
    1. Removes irrelevant / duplicate columns to prevent data leakage.
    2. Normalizes target label column across all files.
    3. Replaces +/- infinity with NaN.
    4. Drops duplicate records.
    5. Drops rows with missing or invalid values.
    Returns: cleaned DataFrame, summary dict
    """
    original_rows = len(df)
    df = df.copy()
    
    # 1. Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Remove irrelevant and duplicate columns to prevent data leakage
    drop_cols = [c for c in IRRELEVANT_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        
    # 3. Check and clean label column
    label_col = find_label_column(df.columns)
    if label_col:
        df[label_col] = df[label_col].apply(normalize_attack_label)
        
    # 4. Replace inf and -inf with NaN across all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
    # Check for string columns that should be numeric
    for col in df.columns:
        if col != label_col and df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notnull().sum() > 0.5 * len(df):
                df[col] = converted
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
    missing_before = int(df.isnull().sum().sum())
    
    # 5. Drop duplicates
    duplicates_count = int(df.duplicated().sum())
    df = df.drop_duplicates()
    
    # 6. Drop rows containing NaNs
    df = df.dropna()
    
    summary = {
        "original_rows": original_rows,
        "clean_rows": len(df),
        "removed_rows": original_rows - len(df),
        "duplicates_removed": duplicates_count,
        "missing_values_handled": missing_before,
        "remaining_columns": len(df.columns)
    }
    
    return df, summary

def prepare_level1_data(df, sample_size=None, test_size=0.2, random_state=42):
    """
    Prepares data for Level 1 (Binary: BENIGN vs ATTACK).
    - Features: selected numeric features
    - Target: 0 for BENIGN, 1 for ATTACK
    - Ensures ATTACK class includes samples across all attack types in the dataset.
    - Uses stratified train/test split.
    """
    label_col = find_label_column(df.columns)
    if not label_col:
        raise ValueError("Target label column ('Label') was not found in dataset!")
        
    df_clean = df.copy()
    
    # Create binary label column: BENIGN -> 0, Everything else -> 1
    df_clean["Binary_Label"] = df_clean[label_col].apply(
        lambda x: 0 if str(x).upper() == "BENIGN" else 1
    )
    
    # Handle sample sizing safely with stratified balance
    if sample_size and sample_size < len(df_clean):
        class_counts = df_clean["Binary_Label"].value_counts()
        if len(class_counts) > 1 and class_counts.min() >= 2:
            target_per_class = int(sample_size / 2)
            df_clean = df_clean.groupby("Binary_Label", group_keys=False).apply(
                lambda x: x.sample(min(len(x), target_per_class), random_state=random_state)
            ).reset_index(drop=True)
        else:
            df_clean = df_clean.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            
    # Feature columns: numeric columns excluding target and label
    feature_cols = [
        col for col in df_clean.select_dtypes(include=[np.number]).columns
        if col not in [label_col, "Binary_Label"]
    ]
    
    # Remove constant columns (zero variance)
    feature_cols = [col for col in feature_cols if df_clean[col].std() > 0]
    
    X = df_clean[feature_cols]
    y = df_clean["Binary_Label"]
    
    stratify = y if (len(y.unique()) > 1 and y.value_counts().min() >= 2) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test, feature_cols

def prepare_level2_data(df, sample_size=None, test_size=0.2, min_samples_per_class=2, random_state=42):
    """
    Prepares data for Level 2 (Multiclass: specific attack types).
    Excludes BENIGN records (attacks only).
    Preserves all attack classes and handles class imbalance appropriately.
    Uses stratified train/test split.
    """
    label_col = find_label_column(df.columns)
    if not label_col:
        raise ValueError("Target label column ('Label') was not found in dataset!")
        
    # Filter attacks only (exclude BENIGN)
    df_attacks = df[df[label_col].astype(str).str.upper() != "BENIGN"].copy()
    
    if len(df_attacks) == 0:
        raise ValueError("No attack traffic found in the dataset! Dataset only contains BENIGN traffic.")
        
    class_counts = df_attacks[label_col].value_counts()
    
    # For rare attack classes with only 1 sample, duplicate so stratified split works
    for cls_name, cnt in class_counts.items():
        if cnt < min_samples_per_class:
            reps = df_attacks[df_attacks[label_col] == cls_name]
            needed = min_samples_per_class - cnt
            df_attacks = pd.concat([df_attacks] + [reps] * needed, ignore_index=True)
            
    valid_classes = sorted(df_attacks[label_col].unique().tolist())
    
    if len(valid_classes) < 2:
        raise ValueError(
            f"Level 2 classification requires at least 2 distinct attack classes. "
            f"Found only {len(valid_classes)} attack class: {valid_classes}"
        )
        
    # Feature columns
    feature_cols = [
        col for col in df_attacks.select_dtypes(include=[np.number]).columns
        if col not in [label_col, "Binary_Label"]
    ]
    feature_cols = [col for col in feature_cols if df_attacks[col].std() > 0]
    
    X = df_attacks[feature_cols]
    y = df_attacks[label_col]
    
    stratify = y if y.value_counts().min() >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test, feature_cols, valid_classes

def prepare_regression_data(df, target_col="Flow Duration", sample_size=None, test_size=0.2, random_state=42):
    """
    Prepares data for Regression Model (predicting a continuous network flow metric).
    - Default target: 'Flow Duration'
    - Excludes target column and non-numeric/label columns from features
    - Splits into train and test sets
    """
    df_reg = df.copy()
    label_col = find_label_column(df_reg.columns)
    
    # Verify target column exists
    if target_col not in df_reg.columns:
        num_cols = df_reg.select_dtypes(include=[np.number]).columns.tolist()
        candidates = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Total Length of Fwd Packets", "Packet Length Mean"]
        found = None
        for c in candidates:
            if c in df_reg.columns:
                found = c
                break
        if found:
            target_col = found
        elif len(num_cols) > 0:
            target_col = num_cols[0]
        else:
            raise ValueError("No numeric columns available in dataset for regression target!")
            
    if sample_size and sample_size < len(df_reg):
        df_reg = df_reg.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        
    exclude_cols = [target_col]
    if label_col:
        exclude_cols.append(label_col)
    if "Binary_Label" in df_reg.columns:
        exclude_cols.append("Binary_Label")
        
    feature_cols = [
        col for col in df_reg.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]
    feature_cols = [col for col in feature_cols if df_reg[col].std() > 0]
    
    X = df_reg[feature_cols]
    y = df_reg[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_cols, target_col
