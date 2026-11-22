# data_loader.py - Functions to scan and load CSV dataset files
import os
import json
import numpy as np
import pandas as pd
from src.utils import (
    MODELS_DIR,
    get_dataset_files,
    clean_column_names,
    find_label_column,
    normalize_attack_label,
    UI_FEATURE_MAPPING
)

SAMPLES_CACHE_FILE = os.path.join(MODELS_DIR, "attack_samples.json")

def scan_dataset_folder():
    """
    Returns information about CSV files found in the dataset folder.
    Returns:
        dict: {
            'file_count': int,
            'files': list of file info dicts (name, path, size_mb)
        }
    """
    file_paths = get_dataset_files()
    file_details = []
    
    for path in file_paths:
        size_bytes = os.path.getsize(path)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        file_details.append({
            "name": os.path.basename(path),
            "path": path,
            "size_mb": size_mb
        })
        
    return {
        "file_count": len(file_details),
        "files": file_details
    }

def load_data(file_paths=None, max_rows=None, sample_fraction=None, random_state=42):
    """
    Loads dataset from one or more CSV files.
    Normalizes labels and trims column names.
    """
    if file_paths is None or len(file_paths) == 0:
        file_paths = get_dataset_files()
        
    if not file_paths:
        return None
        
    dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
            
        try:
            if max_rows is not None and max_rows > 0:
                df_temp = pd.read_csv(file_path, nrows=max_rows, low_memory=False)
            else:
                df_temp = pd.read_csv(file_path, low_memory=False)
                
            df_temp = clean_column_names(df_temp)
            label_col = find_label_column(df_temp.columns)
            if label_col:
                df_temp[label_col] = df_temp[label_col].apply(normalize_attack_label)
                
            dfs.append(df_temp)
        except Exception as e:
            print(f"[DataLoader] Warning: could not load {file_path}: {e}")
            continue
            
    if not dfs:
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    if sample_fraction is not None and 0 < sample_fraction < 1.0:
        combined_df = combined_df.sample(frac=sample_fraction, random_state=random_state).reset_index(drop=True)
        
    return combined_df

def load_balanced_training_data(file_paths=None, sample_size=5000, level="level1", random_state=42):
    """
    Reads ALL CSV files available in the dataset folder, combines compatible files,
    normalizes labels, and handles severe class imbalance so that ALL attack types
    (including rare ones like Heartbleed, SQL Injection, and Infiltration) are preserved
    and learned by the model.
    """
    if file_paths is None or len(file_paths) == 0:
        file_paths = get_dataset_files()
        
    if not file_paths:
        raise ValueError("No CSV files found in dataset folder!")
        
    print(f"\n[DataLoader] Loading and combining {len(file_paths)} dataset files for {level.upper()}...")
    
    chunks = []
    total_raw_rows = 0
    
    for fpath in file_paths:
        if not os.path.exists(fpath):
            continue
            
        try:
            df_raw = pd.read_csv(fpath, low_memory=False)
            df_raw = clean_column_names(df_raw)
            label_col = find_label_column(df_raw.columns)
            if not label_col:
                continue
                
            df_raw[label_col] = df_raw[label_col].apply(normalize_attack_label)
            total_raw_rows += len(df_raw)
            
            if level == "level2":
                # Attacks only (multiclass)
                df_slice = df_raw[df_raw[label_col] != "BENIGN"].copy()
                if not df_slice.empty:
                    # Balanced sample per attack class in this file
                    # Keep all records for rare classes; cap large classes
                    per_class_cap = max(100, int(sample_size / 5)) if sample_size else 1000
                    sampled_slice = df_slice.groupby(label_col, group_keys=False).apply(
                        lambda grp: grp.sample(min(len(grp), per_class_cap), random_state=random_state)
                    )
                    chunks.append(sampled_slice)
            elif level == "level1":
                # Binary: BENIGN vs ATTACK
                # Sample both BENIGN and ATTACKs from each file
                df_benign = df_raw[df_raw[label_col] == "BENIGN"]
                df_attacks = df_raw[df_raw[label_col] != "BENIGN"]
                
                parts = []
                if not df_attacks.empty:
                    # Keep attack samples across all subtypes
                    att_cap = max(100, int(sample_size / 8)) if sample_size else 1000
                    sampled_att = df_attacks.groupby(label_col, group_keys=False).apply(
                        lambda grp: grp.sample(min(len(grp), att_cap), random_state=random_state)
                    )
                    parts.append(sampled_att)
                    
                if not df_benign.empty:
                    ben_cap = max(200, int(sample_size / len(file_paths))) if sample_size else 1500
                    sampled_ben = df_benign.sample(min(len(df_benign), ben_cap), random_state=random_state)
                    parts.append(sampled_ben)
                    
                if parts:
                    chunks.append(pd.concat(parts, ignore_index=True))
            else:
                # Regression: representative flow records from each file
                slice_cap = max(200, int(sample_size / len(file_paths))) if sample_size else 2000
                chunks.append(df_raw.sample(min(len(df_raw), slice_cap), random_state=random_state))
                
        except Exception as e:
            print(f"[DataLoader] Warning processing {os.path.basename(fpath)}: {e}")
            continue
            
    if not chunks:
        raise ValueError("Failed to extract data chunks from dataset files!")
        
    combined_df = pd.concat(chunks, ignore_index=True)
    label_col = find_label_column(combined_df.columns)
    
    if label_col:
        dist = combined_df[label_col].value_counts().to_dict()
        print(f"[DataLoader] Successfully combined dataset: {len(combined_df):,} records from {total_raw_rows:,} raw rows.")
        print(f"[DataLoader] Class distribution ({len(dist)} classes): {dist}")
    
    return combined_df

def extract_attack_samples(file_paths=None, save_cache=True):
    """
    Extracts representative, realistic sample feature values for BENIGN and ALL
    available attack types across the dataset files.
    Returns:
        dict: {
            "success": True,
            "attack_types": list of attack class names,
            "samples": {
                "<class_name>": {
                    "ui_values": { "feat-port": val, "Destination Port": val, ... },
                    "full_values": { ... all numeric features ... }
                }
            }
        }
    """
    if file_paths is None or len(file_paths) == 0:
        file_paths = get_dataset_files()
        
    ui_fields = list(UI_FEATURE_MAPPING.values())
    samples = {}
    
    print("[DataLoader] Extracting realistic attack samples across all dataset files...")
    
    for fpath in file_paths:
        if not os.path.exists(fpath):
            continue
            
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df = clean_column_names(df)
            label_col = find_label_column(df.columns)
            if not label_col:
                continue
                
            df[label_col] = df[label_col].apply(normalize_attack_label)
            
            # Clean numeric columns for sample extraction
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
            for cls_name, grp in df.groupby(label_col):
                if cls_name not in samples:
                    clean_grp = grp.dropna(subset=[c for c in ui_fields if c in grp.columns])
                    if not clean_grp.empty:
                        # Choose median-index row to avoid extreme outliers
                        row = clean_grp.iloc[len(clean_grp) // 2]
                        
                        ui_vals = {}
                        for ui_id, feat_name in UI_FEATURE_MAPPING.items():
                            val = float(row[feat_name]) if feat_name in row and not pd.isna(row[feat_name]) else 0.0
                            ui_vals[ui_id] = val
                            ui_vals[feat_name] = val
                            
                        full_vals = {}
                        for c in numeric_cols:
                            if c in row and not pd.isna(row[c]):
                                full_vals[c] = float(row[c])
                                
                        samples[cls_name] = {
                            "ui_values": ui_vals,
                            "full_values": full_vals
                        }
        except Exception as e:
            print(f"[DataLoader] Warning while extracting samples from {os.path.basename(fpath)}: {e}")
            continue
            
    # Sort attack types with BENIGN first, then alphabetical
    attack_types = sorted(list(samples.keys()))
    if "BENIGN" in attack_types:
        attack_types.remove("BENIGN")
        attack_types = ["BENIGN"] + attack_types
        
    result = {
        "success": True,
        "attack_types": attack_types,
        "samples": samples
    }
    
    if save_cache and samples:
        try:
            with open(SAMPLES_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"[DataLoader] Saved realistic sample profiles to {SAMPLES_CACHE_FILE}")
        except Exception as e:
            print(f"[DataLoader] Warning caching samples: {e}")
            
    return result

def get_attack_samples(refresh=False):
    """
    Returns realistic sample profiles. Loads from cache if available,
    otherwise extracts from dataset.
    """
    if not refresh and os.path.exists(SAMPLES_CACHE_FILE):
        try:
            with open(SAMPLES_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached.get("samples") and len(cached.get("samples")) >= 10:
                    return cached
        except Exception:
            pass
            
    return extract_attack_samples()

def get_dataset_summary(df):
    """
    Calculates summary statistics of the loaded dataset.
    """
    if df is None or df.empty:
        return {}
        
    label_col = find_label_column(df.columns)
    label_counts = {}
    if label_col:
        label_counts = df[label_col].value_counts().to_dict()
        
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "label_column": label_col,
        "label_counts": label_counts,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "missing_values_count": int(df.isnull().sum().sum())
    }
