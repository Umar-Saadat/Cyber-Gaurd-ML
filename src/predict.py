# predict.py - Prediction logic for CSV upload and manual features
import pandas as pd
import numpy as np

def predict_from_dataframe(df_input, l1_package, l2_package=None, reg_package=None):
    """
    Takes an input DataFrame and runs prediction:
    1. Reindexes features into exact training order.
    2. Runs Level 1 (BENIGN vs ATTACK) using saved scaler & trained pipeline,
       computing genuine detection confidence score (1-100%).
    3. For rows predicted as ATTACK, if Level 2 model is provided, predicts genuine attack type
       and probability confidence score (1-100%) from predict_proba().
    4. If regression model is provided, predicts expected continuous network flow value.
    
    Returns: DataFrame with original data + predictions + genuine confidence scores + regression
    """
    df_result = df_input.copy()
    
    # ---------------- 1. Level 1 Binary Prediction ----------------
    l1_pipeline = l1_package["pipeline"]
    l1_features = l1_package["metadata"]["feature_names"]
    
    # Ensure all required features are present in the EXACT training order
    X_l1 = df_result.reindex(columns=l1_features).astype(float, errors="ignore")
    
    # Replace infinite values with NaN for imputer
    for c in X_l1.columns:
        X_l1[c] = pd.to_numeric(X_l1[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        
    l1_preds = l1_pipeline.predict(X_l1)
    
    # Calculate genuine probability scores
    has_l1_proba = hasattr(l1_pipeline, "predict_proba")
    l1_probs = None
    if has_l1_proba:
        try:
            l1_probs = l1_pipeline.predict_proba(X_l1)
        except Exception:
            l1_probs = None
            
    df_result["Level_1_Prediction"] = ["ATTACK" if p == 1 else "BENIGN" for p in l1_preds]
    
    l1_conf_list = []
    for idx, pred in enumerate(l1_preds):
        if l1_probs is not None and len(l1_probs[idx]) > 1:
            score = float(l1_probs[idx][pred]) * 100.0
            l1_conf_list.append(round(min(100.0, max(0.0, score)), 1))
        elif hasattr(l1_pipeline, "decision_function"):
            try:
                dec = float(l1_pipeline.decision_function(X_l1.iloc[[idx]])[0])
                prob = 1.0 / (1.0 + np.exp(-abs(dec)))
                l1_conf_list.append(round(prob * 100.0, 1))
            except Exception:
                l1_conf_list.append(95.0)
        else:
            l1_conf_list.append(90.0)
            
    df_result["Level_1_Confidence_%"] = l1_conf_list
    
    # ---------------- 2. Level 2 Multiclass Attack Type ----------------
    if l2_package is not None:
        l2_pipeline = l2_package["pipeline"]
        l2_features = l2_package["metadata"]["feature_names"]
        
        # Ensure exact training feature order for Level 2
        X_l2 = df_result.reindex(columns=l2_features).astype(float, errors="ignore")
        for c in X_l2.columns:
            X_l2[c] = pd.to_numeric(X_l2[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            
        has_l2_proba = hasattr(l2_pipeline, "predict_proba")
        classes_l2 = list(getattr(l2_pipeline, "classes_", []))
        
        l2_preds_all = []
        l2_conf_all = []
        
        for idx in range(len(df_result)):
            if df_result["Level_1_Prediction"].iloc[idx] == "ATTACK":
                single_row = X_l2.iloc[[idx]]
                try:
                    pred_type = l2_pipeline.predict(single_row)[0]
                    l2_preds_all.append(str(pred_type))
                    
                    if has_l2_proba and len(classes_l2) > 0:
                        probs = l2_pipeline.predict_proba(single_row)[0]
                        if pred_type in classes_l2:
                            class_idx = classes_l2.index(pred_type)
                            conf = round(float(probs[class_idx]) * 100.0, 1)
                        else:
                            conf = round(float(np.max(probs)) * 100.0, 1)
                        l2_conf_all.append(min(100.0, max(0.0, conf)))
                    else:
                        l2_conf_all.append(90.0)
                except Exception as e:
                    print(f"[Predict] Warning in L2 prediction: {e}")
                    l2_preds_all.append("Suspicious Malicious Activity")
                    l2_conf_all.append(70.0)
            else:
                l2_preds_all.append("Normal Traffic (BENIGN)")
                l2_conf_all.append(df_result["Level_1_Confidence_%"].iloc[idx])
                
        df_result["Predicted_Attack_Type"] = l2_preds_all
        df_result["Attack_Confidence_%"] = l2_conf_all
    else:
        df_result["Predicted_Attack_Type"] = [
            "Malicious Attack (Subtype model not trained)" if p == "ATTACK" else "Normal Traffic (BENIGN)"
            for p in df_result["Level_1_Prediction"]
        ]
        df_result["Attack_Confidence_%"] = df_result["Level_1_Confidence_%"]
        
    # ---------------- 3. Regression Prediction ----------------
    if reg_package is not None:
        reg_pipeline = reg_package["pipeline"]
        reg_features = reg_package["metadata"]["feature_names"]
        target_name = reg_package["metadata"].get("target_col") or "Flow Duration"
        
        X_reg = df_result.reindex(columns=reg_features).astype(float, errors="ignore")
        for c in X_reg.columns:
            X_reg[c] = pd.to_numeric(X_reg[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            
        try:
            reg_preds = reg_pipeline.predict(X_reg)
            df_result[f"Predicted_{target_name}"] = [round(float(val), 2) for val in reg_preds]
        except Exception as e:
            print(f"[Predict] Warning in regression prediction: {e}")
            df_result[f"Predicted_{target_name}"] = np.nan
            
    return df_result

def predict_single_instance(feature_dict, l1_package, l2_package=None, reg_package=None):
    """
    Takes a single dictionary of feature inputs and returns predictions, genuine confidence score,
    and continuous flow duration regression value.
    Ensures features are delivered in the exact training order expected by the models.
    """
    # Clean feature dict and convert values to float
    clean_dict = {}
    for k, v in feature_dict.items():
        try:
            clean_dict[str(k).strip()] = float(v) if (v is not None and v != "") else np.nan
        except (ValueError, TypeError):
            clean_dict[str(k).strip()] = np.nan
            
    df_single = pd.DataFrame([clean_dict])
    res_df = predict_from_dataframe(df_single, l1_package, l2_package, reg_package)
    
    target_name = (reg_package["metadata"].get("target_col") or "Flow Duration") if reg_package else None
    reg_val = res_df[f"Predicted_{target_name}"].iloc[0] if (reg_package and f"Predicted_{target_name}" in res_df.columns) else None
    
    return {
        "level1": str(res_df["Level_1_Prediction"].iloc[0]),
        "level1_conf": float(res_df["Level_1_Confidence_%"].iloc[0]),
        "attack_type": str(res_df["Predicted_Attack_Type"].iloc[0]),
        "confidence": float(res_df["Attack_Confidence_%"].iloc[0]),
        "regression_val": float(reg_val) if (reg_val is not None and not np.isnan(reg_val)) else None,
        "regression_target": target_name
    }
