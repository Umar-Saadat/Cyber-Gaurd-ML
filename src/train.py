# train.py - Model training for Level 1, Level 2, and Regression
import time
import os
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.utils import MODELS_DIR

# Dictionary of supported classification models with class imbalance handling
AVAILABLE_MODELS = {
    "Logistic Regression": {
        "class": LogisticRegression,
        "needs_scaling": True,
        "default_params": {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"}
    },
    "K-Nearest Neighbors (KNN)": {
        "class": KNeighborsClassifier,
        "needs_scaling": True,
        "default_params": {"n_neighbors": 5}
    },
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "needs_scaling": True,
        "default_params": {"random_state": 42, "max_depth": 15, "class_weight": "balanced"}
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "needs_scaling": True,
        "default_params": {"n_estimators": 60, "random_state": 42, "max_depth": 15, "class_weight": "balanced"}
    },
    "Support Vector Machine (SVM)": {
        "class": SVC,
        "needs_scaling": True,
        "default_params": {"kernel": "rbf", "random_state": 42, "probability": True, "class_weight": "balanced"}
    }
}

# Dictionary of supported regression models
AVAILABLE_REGRESSION_MODELS = {
    "Linear Regression": {
        "class": LinearRegression,
        "needs_scaling": True,
        "default_params": {}
    },
    "Ridge Regression": {
        "class": Ridge,
        "needs_scaling": True,
        "default_params": {"random_state": 42}
    },
    "Decision Tree Regressor": {
        "class": DecisionTreeRegressor,
        "needs_scaling": True,
        "default_params": {"random_state": 42, "max_depth": 12}
    },
    "Random Forest Regressor": {
        "class": RandomForestRegressor,
        "needs_scaling": True,
        "default_params": {"n_estimators": 40, "random_state": 42, "max_depth": 12}
    }
}

def build_model_pipeline(model_name, model_kwargs=None):
    """
    Creates a scikit-learn Pipeline with SimpleImputer, StandardScaler,
    and the chosen algorithm (classification or regression).
    StandardScaler is preserved across all models so the prediction pipeline
    uses the exact same preprocessing and feature scaling.
    """
    if model_name in AVAILABLE_MODELS:
        config = AVAILABLE_MODELS[model_name]
        step_name = "classifier"
    elif model_name in AVAILABLE_REGRESSION_MODELS:
        config = AVAILABLE_REGRESSION_MODELS[model_name]
        step_name = "regressor"
    else:
        all_avail = list(AVAILABLE_MODELS.keys()) + list(AVAILABLE_REGRESSION_MODELS.keys())
        raise ValueError(f"Unknown model name: {model_name}. Available: {all_avail}")
        
    model_cls = config["class"]
    params = config["default_params"].copy()
    if model_kwargs:
        params.update(model_kwargs)
        
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (step_name, model_cls(**params))
    ]
    return Pipeline(steps)

def train_single_model(model_name, X_train, y_train, model_kwargs=None):
    """
    Trains a single model pipeline on training data.
    Prints training metrics and verifies that all target classes were learned.
    Returns: fitted pipeline, training duration (seconds), training accuracy
    """
    pipeline = build_model_pipeline(model_name, model_kwargs)
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = round(time.time() - start_time, 2)
    
    # Calculate training accuracy
    train_acc = None
    if model_name in AVAILABLE_MODELS:
        y_train_pred = pipeline.predict(X_train)
        train_acc = round(accuracy_score(y_train, y_train_pred), 4)
        
        # Check that every available attack class is learned by the model
        classifier_step = pipeline.named_steps["classifier"]
        learned_classes = list(getattr(classifier_step, "classes_", []))
        expected_classes = sorted(list(set(y_train)))
        
        print(f"\n[Train] Model '{model_name}' trained in {training_time}s")
        print(f"[Train] Training Accuracy: {train_acc * 100:.2f}%")
        print(f"[Train] Expected Classes ({len(expected_classes)}): {expected_classes}")
        print(f"[Train] Learned Classes  ({len(learned_classes)}): {learned_classes}")
        
        missing_classes = set(expected_classes) - set(learned_classes)
        if missing_classes:
            print(f"[Train] ⚠️ WARNING: Model missed classes: {missing_classes}")
        else:
            print(f"[Train] ✅ ALL {len(learned_classes)} attack classes successfully learned by the model!")
    else:
        print(f"\n[Train] Regression Model '{model_name}' trained in {training_time}s on {len(X_train)} samples.")
        
    return pipeline, training_time, train_acc

def save_trained_model(pipeline, model_name, level_name, feature_names, class_names=None, target_col=None, train_acc=None):
    """
    Saves trained pipeline, scaler, and its metadata (feature names, classes, target_col) to disk.
    level_name: 'level1', 'level2', or 'regression'
    """
    clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filename = f"{level_name}_{clean_name}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Extract fitted scaler if present
    scaler_step = pipeline.named_steps.get("scaler")
    
    metadata = {
        "model_name": model_name,
        "level": level_name,
        "feature_names": list(feature_names),
        "class_names": list(class_names) if class_names is not None else None,
        "target_col": target_col,
        "train_accuracy": train_acc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    package = {
        "pipeline": pipeline,
        "scaler": scaler_step,
        "metadata": metadata
    }
    
    joblib.dump(package, filepath)
    print(f"[Train] Saved trained model package to {filepath} (Features: {len(feature_names)}, Classes: {len(class_names) if class_names else 'N/A'})")
    return filepath, metadata

def load_saved_model(filepath):
    """
    Loads saved model package containing pipeline, scaler, and metadata.
    Accepts full path or filename inside models/ directory.
    """
    if not os.path.exists(filepath):
        candidate = os.path.join(MODELS_DIR, filepath)
        if os.path.exists(candidate):
            filepath = candidate
        else:
            return None
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"[Train] Error loading {filepath}: {e}")
        return None

def list_saved_models(level_name=None):
    """
    Returns list of saved models in the models directory.
    """
    saved = []
    if not os.path.exists(MODELS_DIR):
        return saved
        
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".joblib"):
            if level_name and not fname.startswith(level_name):
                continue
            path = os.path.join(MODELS_DIR, fname)
            try:
                data = joblib.load(path)
                saved.append({
                    "filename": fname,
                    "filepath": path,
                    "metadata": data.get("metadata", {})
                })
            except Exception:
                pass
    return saved
