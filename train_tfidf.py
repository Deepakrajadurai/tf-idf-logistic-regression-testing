import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Config
DATA_DIR = Path("Data")
MODEL_DIR = Path("models/tfidf_logreg")
RANDOM_SEED = 42

def train_and_eval(train_df, val_df, vectorizer_config, config_name):
    log.info(f"\n--- Training Configuration: {config_name} ---")
    start_time = time.time()
    
    # 1. TF-IDF Vectorization
    log.info("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(**vectorizer_config)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    log.info(f"TF-IDF shape: Train={X_train.shape}, Val={X_val.shape}")
    log.info(f"TF-IDF extraction took {time.time() - start_time:.1f}s")
    
    y_train = train_df["label"]
    y_val = val_df["label"]
    
    # 2. Logistic Regression tuning
    best_c = 1.0
    best_f1 = 0.0
    best_model = None
    
    c_candidates = [0.1, 1.0, 10.0]
    for c in c_candidates:
        clf_start = time.time()
        log.info(f"Training Logistic Regression (C={c})...")
        clf = LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs", n_jobs=-1)
        clf.fit(X_train, y_train)
        
        y_val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average="macro")
        log.info(f"  Validation: C={c} -> Accuracy={acc:.4f}, Macro F1={f1:.4f} (took {time.time() - clf_start:.1f}s)")
        
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
            best_model = clf
            
    log.info(f"Configuration {config_name} Completed. Best C={best_c} -> Macro F1={best_f1:.4f}")
    return vectorizer, best_model, best_f1

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    log.info("Loading train_tfidf.csv and val_tfidf.csv...")
    train_df = pd.read_csv(DATA_DIR / "train_tfidf.csv", dtype={"text": str, "label": int}).dropna(subset=["text"])
    val_df = pd.read_csv(DATA_DIR / "val_tfidf.csv", dtype={"text": str, "label": int}).dropna(subset=["text"])
    
    configs = {
        "Word-level n-grams (1-2)": {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "max_features": 120_000,
            "sublinear_tf": True,
            "min_df": 2
        },
        "Char-level n-grams (3-5)": {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "max_features": 150_000,
            "sublinear_tf": True,
            "min_df": 2
        }
    }
    
    best_overall_f1 = 0.0
    best_overall_vec = None
    best_overall_model = None
    best_overall_name = ""
    
    for name, config in configs.items():
        vec, model, f1 = train_and_eval(train_df, val_df, config, name)
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_vec = vec
            best_overall_model = model
            best_overall_name = name
            
    log.info(f"\n==============================================")
    log.info(f"BEST CONFIGURATION: {best_overall_name} with F1={best_overall_f1:.4f}")
    log.info(f"==============================================")
    
    # Save the best model and vectorizer
    log.info("Saving best model and vectorizer to models/tfidf_logreg/ ...")
    joblib.dump(best_overall_vec, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(best_overall_model, MODEL_DIR / "classifier.joblib")
    
    # Save a config text file to log what configuration was selected
    with open(MODEL_DIR / "model_config.txt", "w", encoding="utf-8") as f:
        f.write(f"selected_config: {best_overall_name}\n")
        f.write(f"best_val_macro_f1: {best_overall_f1:.6f}\n")
        f.write(f"best_C: {best_overall_model.C}\n")
        f.write(f"vectorizer_params: {best_overall_vec.get_params()}\n")
        
    log.info("Training complete and best model saved successfully.")

if __name__ == "__main__":
    main()
