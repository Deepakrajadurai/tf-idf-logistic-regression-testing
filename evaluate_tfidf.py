import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, brier_score_loss
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
RESULTS_DIR = Path("results")

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            actual_pos_rate = np.mean(y_true[in_bin])
            avg_confidence = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence - actual_pos_rate)
            
    return ece

def get_model_config():
    config = {}
    config_file = MODEL_DIR / "model_config.txt"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    config[k.strip()] = v.strip()
    return config

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 1. Load data & models
    log.info("Loading test dataset and best TF-IDF + Logistic Regression model...")
    test_df = pd.read_csv(DATA_DIR / "test_tfidf.csv", dtype={"text": str, "label": int}).dropna(subset=["text"])
    
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
    classifier = joblib.load(MODEL_DIR / "classifier.joblib")
    
    config = get_model_config()
    selected_config = config.get("selected_config", "Unknown")
    
    # 2. Inference
    start_time = time.time()
    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"].values
    
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    log.info(f"Inference complete in {inference_time:.1f}s (Average: {inference_time / len(test_df) * 1000:.3f} ms per sentence)")
    
    # 3. Overall Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_prob)
    ece = expected_calibration_error(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    log.info(f"Test Accuracy: {acc:.4%}")
    log.info(f"Macro F1 Score: {f1:.4%}")
    log.info(f"ROC-AUC: {auc:.4f}")
    
    # 4. Domain & Model Breakdown
    test_df["pred"] = y_pred
    test_df["prob"] = y_prob
    
    # Domain / Source Breakdown
    domain_results = []
    # Fill in source/domain categories properly
    test_df["domain_clean"] = test_df["source"].fillna("Unknown")
    for group_name, group_df in test_df.groupby("domain_clean"):
        g_y_test = group_df["label"].values
        g_y_pred = group_df["pred"].values
        g_acc = accuracy_score(g_y_test, g_y_pred)
        domain_results.append({
            "Source": group_name,
            "Class": "Human" if group_df["label"].iloc[0] == 0 else "AI",
            "Accuracy": f"{g_acc:.2%}",
            "Count": len(group_df)
        })
    df_domain_report = pd.DataFrame(domain_results)
    
    # AI Model Breakdown (only for AI instances)
    ai_df = test_df[test_df["label"] == 1]
    model_results = []
    if "model" in ai_df.columns:
        for model_name, group_df in ai_df.groupby("model", dropna=False):
            g_y_test = group_df["label"].values
            g_y_pred = group_df["pred"].values
            g_acc = accuracy_score(g_y_test, g_y_pred)
            model_results.append({
                "AI Model": str(model_name),
                "Accuracy": f"{g_acc:.2%}",
                "Count": len(group_df)
            })
    df_model_report = pd.DataFrame(model_results)
    
    # 5. Feature Importances (Coefficients)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    
    # Sort coefficients
    sorted_indices = np.argsort(coefficients)
    
    # Human indicators (negative coefficients)
    top_human_indices = sorted_indices[:50]
    top_human_features = feature_names[top_human_indices]
    top_human_coefs = coefficients[top_human_indices]
    
    # AI indicators (positive coefficients)
    top_ai_indices = sorted_indices[::-1][:50]
    top_ai_features = feature_names[top_ai_indices]
    top_ai_coefs = coefficients[top_ai_indices]
    
    # Save all features coefficients
    df_features = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })
    df_features.sort_values(by="coefficient", ascending=False).to_csv(RESULTS_DIR / "tfidf_feature_importance.csv", index=False, encoding="utf-8")
    log.info("Feature importance saved to results/tfidf_feature_importance.csv")
    
    # 6. Generate comparative report
    # Try reading BERT's results for comparison
    bert_summary = ""
    bert_report_path = RESULTS_DIR / "evaluation_report.md"
    if bert_report_path.exists():
        bert_summary = "\n### 🤖 GBERT-Large Performance Reference (From prior runs)\n"
        try:
            bert_content = bert_report_path.read_text(encoding="utf-8")
            # Extract the overall metrics table
            lines = bert_content.split("\n")
            table_lines = []
            capture = False
            for line in lines:
                if "Model Setup" in line:
                    capture = True
                if capture:
                    table_lines.append(line)
                    if line.strip() == "" and len(table_lines) > 2:
                        break
            bert_summary += "\n".join(table_lines) + "\n"
        except Exception as e:
            log.warning(f"Could not parse GBERT metrics: {e}")
            
    report_md = f"""# TF-IDF + Logistic Regression Evaluation Report

This report summarizes the performance, domain generalization, calibration, and feature importances of the TF-IDF + Logistic Regression model.

- **Selected Pipeline Config**: `{selected_config}`
- **Test Sample Size**: {len(test_df):,} rows (exactly balanced: {len(test_df[test_df['label'] == 0]):,} Human / {len(test_df[test_df['label'] == 1]):,} AI)

---

## 📈 Overall Model Performance

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | {acc:.2%} | Proportion of correctly classified texts |
| **Macro F1** | {f1:.2%} | Balanced metric accounting for class representation |
| **ROC-AUC** | {auc:.4f} | Classifier discrimination ability |
| **ECE (Expected Calibration Error)** | {ece:.6f} | Probability prediction calibration (closer to 0 is better) |
| **Brier Score** | {brier:.6f} | Mean squared error of probability forecasts |
| **Average Inference Latency** | {inference_time / len(test_df) * 1000:.3f} ms | CPU latency per text |

{bert_summary}

---

## 🎯 Accuracy Breakdown by Text Source / Domain

| Source / Domain | Class | Accuracy | Count |
| :--- | :--- | :--- | :--- |
"""
    for _, row in df_domain_report.iterrows():
        report_md += f"| {row['Source']} | {row['Class']} | {row['Accuracy']} | {row['Count']:,} |\n"
        
    report_md += """
---

## 🤖 Accuracy Breakdown by AI Generator Model

| AI Model | Accuracy | Test Count |
| :--- | :--- | :--- |
"""
    for _, row in df_model_report.iterrows():
        report_md += f"| {row['AI Model']} | {row['Accuracy']} | {row['Count']:,} |\n"
        
    report_md += """
---

## 🔍 Interpretability: Top Predictive Stylistic Markers

The table below shows the top 15 stylistic indicators (n-grams) for both class predictions, along with their coefficients.

### 🤖 AI-Generated Indicators (Top 15 Positive Coefficients)
These features (words or characters) are the strongest markers of AI-generated text. Positive values increase the log-odds of a text being classified as AI.

| Rank | Feature | Coefficient | Type |
| :---: | :--- | :---: | :---: |
"""
    for rank in range(15):
        feat = top_ai_features[rank]
        coef = top_ai_coefs[rank]
        # Clean representation of whitespace for char n-grams
        repr_feat = f"`{feat.replace(' ', '•')}`" if "char" in selected_config.lower() else f"`{feat}`"
        report_md += f"| {rank+1} | {repr_feat} | {coef:.4f} | AI |\n"
        
    report_md += """

### 👤 Human-Written Indicators (Top 15 Negative Coefficients)
These features are the strongest markers of Human-written German. More negative values decrease the log-odds of a text being classified as AI.

| Rank | Feature | Coefficient | Type |
| :---: | :--- | :---: | :---: |
"""
    for rank in range(15):
        feat = top_human_features[rank]
        coef = top_human_coefs[rank]
        repr_feat = f"`{feat.replace(' ', '•')}`" if "char" in selected_config.lower() else f"`{feat}`"
        report_md += f"| {rank+1} | {repr_feat} | {coef:.4f} | Human |\n"
        
    report_md_path = RESULTS_DIR / "tfidf_evaluation_report.md"
    report_md_path.write_text(report_md, encoding="utf-8")
    log.info(f"Evaluation report generated successfully at: results/tfidf_evaluation_report.md")

if __name__ == "__main__":
    main()
