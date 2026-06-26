# TF-IDF + Logistic Regression Evaluation Report

This report summarizes the performance, domain generalization, calibration, and feature importances of the TF-IDF + Logistic Regression model.

- **Selected Pipeline Config**: `Word-level n-grams (1-2)`
- **Test Sample Size**: 50,000 rows (exactly balanced: 25,000 Human / 25,000 AI)

---

## 📈 Overall Model Performance

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | 100.00% | Proportion of correctly classified texts |
| **Macro F1** | 100.00% | Balanced metric accounting for class representation |
| **ROC-AUC** | 1.0000 | Classifier discrimination ability |
| **ECE (Expected Calibration Error)** | 0.012462 | Probability prediction calibration (closer to 0 is better) |
| **Brier Score** | 0.000230 | Mean squared error of probability forecasts |
| **Average Inference Latency** | 0.028 ms | CPU latency per text |


### 🤖 GBERT-Large Performance Reference (From prior runs)
| Model Setup | Accuracy | Macro F1 | ROC-AUC | ECE (10 bins) | Brier Score |
|---|---|---|---|---|---|
| Original | 58.70% | 57.38% | 0.6718 | 0.408622 | 0.408989 |
| Rotation 1 | 83.67% | 60.86% | 0.8692 | 0.160050 | 0.157452 |
| Rotation 2 | 94.67% | 90.69% | 0.9932 | 0.039817 | 0.045541 |



---

## 🎯 Accuracy Breakdown by Text Source / Domain

| Source / Domain | Class | Accuracy | Count |
| :--- | :--- | :--- | :--- |
| Unknown | AI | 100.00% | 50,000 |

---

## 🤖 Accuracy Breakdown by AI Generator Model

| AI Model | Accuracy | Test Count |
| :--- | :--- | :--- |
| gemini-1.5-flash | 100.00% | 8,929 |
| gemma2-9b-it | 100.00% | 2,420 |
| llama3 | 100.00% | 2,070 |
| llama3-70b-8192 | 100.00% | 2,458 |
| mistral | 100.00% | 2,154 |
| mistralai/Mistral-7B-Instruct-v0.3 | 100.00% | 2,482 |
| mixtral-8x7b-32768 | 100.00% | 2,401 |
| phi3 | 100.00% | 2,086 |

---

## 🔍 Interpretability: Top Predictive Stylistic Markers

The table below shows the top 15 stylistic indicators (n-grams) for both class predictions, along with their coefficients.

### 🤖 AI-Generated Indicators (Top 15 Positive Coefficients)
These features (words or characters) are the strongest markers of AI-generated text. Positive values increase the log-odds of a text being classified as AI.

| Rank | Feature | Coefficient | Type |
| :---: | :--- | :---: | :---: |
| 1 | `direkt` | 14.1918 | AI |
| 2 | `bezüglich` | 9.5603 | AI |
| 3 | `von` | 8.5070 | AI |
| 4 | `unter` | 8.4854 | AI |
| 5 | `abs` | 6.2211 | AI |
| 6 | `drucksache` | 5.9491 | AI |
| 7 | `az` | 5.4990 | AI |
| 8 | `aufgrund der` | 5.4146 | AI |
| 9 | `aktuellen` | 5.3561 | AI |
| 10 | `aufgrund` | 5.3143 | AI |
| 11 | `der aktuellen` | 5.2350 | AI |
| 12 | `aktuellen lage` | 5.1330 | AI |
| 13 | `bereich` | 5.0404 | AI |
| 14 | `lage` | 4.8754 | AI |
| 15 | `heutigen` | 4.7088 | AI |


### 👤 Human-Written Indicators (Top 15 Negative Coefficients)
These features are the strongest markers of Human-written German. More negative values decrease the log-odds of a text being classified as AI.

| Rank | Feature | Coefficient | Type |
| :---: | :--- | :---: | :---: |
| 1 | `auch` | -4.6290 | Human |
| 2 | `ist` | -3.7079 | Human |
| 3 | `es` | -3.5462 | Human |
| 4 | `sie` | -3.4439 | Human |
| 5 | `ich` | -2.9813 | Human |
| 6 | `haben` | -2.8130 | Human |
| 7 | `aber` | -2.4611 | Human |
| 8 | `das` | -2.2618 | Human |
| 9 | `nicht` | -2.2385 | Human |
| 10 | `und` | -2.1737 | Human |
| 11 | `aus` | -2.1199 | Human |
| 12 | `mit` | -2.0868 | Human |
| 13 | `oder` | -2.0850 | Human |
| 14 | `ein` | -2.0092 | Human |
| 15 | `wir` | -1.9731 | Human |
