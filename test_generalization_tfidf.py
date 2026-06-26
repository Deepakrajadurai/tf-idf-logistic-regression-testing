import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import html
import csv
import urllib.request
import urllib.parse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading TF-IDF + Logistic Regression detector...")
vectorizer = joblib.load("models/tfidf_logreg/vectorizer.joblib")
classifier = joblib.load("models/tfidf_logreg/classifier.joblib")

# Helper to fetch Wikipedia paragraphs
def get_wiki_paragraphs(title, num_paragraphs=15):
    url = f"https://de.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
        paragraphs = re.findall(r'<p\b[^>]*>(.*?)</p>', html_content, re.DOTALL)
        cleaned = []
        for p in paragraphs:
            p_clean = re.sub(r'<.*?>', '', p)
            p_clean = html.unescape(p_clean)
            p_clean = re.sub(r'\[\d+\]', '', p_clean)
            p_clean = re.sub(r'\s+', ' ', p_clean).strip()
            if p_clean.count('|') > 2:
                continue
            if len(p_clean.split()) >= 30:
                cleaned.append(p_clean)
                if len(cleaned) >= num_paragraphs:
                    break
        return cleaned
    except Exception as e:
        print(f"Error fetching wiki '{title}': {e}")
        return []

# 1. Compile Human Samples
print("\n=== Compiling Human Samples ===")

# Bundestag Speeches (from human CSV, skip first 500k rows)
print("Loading Bundestag speeches...")
bundestag_speeches = []
with open("Data/Human_model_ready_dataset.csv", mode="r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 500000: # skip first 500k rows
            bundestag_speeches.append(row["text"])
            if len(bundestag_speeches) >= 15:
                break

print(f"Loaded {len(bundestag_speeches)} Bundestag speeches.")

# German News / Wiki paragraphs
print("Loading News/Wiki paragraphs...")
news_paragraphs = get_wiki_paragraphs("Klimawandel", 8) + get_wiki_paragraphs("Deutsche_Wirtschaft", 7)
print(f"Loaded {len(news_paragraphs)} News/Wiki paragraphs.")

# Legal / Constitutional paragraphs
print("Loading Legal/Constitutional paragraphs...")
legal_paragraphs = get_wiki_paragraphs("Grundgesetz_für_die_Bundesrepublik_Deutschland", 8) + get_wiki_paragraphs("Bürgerliches_Gesetzbuch", 7)
print(f"Loaded {len(legal_paragraphs)} Legal/Constitutional paragraphs.")

# 2. Compile AI Samples
print("\n=== Compiling AI Samples via local Qwen (simulating models) ===")
print(f"Loading local LLM {QWEN_MODEL}...")
llm_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
llm_tokenizer.padding_side = 'left'
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL).to(DEVICE)
llm_model.eval()

# Select 15 human speeches to rewrite
human_seeds = bundestag_speeches[:15]

# Styles to simulate
styles = {
    "ChatGPT": "Schreibe den folgenden Text im typischen Stil von ChatGPT (GPT-4) um. Behalte den Sinn bei, aber mache es flüssig, strukturiert und verständlich. Schreibe nur den umgeschriebenen Text auf Deutsch und sonst nichts:\n\n",
    "Claude": "Schreibe den folgenden Text im typischen Stil von Claude 3.5 Sonnet um. Behalte den Sinn bei, formuliere präzise, nuanciert und sachlich. Schreibe nur den umgeschriebenen Text auf Deutsch und sonst nichts:\n\n",
    "Gemini": "Schreibe den folgenden Text im typischen Stil von Google Gemini um. Behalte den Sinn bei, formuliere direkt, lebendig und klar. Schreibe nur den umgeschriebenen Text auf Deutsch und sonst nichts:\n\n",
    "Qwen": "Schreibe den folgenden Text im typischen Stil von Qwen um. Behalte den Sinn bei, formuliere detailorientiert, neutral und klar. Schreibe nur den umgeschriebenen Text auf Deutsch und sonst nichts:\n\n"
}

ai_dataset = {style_name: [] for style_name in styles}

for style_name, prefix in styles.items():
    print(f"Generating style '{style_name}' in batch...")
    prompts = []
    for text in human_seeds:
        prompt_text = prefix + text
        messages = [
            {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
            {"role": "user", "content": prompt_text}
        ]
        formatted_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)
    
    inputs = llm_tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = llm_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    responses = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    ai_dataset[style_name] = [r.strip() for r in responses]

# Delete LLM model to free GPU memory
del llm_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 3. Evaluate TF-IDF Classifier
print("\n=== Running TF-IDF + Logistic Regression Classifier on Unseen Data ===")

def predict_list(texts):
    results = []
    # Vectorize
    X = vectorizer.transform(texts)
    # Predict
    preds = classifier.predict(X)
    return preds.tolist()

test_groups = {
    "Bundestag Speeches (Human)": (bundestag_speeches, 0),
    "German News/Wiki (Human)": (news_paragraphs, 0),
    "Legal/Constitutional (Human)": (legal_paragraphs, 0),
    "ChatGPT style (AI)": (ai_dataset["ChatGPT"], 1),
    "Gemini style (AI)": (ai_dataset["Gemini"], 1),
    "Claude style (AI)": (ai_dataset["Claude"], 1),
    "Qwen style (AI)": (ai_dataset["Qwen"], 1),
}

print("\n" + "=" * 60)
print(f"{'Source':<35} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<8}")
print("-" * 60)

summary_lines = []
for name, (texts, true_label) in test_groups.items():
    preds = predict_list(texts)
    correct = sum(1 for p in preds if p == true_label)
    acc = correct / len(texts) if len(texts) > 0 else 0
    line = f"{name:<35} | {len(texts):<8} | {correct:<8} | {acc * 100:.1f}%"
    print(line)
    summary_lines.append(line)

print("=" * 60)

# Save result to results folder
with open("results/tfidf_generalization_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print("Generalization report saved to results/tfidf_generalization_report.txt")
