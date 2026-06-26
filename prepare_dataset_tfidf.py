import re
import hashlib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Config
RANDOM_SEED = 42
OUTPUT_DIR = Path("Data")

# Target sizes per class (balanced)
TRAIN_SIZE_PER_CLASS = 250_000
VAL_SIZE_PER_CLASS = 25_000
TEST_SIZE_PER_CLASS = 25_000

ARTIFACT_PATTERNS = [
    # Strip Bundestag plenarsitzung metadata
    (r"\d+\.\s*Plenarsitzung", "Plenarsitzung"),
    # Strip political party tags in speeches
    (r"\((?:CDU(?:/CSU)?|SPD|Grüne|FDP|AfD|Linke|BSW|CSU)\)", ""),
    # Strip common speaker templates
    (r"(?:auf Initiative von |Abgeordnet(?:em|er|en)\s+)[A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+", ""),
    # Collapse double spaces
    (r"\s{2,}", " "),
]

def strip_artifacts(text: str) -> str:
    for pattern, replacement in ARTIFACT_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text.strip()

def clean_text(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    text = strip_artifacts(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    # Ensure text is not extremely short or long
    if len(words) < 15:
        return None
    if len(words) > 100:
        text = " ".join(words[:100])

    # Reject if >25% digits
    if sum(c.isdigit() for c in text) / len(text) > 0.25:
        return None

    # Reject model meta-commentary
    bad_phrases = [
        "als ki ", "als sprachmodell", "ich kann leider",
        "gerne helfe ich", "natürlich, hier", "hier sind die",
        "bitte beachten sie", "als assistent",
    ]
    lower = text.lower()
    if any(p in lower for p in bad_phrases):
        return None

    return text

def fingerprint(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()

def load_and_clean(csv_path: Path, label: int) -> pd.DataFrame:
    log.info(f"Loading and cleaning {csv_path} (label={label})...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df["label"] = label
    log.info(f"  Raw rows: {len(df):,}")
    
    # Process text column
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    log.info(f"  After cleaning: {len(df):,}")
    
    # Exact deduplication
    df["_fp"] = df["text"].apply(fingerprint)
    before = len(df)
    df.drop_duplicates(subset=["_fp"], inplace=True)
    df.drop(columns=["_fp"], inplace=True)
    log.info(f"  After exact dedup: {len(df):,} (removed {before - len(df):,})")
    
    return df.reset_index(drop=True)

def length_balanced_sample(human_df: pd.DataFrame, ai_df: pd.DataFrame, target_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample human and AI texts such that their word-count distributions match exactly.
    Uses quantile-based buckets from the AI distribution (which is narrower).
    """
    human_df = human_df.copy()
    ai_df = ai_df.copy()
    
    human_df["_wc"] = human_df["text"].str.split().apply(len)
    ai_df["_wc"] = ai_df["text"].str.split().apply(len)
    
    # Determine bin boundaries using AI word count quantiles
    N_BUCKETS = 10
    boundaries = ai_df["_wc"].quantile(np.linspace(0, 1, N_BUCKETS + 1)).values
    # Ensure unique boundaries
    boundaries = np.unique(boundaries)
    if len(boundaries) < 2:
        boundaries = np.array([0, ai_df["_wc"].max() + 1])
        
    num_bins = len(boundaries) - 1
    per_bucket_target = target_size // num_bins
    
    human_buckets, ai_buckets = [], []
    
    for j in range(num_bins):
        lo, hi = boundaries[j], boundaries[j + 1]
        
        # Select rows in this word count range
        if j == num_bins - 1:
            h_slice = human_df[(human_df["_wc"] >= lo) & (human_df["_wc"] <= hi)]
            a_slice = ai_df[(ai_df["_wc"] >= lo) & (ai_df["_wc"] <= hi)]
        else:
            h_slice = human_df[(human_df["_wc"] >= lo) & (human_df["_wc"] < hi)]
            a_slice = ai_df[(ai_df["_wc"] >= lo) & (ai_df["_wc"] < hi)]
            
        n_sample = min([per_bucket_target, len(h_slice), len(a_slice)])
        
        if n_sample > 0:
            human_buckets.append(h_slice.sample(n=n_sample, random_state=RANDOM_SEED))
            ai_buckets.append(a_slice.sample(n=n_sample, random_state=RANDOM_SEED))
            
    if not human_buckets or not ai_buckets:
        raise ValueError("Could not find overlapping length bins to perform length balancing.")
        
    human_sampled = pd.concat(human_buckets).drop(columns=["_wc"])
    ai_sampled = pd.concat(ai_buckets).drop(columns=["_wc"])
    
    # If we are slightly short of target_size, top up with remaining rows (matching word count distribution as close as possible)
    remaining = target_size - len(human_sampled)
    if remaining > 0:
        already_h = set(human_sampled.index)
        already_a = set(ai_sampled.index)
        h_pool = human_df[~human_df.index.isin(already_h)]
        a_pool = ai_df[~ai_df.index.isin(already_a)]
        
        # Sample randomly from overlap region
        min_pool = min(len(h_pool), len(a_pool), remaining)
        if min_pool > 0:
            human_sampled = pd.concat([human_sampled, h_pool.sample(n=min_pool, random_state=RANDOM_SEED).drop(columns=["_wc"])])
            ai_sampled = pd.concat([ai_sampled, a_pool.sample(n=min_pool, random_state=RANDOM_SEED).drop(columns=["_wc"])])
            
    log.info(f"  Length-balanced sampling result: Human={len(human_sampled):,}, AI={len(ai_sampled):,}")
    return human_sampled.reset_index(drop=True), ai_sampled.reset_index(drop=True)

def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Load and clean
    human_df = load_and_clean(OUTPUT_DIR / "Human_model_ready_dataset.csv", label=0)
    ai_df = load_and_clean(OUTPUT_DIR / "ai_generated_sentences_500k.csv", label=1)
    
    # 2. Perform train/val/test splits (80/10/10) on each class BEFORE balancing to avoid leakage
    log.info("Splitting datasets into train/val/test splits...")
    h_train, h_temp = train_test_split(human_df, test_size=0.20, random_state=RANDOM_SEED)
    h_val, h_test = train_test_split(h_temp, test_size=0.50, random_state=RANDOM_SEED)
    
    a_train, a_temp = train_test_split(ai_df, test_size=0.20, random_state=RANDOM_SEED)
    a_val, a_test = train_test_split(a_temp, test_size=0.50, random_state=RANDOM_SEED)
    
    # 3. Apply length-balanced subsampling to each split
    log.info(f"Applying length balancing to TRAIN (target={TRAIN_SIZE_PER_CLASS:,} per class)...")
    h_train_bal, a_train_bal = length_balanced_sample(h_train, a_train, TRAIN_SIZE_PER_CLASS)
    
    log.info(f"Applying length balancing to VAL (target={VAL_SIZE_PER_CLASS:,} per class)...")
    h_val_bal, a_val_bal = length_balanced_sample(h_val, a_val, VAL_SIZE_PER_CLASS)
    
    log.info(f"Applying length balancing to TEST (target={TEST_SIZE_PER_CLASS:,} per class)...")
    h_test_bal, a_test_bal = length_balanced_sample(h_test, a_test, TEST_SIZE_PER_CLASS)
    
    # 4. Assemble and shuffle splits
    def assemble(h, a):
        cols = ["text", "label", "source"]
        extra = ["style", "model", "domain", "source_type", "speaker"]
        all_cols = cols + [c for c in extra if c in a.columns or c in h.columns]
        for df in [h, a]:
            for c in all_cols:
                if c not in df.columns:
                    df[c] = None
        merged = pd.concat([h[all_cols], a[all_cols]], ignore_index=True)
        return merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
    train_df = assemble(h_train_bal, a_train_bal)
    val_df = assemble(h_val_bal, a_val_bal)
    test_df = assemble(h_test_bal, a_test_bal)
    
    # 5. Cross-split dedup check (safety net)
    log.info("Cross-split leakage check...")
    train_fps = set(train_df["text"].apply(fingerprint))
    val_fps = set(val_df["text"].apply(fingerprint))
    test_fps = set(test_df["text"].apply(fingerprint))
    
    log.info(f"  Train ∩ Val  : {len(train_fps & val_fps)} duplicates")
    log.info(f"  Train ∩ Test : {len(train_fps & test_fps)} duplicates")
    log.info(f"  Val ∩ Test   : {len(val_fps & test_fps)} duplicates")
    
    if len(train_fps & test_fps) > 0:
        log.warning("  Removing leaked rows from test set...")
        test_df = test_df[~test_df["text"].apply(fingerprint).isin(train_fps)].reset_index(drop=True)
        
    if len(train_fps & val_fps) > 0:
        log.warning("  Removing leaked rows from val set...")
        val_df = val_df[~val_df["text"].apply(fingerprint).isin(train_fps)].reset_index(drop=True)
        
    # 6. Save splits
    train_df.to_csv(OUTPUT_DIR / "train_tfidf.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val_tfidf.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test_tfidf.csv", index=False, encoding="utf-8")
    
    log.info("=" * 60)
    log.info("FINAL TFIDF SPLITS SUMMARY")
    log.info("=" * 60)
    for name, split in [("train_tfidf", train_df), ("val_tfidf", val_df), ("test_tfidf", test_df)]:
        counts = split["label"].value_counts()
        h_n = counts.get(0, 0)
        a_n = counts.get(1, 0)
        h_wc = split[split["label"] == 0]["text"].str.split().apply(len)
        a_wc = split[split["label"] == 1]["text"].str.split().apply(len)
        log.info(f"{name}: Total={len(split):,}, Human={h_n:,} (mean words={h_wc.mean():.1f}), AI={a_n:,} (mean words={a_wc.mean():.1f})")

if __name__ == "__main__":
    main()
