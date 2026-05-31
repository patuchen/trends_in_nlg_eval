import os
import sys
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

# Setup paths
analysis_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, analysis_dir)
from helpers import load_results

RESULTS_CSV = os.path.join(analysis_dir, "../output/results.csv")
RESULTS_JSONL = os.path.join(analysis_dir, "../output/results.jsonl")
PAPERS_CSV = os.path.join(analysis_dir, "../papers.csv")
CACHE_DIR = os.path.join(analysis_dir, "../output")
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "eval_sentence_embeddings.npy")
SENTENCES_OUTPUT_CSV = os.path.join(CACHE_DIR, "bertopic_sentence_results.csv")

EVAL_KEYWORDS = [
    "eval", "metric", "human", "annotator", "agreement", "correlation",
    "likert", "bleu", "rouge", "meteor", "chrf", "bertscore", "bleurt", "comet",
    "judge", "annotated", "assessment", "user study", "crowdsourc", "kappa",
    "pearson", "spearman", "direct assessment", "post-edit", "pairwise", "ranking",
    "questionnaire", "coherence", "fluency", "adequacy", "faithfulness", "accuracy",
    "preference", "quality estimation"
]

def split_sentences(text):
    if not isinstance(text, str):
        return []
    # Split on period, exclamation, or question mark followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])\s*', text)
    return [s.strip() for s in sentences if s.strip()]

def is_eval_sentence(sentence):
    s_lower = sentence.lower()
    return any(kw in s_lower for kw in EVAL_KEYWORDS)

def main():
    # Reconfigure stdout to use UTF-8 to prevent Windows terminal encoding issues
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    print("Loading data...")
    # Load primary annotated dataset
    df_results, _ = load_results(RESULTS_JSONL, RESULTS_CSV)
    print(f"Loaded {len(df_results):,} papers from annotated results.")
    
    # Load papers metadata (which contains abstracts)
    if not os.path.exists(PAPERS_CSV):
        print(f"Error: {PAPERS_CSV} not found! Cannot extract abstracts.")
        return
        
    df_papers = pd.read_csv(PAPERS_CSV, usecols=["id", "abstract"], low_memory=False)
    print(f"Loaded {len(df_papers):,} papers from {PAPERS_CSV}.")
    
    # Merge on paper_id/id
    df = df_results.merge(df_papers, left_on="paper_id", right_on="id", how="inner")
    df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")].copy()
    print(f"Merged dataset with valid abstracts: {len(df):,} papers.")
    
    # Extract evaluation-related sentences from abstracts
    print("Extracting evaluation-related sentences...")
    eval_sentences_data = []
    for idx, row in df.iterrows():
        sentences = split_sentences(row["abstract"])
        for s in sentences:
            if is_eval_sentence(s):
                eval_sentences_data.append({
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "year": row["year"],
                    "venue": row["venue"],
                    "sentence": s
                })
                
    df_sentences = pd.DataFrame(eval_sentences_data)
    print(f"Total evaluation sentences extracted: {len(df_sentences):,}")
    
    if len(df_sentences) == 0:
        print("No evaluation sentences found. Exiting.")
        return

    sentences_list = df_sentences["sentence"].tolist()
    
    # 1. Embedding generation/loading
    embeddings = None
    if os.path.exists(EMBEDDINGS_CACHE):
        print("Loading cached sentence embeddings from", EMBEDDINGS_CACHE)
        try:
            embeddings = np.load(EMBEDDINGS_CACHE)
            if len(embeddings) != len(sentences_list):
                print(f"Warning: Cached embeddings length ({len(embeddings)}) does not match sentence count ({len(sentences_list)}). Recomputing...")
                embeddings = None
        except Exception as e:
            print("Failed to load cache:", e)
            embeddings = None

    if embeddings is None:
        print("Encoding sentences using all-MiniLM-L6-v2 model on CPU...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences_list, show_progress_bar=True)
        # Save to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(EMBEDDINGS_CACHE, embeddings)
        print("Saved embeddings to", EMBEDDINGS_CACHE)

    # 2. Setup BERTopic sub-models for reproducibility and quality
    print("Initializing BERTopic sub-models...")
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=80,  # Adjusted for 23k sentences
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    
    # Vectorizer to remove stopwords and extract unigrams and bigrams
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2)
    )

    # 3. Fit BERTopic
    print("Fitting BERTopic model on sentences...")
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=20,  # Keep to 20 topics for high interpretability
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(sentences_list, embeddings)
    
    # Save results back to DataFrame
    df_sentences["topic"] = topics
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Set up logger to safely handle console encoding issues on Windows
    log_lines = []
    def log(msg):
        log_lines.append(msg)
        try:
            print(msg)
        except Exception:
            try:
                print(msg.encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass

    log("\n==================================================")
    log("BERTOPIC DISCOVERED SENTENCE TOPICS (EVALUATION-SPECIFIC)")
    log("==================================================")
    log(topic_info.to_string(index=False))
    log("\n==================================================")
    
    # Show representative terms for each topic
    log("\nTOPIC DETAILS AND REPRESENTATIVE TERMS:")
    log("==================================================")
    for row in topic_info.itertuples():
        t_id = row.Topic
        if t_id == -1:
            log(f"\nTopic -1 (Outliers/Unclassified): {row.Count} sentences")
            continue
        words = [w[0] for w in topic_model.get_topic(t_id)[:10]]
        log(f"\nTopic {t_id}: {row.Name} ({row.Count} sentences)")
        log(f"  Top terms: {', '.join(words)}")
        
    # Save topic definitions and representative words
    topic_summary_records = []
    for row in topic_info.itertuples():
        t_id = row.Topic
        words = ", ".join([w[0] for w in topic_model.get_topic(t_id)[:10]]) if t_id != -1 else ""
        topic_summary_records.append({
            "topic_id": t_id,
            "name": row.Name,
            "count": row.Count,
            "top_terms": words
        })
    df_summary = pd.DataFrame(topic_summary_records)
    df_summary.to_csv(os.path.join(CACHE_DIR, "bertopic_sentence_topic_summary.csv"), index=False)
    
    # Analyze topic distribution over time (by period)
    def get_period(year):
        if year < 2000:
            return "<2000"
        elif year < 2010:
            return "2000-2009"
        elif year < 2020:
            return "2010-2019"
        else:
            return "2020-2025"
            
    df_sentences["period"] = df_sentences["year"].apply(get_period)
    
    log("\nTOPIC DISTRIBUTION BY PERIOD:")
    log("==================================================")
    # Pivot table of topic percentages by period (excluding outliers -1)
    df_valid = df_sentences[df_sentences["topic"] != -1]
    topic_period = pd.crosstab(df_valid["topic"], df_valid["period"], normalize="columns") * 100
    # Add topic names for context
    topic_names = topic_info.set_index("Topic")["Name"].to_dict()
    topic_period["name"] = topic_period.index.map(topic_names)
    cols = ["name"] + [c for c in topic_period.columns if c != "name"]
    topic_period = topic_period[cols]
    log(topic_period.round(2).to_string())
    
    # Save pivot table
    topic_period.to_csv(os.path.join(CACHE_DIR, "bertopic_sentence_period_trends.csv"))
    
    # Save final sentence-topic mapping
    df_sentences.to_csv(SENTENCES_OUTPUT_CSV, index=False)
    log(f"\nSaved detailed results to {SENTENCES_OUTPUT_CSV}")
    
    # Save log file in UTF-8
    with open(os.path.join(CACHE_DIR, "bertopic_sentence_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Saved run logs to {os.path.join(CACHE_DIR, 'bertopic_sentence_log.txt')}")
    log("Sentence topic modeling completed successfully!")

if __name__ == "__main__":
    main()
