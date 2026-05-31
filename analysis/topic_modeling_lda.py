import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Setup paths
analysis_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, analysis_dir)
from helpers import load_results

RESULTS_CSV = os.path.join(analysis_dir, "../output/results.csv")
RESULTS_JSONL = os.path.join(analysis_dir, "../output/results.jsonl")

def main():
    print("Loading data...")
    # Load corpus using helper
    df, _ = load_results(RESULTS_JSONL, RESULTS_CSV)
    
    # Drop rows without abstracts
    df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")].copy()
    print(f"Total papers with abstracts: {len(df):,}")
    
    if len(df) == 0:
        print("No papers with abstracts found. Exiting.")
        return
        
    # Parameters for topic modeling
    n_topics = 10
    n_top_words = 15
    max_features = 5000
    
    print(f"\nRunning LDA Topic Modeling with {n_topics} topics...")
    
    # Vectorize abstracts using CountVectorizer
    # We use standard English stopwords and exclude words appearing in >95% or <2 papers.
    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=max_features,
        stop_words='english'
    )
    
    tf = tf_vectorizer.fit_transform(df["abstract"])
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    
    # Fit Latent Dirichlet Allocation model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )
    lda.fit(tf)
    
    # Print the top words for each topic
    print("\n==================================================")
    print("DISCOVERED TOPICS IN ABSTRACTS")
    print("==================================================")
    for topic_idx, topic in enumerate(lda.components_):
        message = f"Topic #{topic_idx + 1}: "
        message += " ".join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        print("-" * 50)

if __name__ == "__main__":
    main()
