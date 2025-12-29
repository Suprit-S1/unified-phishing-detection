from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import clean_text

def build_feature_extractor():
    """
    Returns the configured TF-IDF Vectorizer.
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        max_features=5000,       # Top 5000 words
        ngram_range=(1, 2),      # Unigrams and Bigrams
        stop_words='english'
    )