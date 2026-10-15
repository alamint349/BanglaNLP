"""
Feature extraction for classical ML models.

  bow             - word unigram+bigram counts
  tfidf_word      - word unigram+bigram TF-IDF (bigrams capture "ভালো না")
  tfidf_word_char - word TF-IDF + character n-gram TF-IDF. Char n-grams are
                    robust to Bangla's rich inflection, spelling variation and
                    noisy social-media text.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .preprocessing import whitespace_tokenizer


def _word_kwargs(min_df: int) -> dict:
    # token_pattern=None + our tokenizer: sklearn's default regex breaks Bangla words
    return dict(tokenizer=whitespace_tokenizer, token_pattern=None, lowercase=False,
                ngram_range=(1, 2), min_df=min_df, max_features=50_000)


def bow(min_df: int = 1):
    return CountVectorizer(**_word_kwargs(min_df))


def tfidf_word(min_df: int = 1):
    return TfidfVectorizer(sublinear_tf=True, **_word_kwargs(min_df))


def tfidf_word_char(min_df: int = 1):
    return FeatureUnion([
        ("word", tfidf_word(min_df)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False,
                                 min_df=min_df, max_features=150_000, sublinear_tf=True)),
    ])


FEATURE_BUILDERS = {
    "bow": bow,
    "tfidf_word": tfidf_word,
    "tfidf_word_char": tfidf_word_char,
}
