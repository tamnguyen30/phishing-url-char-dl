"""Shared TF-IDF/logistic-regression constructors for the existing baseline."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def estimators(C=10.0):
    return (TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=5,
                           sublinear_tf=True, max_features=500_000, lowercase=False),
            LogisticRegression(max_iter=1000, C=C, n_jobs=-1))
