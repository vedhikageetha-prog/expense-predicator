"""Standalone evaluation helper for SmartSpend AI."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv(Path(__file__).parent / "data" / "synthetic_transactions.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data["description"].fillna("").astype(str),
    data["category"].astype(str),
    test_size=0.25,
    random_state=42,
    stratify=data["category"],
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix labels:", sorted(data["category"].unique()))
print(confusion_matrix(y_test, pred, labels=sorted(data["category"].unique())))
