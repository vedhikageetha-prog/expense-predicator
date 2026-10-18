import io
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.dummy import DummyClassifier

st.set_page_config(page_title="SmartSpend AI", page_icon="💸", layout="wide")
st.title("💸 SmartSpend AI")
st.caption("Personal Expense Category Predictor · Student capstone demonstration")
DATA_PATH = Path(__file__).parent / "data" / "synthetic_transactions.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    data = load_data()
    X = data["description"].fillna("").astype(str)
    y = data["category"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, predictions

data = load_data()
model, X_train, X_test, y_train, y_test, predictions = train_model()
tabs = st.tabs(["Predict expense", "Dashboard", "Model evaluation", "Dataset & privacy"])

with tabs[0]:
    st.subheader("Categorise a transaction")
    col1, col2 = st.columns([2, 1])
    with col1:
        description = st.text_input("Transaction description", placeholder="e.g., bus ticket to college")
    with col2:
        amount = st.number_input("Amount (₹)", min_value=0.0, value=250.0, step=10.0)
    if st.button("Predict category", type="primary"):
        if not description.strip():
            st.warning("Enter a transaction description first.")
        else:
            pred = model.predict([description.strip()])[0]
            probs = model.predict_proba([description.strip()])[0]
            confidence = float(max(probs))
            st.success(f"Predicted category: **{pred}**")
            st.metric("Model score (not a guarantee)", f"{confidence:.0%}")
            if len(description.strip().split()) <= 2 or confidence < 0.40:
                st.warning("This description may be ambiguous. Please review or correct the category.")
            categories = sorted(data["category"].unique())
            corrected = st.selectbox("Confirm or override category", categories,
                                     index=categories.index(pred), key="corrected_category")
            if st.button("Use corrected category"):
                st.info(f"Recorded for this demo only: ₹{amount:,.2f} · {corrected}. This does not retrain the model.")
    st.caption("The model uses transaction text. Amount is displayed but is not used by this baseline classifier.")

with tabs[1]:
    st.subheader("Expense dashboard (synthetic sample data)")
    total = float(data["amount"].sum())
    a, b, c = st.columns(3)
    a.metric("Transactions", len(data))
    b.metric("Total sample spend", f"₹{total:,.2f}")
    c.metric("Categories", data["category"].nunique())
    cat_summary = data.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(cat_summary["category"], cat_summary["amount"])
    ax.set_ylabel("Amount (₹)")
    ax.set_title("Total synthetic spend by category")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    st.pyplot(fig)
    monthly = data.copy()
    monthly["month"] = pd.to_datetime(monthly["date"], errors="coerce").dt.to_period("M").astype(str)
    monthly_summary = monthly.groupby("month", as_index=False)["amount"].sum()
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.plot(monthly_summary["month"], monthly_summary["amount"], marker="o")
    ax2.set_title("Synthetic spend by month")
    ax2.set_ylabel("Amount (₹)")
    ax2.tick_params(axis="x", rotation=35)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.dataframe(cat_summary.rename(columns={"category":"Category","amount":"Total (₹)"}), use_container_width=True)

with tabs[2]:
    st.subheader("Held-out test evaluation")
    st.write("The model is evaluated on a test split not used for training.")
    st.metric("Test accuracy", f"{accuracy_score(y_test, predictions):.1%}")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
    labels = sorted(data["category"].unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax3, xticks_rotation=45, colorbar=False
    )
    ax3.set_title("Confusion matrix")
    fig3.tight_layout()
    st.pyplot(fig3)
    st.info("Synthetic data is small and simplified. Scores are for demonstration and should not be treated as real-world banking performance.")

with tabs[3]:
    st.subheader("Dataset preview")
    st.write(f"Rows: {len(data)} · Missing descriptions: {int(data['description'].isna().sum())}")
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button("Download synthetic dataset CSV", data=data.to_csv(index=False).encode("utf-8"),
                       file_name="synthetic_transactions.csv", mime="text/csv")
    st.markdown("""
    **Privacy safeguards**
    - This starter project uses only synthetic sample transactions.
    - Do not upload real bank credentials, PINs, passwords, account numbers, or unredacted statements.
    - The demo has no bank connection and does not send data to a bank.
    - Predictions are categorisation suggestions, not financial advice.

    **Limitations**
    - Merchant names and short descriptions can be ambiguous.
    - The current baseline classifies text only; transaction amount is not a model feature.
    - The correction control demonstrates human review but does not persist or retrain from corrections.
    - Real deployment would require consent, secure storage, data retention rules, stronger testing, and bias/error monitoring.
    """)
