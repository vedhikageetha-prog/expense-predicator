# SmartSpend AI — Personal Expense Category Predictor

A student capstone demo that predicts expense categories from transaction descriptions using TF-IDF + Logistic Regression. Includes synthetic data, evaluation metrics, a dashboard, and privacy/limitations notes.

## Features
- Rule-free ML baseline using text features (TF-IDF) and Logistic Regression
- Synthetic labeled dataset and downloadable CSV
- User-entered transaction prediction with a manual category override demonstration
- Category/monthly spending charts
- Held-out test metrics, per-category precision/recall/F1 and confusion matrix
- Privacy and limitations notes

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract the ZIP to a folder.
3. Open Command Prompt or the VS Code terminal in this folder.
4. Create a virtual environment (recommended):
   `python -m venv .venv`
5. Activate it:
   `.venv\Scripts\activate`
6. Install dependencies:
   `python -m pip install -r requirements.txt`
7. Start the app:
   `python -m streamlit run app.py`
8. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Run in Google Colab
This is a Streamlit app, so it is easiest to run locally or deploy to a Streamlit-compatible host. The ML pipeline itself can be inspected in `app.py`.

## Project files
- `app.py` — interactive application, model training and evaluation
- `data/synthetic_transactions.csv` — synthetic labeled transactions
- `DATA_DICTIONARY.md` — field definitions
- `REPORT.md` — report draft and methodology
- `requirements.txt` — Python dependencies

## Important
This is an educational prototype, not a financial product. The dataset is synthetic and small, so model scores are not evidence of real-world performance. Do not enter bank credentials or sensitive financial information. Amount is shown in the interface but is not used by the text-only model.
