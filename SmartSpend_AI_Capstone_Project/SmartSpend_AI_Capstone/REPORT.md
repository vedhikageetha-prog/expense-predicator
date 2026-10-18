# SmartSpend AI: Personal Expense Category Predictor
## Capstone Project Report (Draft)

### 1. Abstract
SmartSpend AI is an educational machine-learning prototype that classifies personal expense descriptions into a fixed set of spending categories. It combines text preprocessing, TF-IDF features, Logistic Regression, evaluation on a held-out test split, and an interactive dashboard. The demonstration uses synthetic data only.

### 2. Problem statement
Manually categorising expenses can be repetitive. The project investigates whether short transaction descriptions can be mapped to broad spending categories while keeping privacy risks visible and allowing users to correct predictions.

### 3. Objectives
- Create and document a synthetic, labeled transaction dataset.
- Inspect data quality and category distribution.
- Establish a simple, reproducible text-classification pipeline.
- Evaluate per-category precision, recall, F1-score, and confusion matrix.
- Build a demo that accepts a description and allows human override.
- Present category/month summaries and document privacy safeguards and limitations.

### 4. Methodology
1. Generate synthetic transactions with a category label.
2. Split descriptions and labels into training and testing sets using a fixed random seed and stratification.
3. Transform descriptions with TF-IDF using unigrams and bigrams.
4. Train Logistic Regression with balanced class weights.
5. Evaluate predictions on the held-out test set.
6. Display metrics and visual summaries in Streamlit.

### 5. Tools
Python, Pandas, Scikit-learn, Streamlit, Matplotlib, CSV.

### 6. Evaluation
The application computes test accuracy, per-class precision, recall, F1-score, support, and a confusion matrix at runtime. Insert the actual metrics from the running app into the final presentation/report. Do not invent results or present synthetic-data scores as real-world performance.

### 7. Privacy and responsible use
- Only synthetic sample records are included.
- Never request bank passwords, PINs, or login credentials.
- Avoid storing account numbers or personally identifying transaction details.
- Keep user control through category override.
- Present predictions as suggestions, not financial advice.
- A production system would need informed consent, encryption, access control, retention/deletion processes, and security testing.

### 8. Limitations
- Small handcrafted dataset.
- Real merchants and descriptions may be ambiguous.
- Categories may overlap (for example, supermarket food versus groceries).
- Amount is not used by the current classifier.
- Correction is a demo-only interaction and does not persist or retrain the model.
- Synthetic test results do not establish performance on real users.

### 9. Future improvements
- Expand a consented, anonymised labeled dataset.
- Compare Linear SVM and Naive Bayes with the same test split.
- Add calibrated confidence and a review threshold.
- Persist user-approved corrections securely.
- Add recurring transaction detection and anomaly detection as separate tasks.
- Add import validation for redacted CSV files.

### 10. Conclusion
The prototype demonstrates a reproducible baseline for text-based expense categorisation and an interface for reviewing predictions. Its main deliverables are the synthetic dataset, data dictionary, model pipeline, evaluation view, dashboard, and privacy/limitations documentation.
