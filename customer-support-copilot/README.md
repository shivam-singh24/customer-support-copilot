# AI Customer Support Copilot

## Project Overview

AI Customer Support Copilot is an AI/ML project that automates the first stage of customer support ticket handling.

The current system predicts:

* **Ticket Type**: what kind of issue the ticket represents
* **Ticket Queue**: which team or department should handle the ticket

This is the first milestone of a larger system that will later include priority detection, sentiment analysis, RAG-based policy retrieval, reply generation, API development, dashboarding, and deployment.

---

## Business Problem

Support teams receive many customer tickets across different departments. Manual ticket triage is slow, inconsistent, and prone to routing mistakes.

This project helps automate triage by predicting:

```text
subject + body → ticket type
subject + body → ticket queue
```

---

## Dataset

The dataset contains multilingual customer support tickets in **English and German**.

Important columns:

| Column     | Use                                        |
| ---------- | ------------------------------------------ |
| `subject`  | Short ticket summary                       |
| `body`     | Main ticket description                    |
| `type`     | Target for ticket type classification      |
| `queue`    | Target for ticket routing                  |
| `priority` | Future urgency model target                |
| `language` | Used for language-wise evaluation          |
| `answer`   | Not used because it may cause data leakage |

---

## Current Milestone: Days 1-5

Completed:

* Dataset exploration
* Text preprocessing
* Ticket Type Classifier
* Ticket Queue Classifier
* Model comparison
* Error analysis
* Language-wise evaluation
* Saved models
* Prediction script

---

## Models Built

### 1. Ticket Type Classifier

Predicts:

```text
Change / Incident / Problem / Request
```

### 2. Ticket Queue Classifier

Predicts departments such as:

```text
Technical Support
Billing and Payments
Product Support
Customer Service
IT Support
Sales and Pre-Sales
Human Resources
Returns and Exchanges
Service Outages and Maintenance
General Inquiry
```

---

## Preprocessing

Text preprocessing includes:

* Combining `subject` and `body`
* Lowercasing text
* Removing URLs
* Removing extra whitespace
* Preserving German characters such as `ä`, `ö`, `ü`, and `ß`

Aggressive English-only regex cleaning was avoided because the dataset contains German tickets.

---

## Data Leakage Prevention

The model only uses:

```text
subject + body
```

The following columns are **not used as input**:

```text
answer
type
queue
priority
```

`answer` is excluded because it is created after ticket resolution. Using it would leak post-resolution information.

`type` and `queue` are used only as target labels.

---

## Models Tested

* TF-IDF + Logistic Regression
* TF-IDF + LinearSVC
* TF-IDF + LinearSVC + GridSearchCV
* Word + Character TF-IDF experiment for multilingual robustness

Final selected models:

```text
Ticket Type Classifier  : TF-IDF + LinearSVC + GridSearchCV
Ticket Queue Classifier : TF-IDF + LinearSVC + GridSearchCV
```

---

## Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* Confusion matrix
* Wrong prediction analysis
* Language-wise evaluation

### Queue Classifier Results

| Model                    | Accuracy | Macro F1 | Weighted F1 |
| ------------------------ | -------: | -------: | ----------: |
| Logistic Regression      |     0.47 |     0.47 |        0.47 |
| LinearSVC                |     0.59 |     0.59 |        0.59 |
| LinearSVC + GridSearchCV |     0.61 |     0.60 |        0.61 |

### Queue Language-Wise Evaluation

| Language | Accuracy | Macro F1 | Weighted F1 |
| -------- | -------: | -------: | ----------: |
| German   |     0.49 |     0.46 |        0.49 |
| English  |     0.70 |     0.71 |        0.70 |

Type classifier results are saved in:

```text
reports/type_model_comparison.csv
```

Queue classifier results are saved in:

```text
reports/queue_model_comparison.csv
```

---

## Multilingual Experiment

German queue performance was weaker than English queue performance.

To improve this, a **Word + Character TF-IDF** model was tested.

| Model                                | German Weighted F1 | English Weighted F1 | Decision     |
| ------------------------------------ | -----------------: | ------------------: | ------------ |
| Word TF-IDF + LinearSVC/GridSearchCV |              0.489 |               0.700 | Kept         |
| Word + Character TF-IDF + LinearSVC  |              0.502 |               0.678 | Not selected |

The Word + Character model slightly improved German performance but reduced English and overall performance. Therefore, the original GridSearchCV model was kept.

---

## Challenges Faced and Resolutions

| Challenge                            | Resolution                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Confusion between `type` and `queue` | Defined `type` as issue nature and `queue` as routing department. Built separate models for both. |
| Data leakage concern                 | Excluded post-resolution fields like `answer`. Used only `subject + body` as input.               |
| German text handling                 | Used light preprocessing and preserved German characters.                                         |
| Logistic Regression underperformed   | Tested LinearSVC, which improved results.                                                         |
| Large GridSearchCV search time       | Reduced the parameter grid to a practical size.                                                   |
| LinearSVC has no `predict_proba()`   | Used `decision_function()` and confidence margin instead.                                         |
| Problem vs Incident confusion        | Saved wrong predictions and documented it as a limitation.                                        |
| Weak German queue performance        | Added language-wise evaluation and tested Word + Character TF-IDF.                                |

---

## Project Structure

```text
customer-support-copilot/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run prediction:

```bash
python src/predict_ticket.py
```

---

## Sample Output

Example input:

```text
I was charged twice for my subscription.
Please refund the duplicate payment.
```

Example output:

```text
Predicted Type  : Problem
Predicted Queue : Billing and Payments
```

The script also prints LinearSVC decision scores and confidence margins.

---

## Current Limitations

* LinearSVC gives decision scores, not true probabilities.
* German queue classification is weaker than English queue classification.
* `Problem` and `Incident` can be confused.
* Some queue labels overlap, especially Technical Support, IT Support, and Product Support.
* The current system is a classical ML baseline.

---

## Future Improvements

Planned next modules:

* Priority classifier
* Sentiment analyzer
* RAG policy retriever
* Human-reviewable reply generator
* FastAPI backend
* Streamlit dashboard
* Deployment
* Multilingual transformer upgrade using `distilbert-base-multilingual-cased` or `xlm-roberta-base`

---

## Current Status

```text
Milestone 1 completed:
Ticket Type Classification + Ticket Queue Routing
```

Next milestone:

```text
Priority Classifier
```
