# AI Customer Support Copilot

## Project Overview

AI Customer Support Copilot is an AI/ML project that automates the first stage of customer support ticket handling.

The current system predicts:

* **Ticket Type**: what kind of issue the ticket represents
* **Ticket Queue**: which team or department should handle the ticket
* **Ticket Priority**: how urgent the ticket is
* **Customer Sentiment**: whether the ticket tone is positive, neutral, or negative

This is part of a larger customer support automation system that will later include RAG-based knowledge retrieval, reply generation, API development, dashboarding, and deployment.

---

## Business Problem

Support teams receive many customer tickets across different departments. Manual ticket triage is slow, inconsistent, and prone to routing mistakes.

This project helps automate early ticket handling by predicting:

```text
subject + body → ticket type
subject + body → ticket queue
subject + body → ticket priority
ticket text     → customer sentiment
```

---

## Dataset

The main ticket dataset contains multilingual customer support tickets in **English and German**.

Important columns:

| Column     | Use                                      |
| ---------- | ---------------------------------------- |
| `subject`  | Short ticket summary                     |
| `body`     | Main ticket description                  |
| `type`     | Target for ticket type classification    |
| `queue`    | Target for ticket routing                |
| `priority` | Target for priority prediction           |
| `language` | Used for language-wise evaluation        |
| `answer`   | Reserved for future RAG/reply generation |

For sentiment analysis, an external Amazon reviews dataset was used. Amazon review ratings were converted into sentiment labels.

| Rating    | Sentiment |
| --------- | --------- |
| 1-2 stars | Negative  |
| 3 stars   | Neutral   |
| 4-5 stars | Positive  |

---

## Completed Modules

| Phase    | Module                                    | Status    |
| -------- | ----------------------------------------- | --------- |
| Phase 1  | Data understanding and preprocessing      | Completed |
| Phase 2  | Ticket type classification                | Completed |
| Phase 3  | Queue / department routing classification | Completed |
| Phase 4  | Priority prediction                       | Completed |
| Phase 5  | Sentiment analysis                        | Completed |
| Phase 6  | RAG knowledge retrieval                   | Pending   |
| Phase 7  | Reply generation                          | Pending   |
| Phase 8  | FastAPI backend                           | Pending   |
| Phase 9  | Streamlit dashboard                       | Pending   |
| Phase 10 | Deployment                                | Pending   |

---

## Models Built

### 1. Ticket Type Classifier

Predicts the type of customer support issue.

Example classes:

```text
Change
Incident
Problem
Request
```

Output model:

```text
models/ticket_type_baseline.pkl
```

---

### 2. Ticket Queue Classifier

Predicts which support queue or department should handle the ticket.

Example classes:

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

Output model:

```text
models/ticket_queue_baseline.pkl
```

---

### 3. Priority Classifier

Predicts the urgency level of a customer support ticket.

Output model:

```text
models/ticket_priority_baseline.pkl
```

Reports:

```text
reports/priority_model_comparison.csv
reports/wrong_priority_predictions.csv
```

---

### 4. Sentiment Analyzer

Predicts customer sentiment as:

```text
positive
neutral
negative
```

The sentiment model was trained using an external Amazon reviews dataset. Logistic Regression was selected as the final sentiment model because it achieved better macro F1 and better neutral-class performance than LinearSVC.

Since the main support ticket dataset contains both English and German tickets, but the sentiment training dataset is English-only, sentiment is handled as follows:

| Ticket language | Sentiment method             |
| --------------- | ---------------------------- |
| English         | Trained ML sentiment model   |
| German          | German keyword-rule fallback |

Output model:

```text
models/sentiment_model.pkl
```

Reports:

```text
reports/sentiment_model_comparison.csv
reports/wrong_sentiment_predictions.csv
reports/ticket_sentiment_predictions.csv
```

---

## Preprocessing

Text preprocessing includes:

* Combining `subject` and `body`
* Lowercasing text
* Removing URLs
* Removing extra whitespace
* Preserving German characters during ticket preprocessing
* Avoiding aggressive English-only cleaning for multilingual ticket models

For the external Amazon sentiment dataset, review text was cleaned separately for English sentiment classification.

---

## Data Leakage Prevention

The ticket classifiers only use:

```text
subject + body
```

The following columns are **not used as model input**:

```text
answer
type
queue
priority
```

`answer` is excluded from classification models because it is created after ticket resolution. Using it would leak post-resolution information.

`type`, `queue`, and `priority` are used only as target labels for their respective supervised learning tasks.

---

## Models Tested

For ticket type, queue, and priority prediction:

* TF-IDF + Logistic Regression
* TF-IDF + LinearSVC
* TF-IDF + LinearSVC + GridSearchCV
* Word + Character TF-IDF experiment for multilingual robustness

For sentiment analysis:

* TF-IDF + Logistic Regression
* TF-IDF + LinearSVC

Final selected models:

```text
Ticket Type Classifier  : TF-IDF + LinearSVC/GridSearchCV
Ticket Queue Classifier : TF-IDF + LinearSVC/GridSearchCV
Priority Classifier     : TF-IDF-based best classifier
Sentiment Analyzer      : TF-IDF + Logistic Regression
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
* Language-wise evaluation where applicable

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

Priority and sentiment evaluation reports are saved in:

```text
reports/priority_model_comparison.csv
reports/sentiment_model_comparison.csv
```

---

## Multilingual Handling

The ticket type, queue, and priority models were trained on the multilingual ticket dataset containing both English and German examples.

Sentiment analysis is handled differently:

* English tickets use the trained ML sentiment model.
* German tickets use a lightweight keyword-rule fallback because the sentiment training data is English-only.

This avoids pretending that the English sentiment model magically understands German. A rare triumph over delusion.

---

## Challenges Faced and Resolutions

| Challenge                                                | Resolution                                                                               |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Confusion between `type` and `queue`                     | Defined `type` as issue nature and `queue` as routing department. Built separate models. |
| Data leakage concern                                     | Excluded post-resolution fields like `answer`.                                           |
| German text handling                                     | Used light preprocessing and preserved German characters in ticket preprocessing.        |
| Logistic Regression underperformed for some ticket tasks | Tested LinearSVC and GridSearchCV.                                                       |
| Large GridSearchCV search time                           | Reduced the parameter grid to a practical size.                                          |
| LinearSVC has no `predict_proba()`                       | Used `decision_function()` and confidence margin instead.                                |
| Problem vs Incident confusion                            | Saved wrong predictions and documented it as a limitation.                               |
| Weak German queue performance                            | Added language-wise evaluation and tested Word + Character TF-IDF.                       |
| Sentiment dataset did not include German labels          | Used English ML sentiment model plus German keyword-rule fallback.                       |
| Neutral sentiment was difficult                          | Selected model using macro F1 and neutral-class performance instead of accuracy alone.   |

---

## Project Structure

```text
customer-support-copilot/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       ├── type_mapping.csv
│       ├── queue_mapping.csv
│       ├── sentiment_clean.csv
│       ├── sentiment_train.csv
│       └── sentiment_test.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_type_classifier.ipynb
│   ├── 04_queue_classifier.ipynb
│   ├── 05_priority_classifier.ipynb
│   └── 06_sentiment_analysis.ipynb
│
├── src/
│   └── predict_ticket.py
│
├── models/
│   ├── ticket_type_baseline.pkl
│   ├── ticket_queue_baseline.pkl
│   ├── ticket_priority_baseline.pkl
│   └── sentiment_model.pkl
│
├── reports/
│   ├── type_model_comparison.csv
│   ├── queue_model_comparison.csv
│   ├── priority_model_comparison.csv
│   ├── sentiment_model_comparison.csv
│   ├── wrong_type_predictions.csv
│   ├── wrong_queue_predictions.csv
│   ├── wrong_priority_predictions.csv
│   ├── wrong_sentiment_predictions.csv
│   └── ticket_sentiment_predictions.csv
│
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

## Sample Input

```text
I was charged twice for my subscription.
Please refund the duplicate payment.
```

## Sample Output

```text
Type      : Problem
Queue     : Billing and Payments
Priority  : High
Sentiment : Negative
```

The script also prints decision scores and confidence margins for ticket type, queue, and priority predictions.

---

## Current Limitations

* LinearSVC decision scores are not true probabilities.
* German queue classification is weaker than English queue classification.
* `Problem` and `Incident` can be confused.
* Some queue labels overlap, especially Technical Support, IT Support, and Product Support.
* Sentiment analysis is trained on an external English review dataset, not the original ticket dataset.
* German sentiment uses keyword-rule fallback, not a trained German sentiment model.
* The current system is still a classical ML baseline.

---

## Future Improvements

Planned next modules:

* RAG knowledge retriever
* Human-reviewable reply generator
* FastAPI backend
* Streamlit dashboard
* Deployment
* Multilingual transformer upgrade using `distilbert-base-multilingual-cased` or `xlm-roberta-base`
* Better German sentiment model using a German or multilingual sentiment dataset

---

## Current Status

```text
Completed:
Ticket Type Classification
Ticket Queue Routing
Priority Prediction
Sentiment Analysis
```

Next milestone:

```text
RAG Knowledge Retrieval
```
