# AI Customer Support Copilot

## Project Overview

AI Customer Support Copilot is an end-to-end AI/ML and NLP project that automates customer support ticket analysis and response assistance.

The current system performs:

* **Ticket Type Prediction**: identifies what kind of issue the ticket represents
* **Ticket Queue Prediction**: routes the ticket to the right team or department
* **Ticket Priority Prediction**: estimates how urgent the ticket is
* **Customer Sentiment Analysis**: detects whether the customer tone is positive, neutral, or negative
* **RAG Knowledge Retrieval**: retrieves relevant company policy context for a ticket
* **Support Reply Generation**: generates a human-reviewable customer support response using predictions and retrieved policy context

The project is being developed as a complete customer support automation system with machine learning classification, retrieval-augmented generation, response assistance, API development, dashboarding, and deployment.

---

## Business Problem

Support teams receive many customer tickets across different departments. Manual ticket triage is slow, inconsistent, and prone to routing mistakes.

This project helps automate early ticket handling by predicting ticket metadata, retrieving relevant support knowledge, and generating a draft response for human review.

```text
subject + body → ticket type
subject + body → ticket queue
subject + body → ticket priority
ticket text     → customer sentiment
ticket text     → relevant company policy
ticket + predictions + policy context → support reply
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
| `answer`   | Excluded from model input to prevent leakage |

For sentiment analysis, an external Amazon reviews dataset was used. Amazon review ratings were converted into sentiment labels.

| Rating    | Sentiment |
| --------- | --------- |
| 1-2 stars | Negative  |
| 3 stars   | Neutral   |
| 4-5 stars | Positive  |

For RAG-based retrieval, custom company policy documents were created and stored locally inside the project.

---

## Completed Modules

| Phase    | Module                                    | Status    |
| -------- | ----------------------------------------- | --------- |
| Phase 1  | Data understanding and preprocessing      | Completed |
| Phase 2  | Ticket type classification                | Completed |
| Phase 3  | Queue / department routing classification | Completed |
| Phase 4  | Priority prediction                       | Completed |
| Phase 5  | Sentiment analysis                        | Completed |
| Phase 6  | RAG knowledge retrieval                   | Completed |
| Phase 7  | Support reply generation                  | Completed |
| Phase 8  | FastAPI backend                           | Pending   |
| Phase 9  | Streamlit dashboard                       | Pending   |
| Phase 10 | Deployment                                | Pending   |

---

## Modules Built

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

### 5. RAG Knowledge Retrieval

A Retrieval-Augmented Generation (RAG) module was implemented to retrieve relevant company policy context before generating customer responses.

Traditional classification models can predict ticket type, queue, priority, and sentiment, but they do not know company-specific policies. The RAG module solves this by searching an internal knowledge base and returning the most relevant policy information for the ticket.

Technologies used:

* LangChain
* Sentence Transformers
* Hugging Face Embeddings
* FAISS Vector Database

Policy knowledge base:

```text
docs/company_policies/
```

Current policy documents:

```text
refund_policy.txt
shipping_policy.txt
warranty_policy.txt
account_policy.txt
technical_support_policy.txt
```

Retrieval pipeline:

```text
Customer Ticket
        ↓
Text Embedding
        ↓
FAISS Similarity Search
        ↓
Relevant Policy Retrieval
```

Example input:

```text
My product arrived damaged and I want a refund.
```

Example retrieved policy:

```text
Refund Policy
Customers can request a refund within 30 days of purchase.
For damaged products, customers may choose either a refund or replacement.
```

Output files:

```text
notebooks/07_rag_retrieval.ipynb
src/rag_pipeline.py
vectorstore/faiss_policy_index/
reports/rag_test_results.csv
```

---

### 6. Support Reply Generation

A support reply generation module was developed to create professional, human-reviewable responses.

The reply generator combines machine learning predictions with retrieved company policy context. This allows the system to generate responses that are not only based on the ticket text, but also grounded in relevant support policies.

Inputs used:

* Customer ticket text
* Predicted ticket type
* Predicted support queue
* Predicted ticket priority
* Predicted customer sentiment
* Retrieved policy context

End-to-end reply pipeline:

```text
Customer Ticket
        ↓
Ticket Type Prediction
        ↓
Queue Prediction
        ↓
Priority Prediction
        ↓
Sentiment Analysis
        ↓
RAG Policy Retrieval
        ↓
Support Reply Generation
```

Example input:

```text
My laptop arrived damaged and I want a refund.
```

Example generated response:

```text
Dear Customer,

We're sorry to hear about the issue you are facing.

Based on your message, this request appears to be related to the predicted ticket type and should be handled by the predicted support queue.

Relevant policy guidance has been retrieved to help the support team respond accurately.

Please keep your order details, product information, screenshots, or proof of purchase available if required by the support team.

Best regards,
Customer Support Team
```

Output files:

```text
notebooks/08_reply_generation.ipynb
src/generate_reply.py
reports/generated_reply_examples.csv
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

For RAG retrieval:

* Sentence Transformer embeddings
* FAISS similarity search
* Local policy knowledge base retrieval

Final selected models and components:

```text
Ticket Type Classifier  : TF-IDF + LinearSVC/GridSearchCV
Ticket Queue Classifier : TF-IDF + LinearSVC/GridSearchCV
Priority Classifier     : TF-IDF-based best classifier
Sentiment Analyzer      : TF-IDF + Logistic Regression
RAG Retriever           : Sentence Transformers + FAISS
Reply Generator         : Template-based response generation using ML predictions and RAG context
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
* RAG retrieval test cases
* Generated reply examples

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

Evaluation reports are saved in:

```text
reports/priority_model_comparison.csv
reports/sentiment_model_comparison.csv
reports/rag_test_results.csv
reports/generated_reply_examples.csv
```

---

## Multilingual Handling

The ticket type, queue, and priority models were trained on the multilingual ticket dataset containing both English and German examples.

Sentiment analysis is handled differently:

* English tickets use the trained ML sentiment model.
* German tickets use a lightweight keyword-rule fallback because the sentiment training data is English-only.

This avoids pretending that the English sentiment model magically understands German. A rare triumph over delusion.

The current RAG policy knowledge base is written in English. Multilingual policy retrieval can be improved later by adding German policy documents or using multilingual embeddings.

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
| Models lacked company-specific knowledge                 | Added RAG retrieval using FAISS and policy documents.                                    |
| Reply generation needed to be reviewable                 | Built a template-based reply generator grounded in predictions and retrieved context.    |

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
├── docs/
│   └── company_policies/
│       ├── refund_policy.txt
│       ├── shipping_policy.txt
│       ├── warranty_policy.txt
│       ├── account_policy.txt
│       └── technical_support_policy.txt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_type_classifier.ipynb
│   ├── 04_queue_classifier.ipynb
│   ├── 05_priority_classifier.ipynb
│   ├── 06_sentiment_analysis.ipynb
│   ├── 07_rag_retrieval.ipynb
│   └── 08_reply_generation.ipynb
│
├── src/
│   ├── predict_ticket.py
│   ├── rag_pipeline.py
│   └── generate_reply.py
│
├── vectorstore/
│   └── faiss_policy_index/
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
│   ├── ticket_sentiment_predictions.csv
│   ├── rag_test_results.csv
│   └── generated_reply_examples.csv
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

Run ticket prediction:

```bash
python src/predict_ticket.py
```

Run RAG policy retrieval:

```bash
python src/rag_pipeline.py
```

Run full reply generation:

```bash
python src/generate_reply.py
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

The prediction script also prints decision scores and confidence margins for ticket type, queue, and priority predictions.

## Sample RAG + Reply Output

```text
Policy Source: refund_policy.txt

Generated Reply:
Dear Customer,

We're sorry to hear about the issue you are facing.

Based on your message, this request appears to be related to the predicted ticket type and should be handled by the predicted support queue.

Relevant policy guidance has been retrieved to help the support team respond accurately.

Please keep your order details, product information, screenshots, or proof of purchase available if required by the support team.

Best regards,
Customer Support Team
```

---

## Current Limitations

* LinearSVC decision scores are not true probabilities.
* German queue classification is weaker than English queue classification.
* `Problem` and `Incident` can be confused.
* Some queue labels overlap, especially Technical Support, IT Support, and Product Support.
* Sentiment analysis is trained on an external English review dataset, not the original ticket dataset.
* German sentiment uses keyword-rule fallback, not a trained German sentiment model.
* The current RAG policy base is small and manually created.
* Reply generation is currently template-based, not LLM-powered.
* The system is not yet exposed through an API or dashboard.

---

## Future Improvements

Planned next modules and upgrades:

* FastAPI backend
* Streamlit dashboard
* Cloud deployment
* OpenAI/LLM-powered response generation
* Multilingual RAG knowledge base
* Better German sentiment model using a German or multilingual sentiment dataset
* Multilingual transformer upgrade using `distilbert-base-multilingual-cased` or `xlm-roberta-base`
* Model monitoring and feedback loop for support agent corrections

---

## Current Status

```text
Completed:
Data Understanding & Preprocessing
Ticket Type Classification
Ticket Queue Routing
Priority Prediction
Sentiment Analysis
RAG Knowledge Retrieval
Support Reply Generation
```

Next milestone:

```text
FastAPI Backend
```
