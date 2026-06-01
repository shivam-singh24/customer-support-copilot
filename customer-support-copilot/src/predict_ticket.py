import joblib
import numpy as np
from pathlib import Path


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TYPE_MODEL_PATH = PROJECT_ROOT / "models" / "ticket_type_baseline.pkl"
QUEUE_MODEL_PATH = PROJECT_ROOT / "models" / "ticket_queue_baseline.pkl"


# -----------------------------
# Load trained models
# -----------------------------
type_model = joblib.load(TYPE_MODEL_PATH)
queue_model = joblib.load(QUEUE_MODEL_PATH)


# -----------------------------
# Sample ticket
# -----------------------------
sample_ticket = """
I was charged twice for my subscription.
Please refund the duplicate payment.
"""


# -----------------------------
# Helper function for LinearSVC
# -----------------------------
def analyze_with_decision_scores(model, text):
    prediction = model.predict([text])[0]
    scores = model.decision_function([text])[0]
    classes = model.classes_

    ranked_indices = np.argsort(scores)[::-1]

    top_index = ranked_indices[0]
    second_index = ranked_indices[1]

    top_class = classes[top_index]
    top_score = scores[top_index]

    second_class = classes[second_index]
    second_score = scores[second_index]

    margin = top_score - second_score

    return {
        "prediction": prediction,
        "top_class": top_class,
        "top_score": top_score,
        "second_class": second_class,
        "second_score": second_score,
        "margin": margin,
        "ranked_results": [
            {
                "class": classes[i],
                "score": scores[i]
            }
            for i in ranked_indices
        ]
    }


def interpret_margin(margin):
    if margin >= 1.0:
        return "High confidence"
    elif margin >= 0.4:
        return "Medium confidence"
    else:
        return "Low confidence"


# -----------------------------
# Run predictions
# -----------------------------
type_result = analyze_with_decision_scores(type_model, sample_ticket)
queue_result = analyze_with_decision_scores(queue_model, sample_ticket)


# -----------------------------
# Display results
# -----------------------------
print("=" * 60)
print("TICKET ANALYSIS")
print("=" * 60)

print("\nInput Ticket:")
print(sample_ticket.strip())

print("\n" + "-" * 60)
print("TYPE PREDICTION")
print("-" * 60)
print("Predicted Type:", type_result["prediction"])
print("Top Score:", round(type_result["top_score"], 4))
print("Second Best:", type_result["second_class"])
print("Second Score:", round(type_result["second_score"], 4))
print("Confidence Margin:", round(type_result["margin"], 4))
print("Confidence Level:", interpret_margin(type_result["margin"]))

print("\nAll Type Decision Scores:")
for item in type_result["ranked_results"]:
    print(f"{item['class']:<15} {round(item['score'], 4)}")


print("\n" + "-" * 60)
print("QUEUE PREDICTION")
print("-" * 60)
print("Predicted Queue:", queue_result["prediction"])
print("Top Score:", round(queue_result["top_score"], 4))
print("Second Best:", queue_result["second_class"])
print("Second Score:", round(queue_result["second_score"], 4))
print("Confidence Margin:", round(queue_result["margin"], 4))
print("Confidence Level:", interpret_margin(queue_result["margin"]))

print("\nAll Queue Decision Scores:")
for item in queue_result["ranked_results"]:
    print(f"{item['class']:<35} {round(item['score'], 4)}")

print("\nNote:")
print("Decision scores are not probabilities.")
print("Higher score means stronger model preference.")
print("Margin = top score - second-best score.")