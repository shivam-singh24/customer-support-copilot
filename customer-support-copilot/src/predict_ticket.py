import re
from pathlib import Path

import joblib
import numpy as np


# ============================================================
# Paths and model loading
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = {
    "type": MODELS_DIR / "ticket_type_baseline.pkl",
    "queue": MODELS_DIR / "ticket_queue_baseline.pkl",
    "priority": MODELS_DIR / "ticket_priority_baseline.pkl",
    "sentiment": MODELS_DIR / "sentiment_model.pkl",
}


def load_models():
    """Load all trained models once."""
    return {
        model_name: joblib.load(model_path)
        for model_name, model_path in MODEL_PATHS.items()
    }


MODELS = load_models()


# ============================================================
# Text preprocessing
# ============================================================

def clean_text(text):
    """Clean ticket text using the same logic used during model training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Decision-score analysis
# ============================================================

def confidence_from_margin(margin):
    """Convert top-vs-second decision-score margin into a readable label."""
    if margin is None:
        return "Not available"
    if margin >= 1.0:
        return "High confidence"
    if margin >= 0.4:
        return "Medium confidence"
    return "Low confidence"


def get_prediction_analysis(model, text):
    """
    Predict class and return ranked decision scores when available.

    Decision scores are not probabilities. Higher score means stronger
    model preference.
    """
    prediction = model.predict([text])[0]

    if not hasattr(model, "decision_function"):
        return {
            "prediction": prediction,
            "confidence": "Not available",
            "margin": None,
            "ranked_scores": [],
        }

    scores = model.decision_function([text])
    classes = model.classes_

    # Binary classifiers may return shape (1,)
    if scores.ndim == 1:
        score = float(scores[0])
        scores = np.array([-score, score])
    else:
        scores = scores[0]

    ranked_indices = np.argsort(scores)[::-1]
    top_idx = ranked_indices[0]
    second_idx = ranked_indices[1]

    margin = float(scores[top_idx] - scores[second_idx])

    ranked_scores = [
        {
            "class": classes[i],
            "score": float(scores[i]),
        }
        for i in ranked_indices
    ]

    return {
        "prediction": prediction,
        "confidence": confidence_from_margin(margin),
        "margin": margin,
        "ranked_scores": ranked_scores,
    }


# ============================================================
# German sentiment fallback
# ============================================================

GERMAN_NEGATIVE_PHRASES = [
    "funktioniert nicht",
    "nicht funktioniert",
    "geht nicht",
    "klappt nicht",
    "nicht gut",
    "sehr schlecht",
]

GERMAN_NEGATIVE_WORDS = [
    "schlecht",
    "defekt",
    "fehler",
    "problem",
    "probleme",
    "kaputt",
    "enttäuscht",
    "unzufrieden",
    "beschwerde",
    "verzögerung",
    "ausfall",
    "störung",
    "dringend",
    "kritisch",
    "inakzeptabel",
    "frustriert",
    "fehlgeschlagen",
]

GERMAN_POSITIVE_WORDS = [
    "gut",
    "zufrieden",
    "danke",
    "hilfreich",
    "gelöst",
    "schnell",
    "perfekt",
    "ausgezeichnet",
    "erfolgreich",
]


def get_german_sentiment_analysis(raw_text):
    """
    Rule-based German sentiment analysis.

    Uses raw text so German characters and phrases like
    'funktioniert nicht' are preserved.
    """
    text = str(raw_text).lower()

    matched_negative_phrases = [
        phrase for phrase in GERMAN_NEGATIVE_PHRASES
        if phrase in text
    ]

    matched_negative_words = [
        word for word in GERMAN_NEGATIVE_WORDS
        if word in text
    ]

    matched_positive_words = [
        word for word in GERMAN_POSITIVE_WORDS
        if word in text
    ]

    negative_score = (2 * len(matched_negative_phrases)) + len(matched_negative_words)
    positive_score = len(matched_positive_words)

    if negative_score > positive_score:
        sentiment = "negative"
    elif positive_score > negative_score:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "method": "German keyword-rule fallback",
        "negative_score": negative_score,
        "positive_score": positive_score,
        "matched_negative_phrases": matched_negative_phrases,
        "matched_negative_words": matched_negative_words,
        "matched_positive_words": matched_positive_words,
    }


def predict_sentiment(raw_text, cleaned_text, language="en"):
    """
    English sentiment uses the trained ML model.
    German sentiment uses keyword-rule fallback because the sentiment
    model was trained on external English review data.
    """
    language = str(language).lower()

    if language in ["de", "german", "deutsch"]:
        return get_german_sentiment_analysis(raw_text)

    return {
        "sentiment": MODELS["sentiment"].predict([cleaned_text])[0],
        "method": "English ML sentiment model",
    }


# ============================================================
# Main prediction
# ============================================================

def predict_ticket(text, language="en", include_analysis=True):
    """Predict ticket type, queue, priority, and sentiment."""
    cleaned = clean_text(text)

    type_analysis = get_prediction_analysis(MODELS["type"], cleaned)
    queue_analysis = get_prediction_analysis(MODELS["queue"], cleaned)
    priority_analysis = get_prediction_analysis(MODELS["priority"], cleaned)

    sentiment_analysis = predict_sentiment(
        raw_text=text,
        cleaned_text=cleaned,
        language=language,
    )

    result = {
        "input_text": str(text).strip(),
        "language": language,
        "clean_text": cleaned,
        "type": type_analysis["prediction"],
        "queue": queue_analysis["prediction"],
        "priority": priority_analysis["prediction"],
        "sentiment": sentiment_analysis["sentiment"],
        "sentiment_method": sentiment_analysis["method"],
        "sentiment_analysis": sentiment_analysis,
    }

    if include_analysis:
        result["analysis"] = {
            "type": type_analysis,
            "queue": queue_analysis,
            "priority": priority_analysis,
        }

    return result


# ============================================================
# Display helpers
# ============================================================

def print_model_usage_note(result):
    """Explain which models/rules were used."""
    language = str(result["language"]).lower()

    print("\nModel Usage:")
    print(
        "- Type, queue, and priority were predicted using trained ML models "
        "from the support-ticket dataset."
    )

    if language in ["de", "german", "deutsch"]:
        print(
            "- These ticket models can handle German because their training data "
            "contained both English and German tickets."
        )
        print(
            "- Sentiment used German keyword rules because the sentiment model "
            "was trained on an external English review dataset."
        )
    else:
        print(
            "- Sentiment was predicted using the trained English ML sentiment model."
        )


def print_sentiment_analysis(result):
    """Print sentiment-specific explanation."""
    details = result["sentiment_analysis"]

    print("\nSentiment Analysis:")
    print(f"Method: {details['method']}")

    if details["method"] != "German keyword-rule fallback":
        return

    print(f"Negative score: {details['negative_score']}")
    print(f"Positive score: {details['positive_score']}")

    if details["matched_negative_phrases"]:
        print("Matched negative phrases:", ", ".join(details["matched_negative_phrases"]))

    if details["matched_negative_words"]:
        print("Matched negative words:", ", ".join(details["matched_negative_words"]))

    if details["matched_positive_words"]:
        print("Matched positive words:", ", ".join(details["matched_positive_words"]))

    print(
        "Decision: sentiment is negative if negative score is higher, "
        "positive if positive score is higher, otherwise neutral."
    )


def print_model_analysis(task_name, details, top_n=5):
    """Print compact model analysis for one prediction task."""
    print("\n" + "-" * 70)
    print(f"{task_name.upper()} MODEL ANALYSIS")
    print("-" * 70)

    print(f"Prediction: {details['prediction']}")
    print(f"Confidence: {details['confidence']}")

    if details["margin"] is not None:
        print(f"Decision margin: {details['margin']:.4f}")

    if details["ranked_scores"]:
        print(f"\nTop {top_n} ranked scores:")
        for item in details["ranked_scores"][:top_n]:
            print(f"{item['class']:<35} {item['score']:.4f}")


def print_summary(result):
    """Print readable ticket prediction output."""
    print("=" * 70)
    print("TICKET ANALYSIS")
    print("=" * 70)

    print("\nInput Ticket:")
    print(result["input_text"])

    print("\nLanguage:", result["language"])

    print("\nFinal Predictions:")
    print(f"Type:      {result['type']}")
    print(f"Queue:     {result['queue']}")
    print(f"Priority:  {result['priority']}")
    print(f"Sentiment: {result['sentiment']}")

    print_model_usage_note(result)
    print_sentiment_analysis(result)

    if "analysis" in result:
        for task_name, details in result["analysis"].items():
            print_model_analysis(task_name, details)

    print("\nNote:")
    print("Decision scores are not probabilities.")
    print("Higher score means stronger model preference.")
    print("Margin = top score - second-best score.")


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":
    english_ticket = """
    I was charged twice for my subscription.
    Please refund the duplicate payment.
    """

    german_ticket = """
    Das Produkt funktioniert nicht und ich bin sehr enttäuscht.
    Bitte helfen Sie mir so schnell wie möglich.
    """

    print("\n\nENGLISH TICKET TEST")
    english_result = predict_ticket(
        english_ticket,
        language="en",
        include_analysis=True,
    )
    print_summary(english_result)

    print("\n\nGERMAN TICKET TEST")
    german_result = predict_ticket(
        german_ticket,
        language="de",
        include_analysis=True,
    )
    print_summary(german_result)
