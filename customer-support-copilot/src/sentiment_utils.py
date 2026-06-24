"""Shared language-aware sentiment helpers for runtime inference.

English tickets use the trained sentiment model. German tickets use a
lightweight keyword-rule fallback because the saved sentiment model was trained
on English review data.
"""

import re
from typing import Any, Dict


GERMAN_NEGATIVE_PHRASES = [
    "funktioniert nicht",
    "nicht funktioniert",
    "geht nicht",
    "klappt nicht",
    "nicht gut",
    "sehr schlecht",
    "nicht erhalten",
    "nicht angekommen",
    "nicht geliefert",
    "nicht mehr",
]

GERMAN_NEGATIVE_WORDS = [
    "schlecht",
    "defekt",
    "fehler",
    "problem",
    "probleme",
    "kaputt",
    "beschädigt",
    "beschädigte",
    "falsch",
    "falschen",
    "enttäuscht",
    "unzufrieden",
    "beschwerde",
    "verzögerung",
    "verspätet",
    "ausfall",
    "störung",
    "dringend",
    "kritisch",
    "inakzeptabel",
    "frustriert",
    "fehlgeschlagen",
    "verloren",
    "gehackt",
    "kompromittiert",
    "gesperrt",
    "absturz",
    "stürzt",
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


def get_german_sentiment_analysis(raw_text: str) -> Dict[str, Any]:
    """Return deterministic keyword-rule sentiment details for German text."""
    text = str(raw_text).lower()

    matched_negative_phrases = [
        phrase for phrase in GERMAN_NEGATIVE_PHRASES if phrase in text
    ]
    matched_negative_words = [
        word
        for word in GERMAN_NEGATIVE_WORDS
        if re.search(rf"\b{re.escape(word)}\b", text)
    ]
    matched_positive_words = [
        word
        for word in GERMAN_POSITIVE_WORDS
        if re.search(rf"\b{re.escape(word)}\b", text)
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


def predict_sentiment(
    *,
    raw_text: str,
    model_text: str,
    language: str,
    sentiment_model: Any,
) -> Dict[str, Any]:
    """Predict sentiment with the language-appropriate runtime method."""
    normalized_language = str(language).lower()

    if normalized_language in {"de", "german", "deutsch"}:
        return get_german_sentiment_analysis(raw_text)

    return {
        "sentiment": sentiment_model.predict([model_text])[0],
        "method": "English ML sentiment model",
    }
