"""
Reply generation pipeline for the Customer Support Copilot.

Default behavior:
- Loads trained ML models.
- Retrieves relevant policy context using multilingual RAG.
- Detects English/German ticket language.
- Generates a human-reviewable, policy-led template reply.

Optional behavior:
- If use_llm=True and an OpenAI API key is configured, generates an LLM-based reply.
- If LLM generation fails, the code falls back to the template reply.

Design note:
- ML predictions are kept as internal decision-support metadata.
- Customer-facing replies are policy-led and do not expose raw model labels such as
  "Incident", "Problem", or the predicted queue. This avoids showing noisy model
  predictions directly to customers while still preserving them for dashboards,
  routing, FastAPI responses, and agent review.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
from langdetect import detect

try:
    from .rag_pipeline import retrieve_policy_context
except ImportError:  # Allows running this file directly from src/
    from rag_pipeline import retrieve_policy_context


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_type_baseline.pkl")
QUEUE_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_queue_baseline.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_priority_baseline.pkl")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


POLICY_ROUTING_HINTS: Dict[str, Dict[str, str]] = {
    "refund_policy.txt": {
        "queue": "Billing or Customer Support",
        "action_en": "review refund eligibility and ask for proof of purchase",
        "action_de": "die Erstattungsberechtigung prüfen und gegebenenfalls einen Kaufnachweis anfordern",
    },
    "shipping_policy.txt": {
        "queue": "Logistics or Customer Support",
        "action_en": "check tracking status and verify courier information",
        "action_de": "den Sendungsverlauf prüfen und die Kurierinformationen verifizieren",
    },
    "account_policy.txt": {
        "queue": "Account Support",
        "action_en": "verify identity and check account access status",
        "action_de": "die Identität verifizieren und den Kontozugriffsstatus prüfen",
    },
    "technical_support_policy.txt": {
        "queue": "Technical Support",
        "action_en": "collect device details, error messages, and troubleshooting steps already tried",
        "action_de": "Gerätedetails, Fehlermeldungen und bereits versuchte Schritte zur Fehlerbehebung erfassen",
    },
    "warranty_policy.txt": {
        "queue": "Warranty Support or Technical Support",
        "action_en": "verify purchase date and warranty eligibility",
        "action_de": "das Kaufdatum und die Garantiebedingungen prüfen",
    },
}


def get_policy_guidance(policy_source: str) -> Dict[str, str]:
    """
    Return policy-based routing/action guidance.

    This is separate from the ML-predicted queue. The ML queue remains useful
    internally, while this policy guidance helps keep replies grounded in the
    retrieved company policy.
    """
    normalized_source = os.path.basename(str(policy_source))
    return POLICY_ROUTING_HINTS.get(
        normalized_source,
        {
            "queue": "Customer Support",
            "action_en": "review the issue manually",
            "action_de": "das Anliegen manuell prüfen",
        },
    )


def detect_language_safe(text: str) -> str:
    """Detect ticket language safely. Returns 'en', 'de', or 'unknown/other code'."""
    try:
        text = str(text).strip()
        if len(text) < 20:
            return "unknown"
        detected = detect(text)
        return detected if detected else "unknown"
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def load_models() -> Dict[str, Any]:
    """Load all trained ML models once and cache them."""
    model_paths = {
        "type_model": TYPE_MODEL_PATH,
        "queue_model": QUEUE_MODEL_PATH,
        "priority_model": PRIORITY_MODEL_PATH,
        "sentiment_model": SENTIMENT_MODEL_PATH,
    }

    missing = [path for path in model_paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "One or more model files are missing:\n"
            + "\n".join(missing)
            + "\nTrain/save the models before running reply generation."
        )

    return {name: joblib.load(path) for name, path in model_paths.items()}


def format_policy_context(retrieved_policy: List[Dict[str, str]]) -> str:
    """Format retrieved policy chunks for display or LLM prompting."""
    if not retrieved_policy:
        return "No relevant policy context found."

    context_parts = []
    for i, item in enumerate(retrieved_policy, start=1):
        source = item.get("source", "unknown_source")
        content = item.get("content", "").strip()
        context_parts.append(f"[Policy Source {i}: {source}]\n{content}")

    return "\n\n".join(context_parts)


def get_opening_line(predicted_sentiment: str, language: str = "en") -> str:
    """Use sentiment internally to choose a suitable tone without exposing the label."""
    sentiment_value = str(predicted_sentiment).lower()

    if language == "de":
        if sentiment_value in {"negative", "neg", "0"}:
            return "Es tut uns leid zu hören, dass Sie dieses Problem haben."
        if sentiment_value in {"positive", "pos", "1"}:
            return "Vielen Dank für Ihre Nachricht."
        return "Vielen Dank, dass Sie den Kundensupport kontaktiert haben."

    if sentiment_value in {"negative", "neg", "0"}:
        return "We’re sorry to hear about the issue you’re facing."
    if sentiment_value in {"positive", "pos", "1"}:
        return "Thank you for reaching out to us."
    return "Thank you for contacting customer support."


def get_priority_line(predicted_priority: str, language: str = "en") -> str:
    """Use priority internally to adjust urgency wording without exposing the label."""
    priority_value = str(predicted_priority).lower()

    if language == "de":
        if priority_value in {"high", "urgent", "critical"}:
            return "Wir verstehen, dass dieses Anliegen dringend sein kann und entsprechend zeitnah geprüft werden sollte."
        if priority_value == "medium":
            return "Ihre Anfrage wurde erfasst und sollte vom zuständigen Support-Team bearbeitet werden."
        return "Ihre Anfrage wurde aufgenommen und wird von unserem Support-Team geprüft."

    if priority_value in {"high", "urgent", "critical"}:
        return "We understand this may need prompt attention, so the support team should review it carefully."
    if priority_value == "medium":
        return "Your request has been noted and should be handled by the appropriate support team."
    return "Your request has been recorded and will be reviewed by our support team."


def generate_template_reply(
    ticket_text: str,
    predicted_type: str,
    predicted_queue: str,
    predicted_priority: str,
    predicted_sentiment: str,
    retrieved_policy: List[Dict[str, str]],
    language: str = "en",
) -> Dict[str, str]:
    """
    Generate a human-reviewable template reply.

    The customer-facing draft is policy-led:
    - It uses sentiment for tone.
    - It uses priority for urgency wording.
    - It uses retrieved policy for grounding.
    - It does not expose raw ML labels such as predicted type or queue.
    """
    policy_text = retrieved_policy[0]["content"] if retrieved_policy else "No relevant policy context found."
    policy_source = retrieved_policy[0]["source"] if retrieved_policy else "No policy source"
    full_policy_context = format_policy_context(retrieved_policy)
    guidance = get_policy_guidance(policy_source)

    opening = get_opening_line(predicted_sentiment, language)
    priority_line = get_priority_line(predicted_priority, language)

    if language == "de":
        reply = f"""
Sehr geehrte Kundin, sehr geehrter Kunde,

{opening}

Basierend auf Ihrer Nachricht wurde eine relevante Unternehmensrichtlinie zu diesem Anliegen gefunden.

{priority_line}

Richtlinienhinweis für das Support-Team:
Es wurde eine relevante Unternehmensrichtlinie gefunden: {policy_source}. Die Richtliniendokumente liegen aktuell auf Englisch vor und sollten vor dem Versenden geprüft werden.

Empfohlener nächster Schritt:
Bitte halten Sie Bestelldaten, Produktinformationen, Screenshots oder Kaufnachweise bereit, falls diese vom Support-Team benötigt werden.

Interner Bearbeitungshinweis:
Das Support-Team sollte {guidance["action_de"]}.

Mit freundlichen Grüßen,
Customer Support Team
""".strip()

    else:
        reply = f"""
Dear Customer,

{opening}

Based on your message, we found a relevant company policy related to this issue.

{priority_line}

Relevant policy reference:
{policy_text}

Recommended next step:
Please keep your order details, product information, screenshots, or proof of purchase available if required by the support team.

Internal handling note:
This request may require the support team to {guidance["action_en"]}.

Best regards,
Customer Support Team
""".strip()

    return {
        "reply": reply,
        "policy_source": policy_source,
        "policy_context": full_policy_context,
        "policy_suggested_queue": guidance["queue"],
        "policy_suggested_action": guidance["action_en"],
        "policy_suggested_action_de": guidance["action_de"],
    }


def generate_llm_reply(
    ticket_text: str,
    predicted_type: str,
    predicted_queue: str,
    predicted_priority: str,
    predicted_sentiment: str,
    retrieved_policy: List[Dict[str, str]],
    language: str = "en",
    model: Optional[str] = None,
) -> str:
    """
    Optional LLM-based reply generation.

    Requires:
    - OPENAI_API_KEY in environment or .env
    - openai and python-dotenv installed

    This function is not used unless use_llm=True.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Falling back to template reply is recommended.")

    client = OpenAI()
    selected_model = model or DEFAULT_OPENAI_MODEL
    policy_context = format_policy_context(retrieved_policy)
    policy_source = retrieved_policy[0]["source"] if retrieved_policy else "No policy source"
    guidance = get_policy_guidance(policy_source)
    reply_language = "German" if language == "de" else "English"

    instructions = f"""
You are a professional customer support assistant.

Generate a concise, polite, human-reviewable customer support reply.

Rules:
- Reply in {reply_language}.
- Use only the provided company policy context.
- Do not invent refund periods, warranty terms, shipping timelines, or escalation promises.
- Do not mention internal ML model names.
- Do not expose raw labels such as ticket type, predicted queue, priority, or sentiment.
- Use the internal predictions only to adjust tone, urgency, and support-team handling.
- If policy context is insufficient, say the support team will review the issue manually.
""".strip()

    user_input = f"""
Customer ticket:
{ticket_text}

Internal predictions for agent decision support only:
Ticket Type: {predicted_type}
Predicted Queue: {predicted_queue}
Priority: {predicted_priority}
Sentiment: {predicted_sentiment}

Policy-based handling guidance:
Suggested Queue: {guidance["queue"]}
Suggested Action: {guidance["action_en"]}

Retrieved company policy context:
{policy_context}

Generate the final customer support reply.
""".strip()

    response = client.responses.create(
        model=selected_model,
        instructions=instructions,
        input=user_input,
    )

    return response.output_text


def generate_support_reply(
    ticket_text: str,
    predicted_type: str,
    predicted_queue: str,
    predicted_priority: str,
    predicted_sentiment: str,
    retrieved_policy: List[Dict[str, str]],
    language: str = "en",
    use_llm: bool = False,
) -> Dict[str, str]:
    """Generate final reply using optional LLM, with template fallback."""
    template_result = generate_template_reply(
        ticket_text=ticket_text,
        predicted_type=predicted_type,
        predicted_queue=predicted_queue,
        predicted_priority=predicted_priority,
        predicted_sentiment=predicted_sentiment,
        retrieved_policy=retrieved_policy,
        language=language,
    )

    if not use_llm:
        template_result["generation_mode"] = "template"
        return template_result

    try:
        llm_reply = generate_llm_reply(
            ticket_text=ticket_text,
            predicted_type=predicted_type,
            predicted_queue=predicted_queue,
            predicted_priority=predicted_priority,
            predicted_sentiment=predicted_sentiment,
            retrieved_policy=retrieved_policy,
            language=language,
        )
        return {
            "reply": llm_reply,
            "policy_source": template_result["policy_source"],
            "policy_context": template_result["policy_context"],
            "policy_suggested_queue": template_result["policy_suggested_queue"],
            "policy_suggested_action": template_result["policy_suggested_action"],
            "policy_suggested_action_de": template_result["policy_suggested_action_de"],
            "generation_mode": "llm",
        }
    except Exception as exc:
        template_result["generation_mode"] = "template_fallback"
        template_result["llm_error"] = str(exc)
        return template_result


def analyze_ticket_and_generate_reply(ticket_text: str, use_llm: bool = False) -> Dict[str, Any]:
    """
    Run ticket classification, queue prediction, priority prediction, sentiment
    analysis, multilingual RAG retrieval, language detection, and reply generation.
    """
    models = load_models()
    language = detect_language_safe(ticket_text)

    # These models are assumed to be saved sklearn Pipelines, so they can accept raw text.
    # If your models were trained on pre-cleaned text outside a Pipeline, apply the exact
    # same preprocessing here before prediction.
    predicted_type = models["type_model"].predict([ticket_text])[0]
    predicted_queue = models["queue_model"].predict([ticket_text])[0]
    predicted_priority = models["priority_model"].predict([ticket_text])[0]
    predicted_sentiment = models["sentiment_model"].predict([ticket_text])[0]

    retrieved_policy = retrieve_policy_context(ticket_text, top_k=3)

    reply_result = generate_support_reply(
        ticket_text=ticket_text,
        predicted_type=predicted_type,
        predicted_queue=predicted_queue,
        predicted_priority=predicted_priority,
        predicted_sentiment=predicted_sentiment,
        retrieved_policy=retrieved_policy,
        language=language,
        use_llm=use_llm,
    )

    model_predictions = {
        "ticket_type": predicted_type,
        "queue": predicted_queue,
        "priority": predicted_priority,
        "sentiment": predicted_sentiment,
    }

    result = {
        "ticket": ticket_text,
        "language": language,
        "model_predictions": model_predictions,
        "policy_source": reply_result["policy_source"],
        "policy_context": reply_result["policy_context"],
        "policy_suggested_queue": reply_result["policy_suggested_queue"],
        "policy_suggested_action": reply_result["policy_suggested_action"],
        "generation_mode": reply_result.get("generation_mode", "template"),
        "generated_reply": reply_result["reply"],
        "llm_error": reply_result.get("llm_error"),
    }

    # Backward-compatible flat keys for notebooks or scripts that still expect them.
    result.update(
        {
            "predicted_type": predicted_type,
            "predicted_queue": predicted_queue,
            "predicted_priority": predicted_priority,
            "predicted_sentiment": predicted_sentiment,
        }
    )

    return result


if __name__ == "__main__":
    sample_tickets = [
        "My laptop arrived damaged and I want a refund.",
        "Ich habe einen beschädigten Laptop erhalten und möchte eine Rückerstattung.",
    ]

    for sample_ticket in sample_tickets:
        result = analyze_ticket_and_generate_reply(sample_ticket, use_llm=False)
        print("=" * 100)
        print("Ticket:", result["ticket"])
        print("Language:", result["language"])
        print("Model Predictions:", result["model_predictions"])
        print("Policy Source:", result["policy_source"])
        print("Policy Suggested Queue:", result["policy_suggested_queue"])
        print("Policy Suggested Action:", result["policy_suggested_action"])
        print("Generation Mode:", result["generation_mode"])
        print("\nGenerated Reply:\n")
        print(result["generated_reply"])
