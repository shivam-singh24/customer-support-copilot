"""
Reply generation pipeline for the Customer Support Copilot.

Default behavior:
- Loads trained ML models.
- Detects English/German ticket language.
- Uses a trained intent router to decide whether RAG is needed.
- Retrieves relevant policy context using multilingual RAG only for support-policy issues.
- Generates a human-reviewable, policy-led or intent-led template reply.

Optional behavior:
- If use_llm=True and an OpenAI API key is configured, generates an LLM-based reply for support-policy issues.
- If LLM generation fails, the code falls back to the template reply.

Design note:
- ML predictions are returned as internal metadata.
- The customer-facing reply avoids exposing raw labels such as Incident, Problem, or predicted queue.
- Product/sales/feedback/ambiguous messages skip RAG so FAISS does not force an irrelevant policy.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import joblib
from langdetect import detect

try:
    from .rag_pipeline import retrieve_policy_context
    from .sentiment_utils import predict_sentiment as predict_sentiment_for_language
except ImportError:  # Allows running this file directly from src/
    from rag_pipeline import retrieve_policy_context
    from sentiment_utils import predict_sentiment as predict_sentiment_for_language


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
POLICY_DOC_DIR = os.path.join(BASE_DIR, "docs", "company_policies")

TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_type_baseline.pkl")
QUEUE_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_queue_baseline.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_priority_baseline.pkl")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
INTENT_ROUTER_PATH = os.path.join(MODEL_DIR, "intent_router.pkl")

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
LOW_INTENT_CONFIDENCE_THRESHOLD = 0.45

NON_POLICY_INTENTS = {
    "sales_or_product_inquiry",
    "positive_feedback",
    "ambiguous",
    "general_inquiry",
}

SUPPORT_POLICY_INTENTS = {
    "refund_issue",
    "shipping_issue",
    "account_issue",
    "technical_issue",
    "warranty_issue",
    # Backward-compatible fallback label from the older rule router.
    "support_policy_issue",
}

ALL_INTENTS = NON_POLICY_INTENTS.union(SUPPORT_POLICY_INTENTS)

INTENT_TO_POLICY = {
    "refund_issue": "refund_policy.txt",
    "shipping_issue": "shipping_policy.txt",
    "account_issue": "account_policy.txt",
    "technical_issue": "technical_support_policy.txt",
    "warranty_issue": "warranty_policy.txt",
}

POLICY_ROUTING_HINTS = {
    "refund_policy.txt": {
        "queue": "Billing or Customer Support",
        "action": "review refund eligibility and ask for proof of purchase",
        "customer_note_en": (
            "The support team will review refund eligibility and may ask for proof of purchase. "
            "For damaged products, a refund or replacement may be available depending on policy conditions."
        ),
        "customer_note_de": (
            "Das Support-Team wird die Erstattungsberechtigung prüfen und gegebenenfalls einen Kaufnachweis anfordern. "
            "Bei beschädigten Produkten kann je nach Richtlinie eine Erstattung oder ein Ersatz möglich sein."
        ),
    },
    "shipping_policy.txt": {
        "queue": "Logistics or Customer Support",
        "action": "check tracking status and verify courier information",
        "customer_note_en": (
            "The support team will check the tracking status and courier information before deciding the next step."
        ),
        "customer_note_de": (
            "Das Support-Team wird den Sendungsstatus und die Kurierinformationen prüfen, bevor die nächsten Schritte festgelegt werden."
        ),
    },
    "account_policy.txt": {
        "queue": "Account Support",
        "action": "verify identity and check account access status",
        "customer_note_en": (
            "The support team may need to verify your identity before helping with account access or account changes."
        ),
        "customer_note_de": (
            "Das Support-Team muss möglicherweise Ihre Identität prüfen, bevor es bei Kontozugriff oder Kontoänderungen helfen kann."
        ),
    },
    "technical_support_policy.txt": {
        "queue": "Technical Support",
        "action": "collect device details, error messages, and troubleshooting steps",
        "customer_note_en": (
            "The support team may ask for product details, device model, error messages, and troubleshooting steps already tried."
        ),
        "customer_note_de": (
            "Das Support-Team kann Produktdetails, Gerätemodell, Fehlermeldungen und bereits ausprobierte Schritte zur Fehlerbehebung anfordern."
        ),
    },
    "warranty_policy.txt": {
        "queue": "Warranty Support or Technical Support",
        "action": "verify purchase date and warranty eligibility",
        "customer_note_en": (
            "The support team will review the purchase date and warranty eligibility before suggesting repair, replacement, or service support."
        ),
        "customer_note_de": (
            "Das Support-Team wird Kaufdatum und Garantieanspruch prüfen, bevor Reparatur, Ersatz oder Serviceunterstützung empfohlen wird."
        ),
    },
}


def get_policy_guidance(policy_source: str) -> Dict[str, str]:
    """Return internal routing/action hints and customer-safe policy note."""
    return POLICY_ROUTING_HINTS.get(
        policy_source,
        {
            "queue": "Customer Support",
            "action": "review the issue manually",
            "customer_note_en": "The support team will review your request and determine the appropriate next step.",
            "customer_note_de": "Das Support-Team wird Ihre Anfrage prüfen und die passenden nächsten Schritte festlegen.",
        },
    )


def detect_language_safe(text: str) -> str:
    """
    Detect ticket language safely. Returns 'en', 'de', or 'unknown'.

    For short tickets, langdetect can be unreliable, so a few German hints
    are checked before falling back to 'unknown'.
    """
    try:
        text = str(text).strip()
        lowered = text.lower()

        german_hints = [
            "ich ", "mein ", "meine ", "möchte", "rückerstattung", "konto",
            "passwort", "hilfe", "bitte", "paket", "garantie", "beschädigt",
            "funktioniert", "danke", "preis", "verfügbarkeit", "geld zurück",
        ]
        if any(hint in f" {lowered} " for hint in german_hints):
            return "de"

        if len(text) < 20:
            return "unknown"

        detected = detect(text)
        return detected if detected in {"en", "de"} else detected
    except Exception:
        return "unknown"


def detect_workflow_intent_rule_fallback(ticket_text: str) -> str:
    """
    Lightweight fallback router used only when models/intent_router.pkl is missing
    or returns an unexpected label.
    """
    text = str(ticket_text).lower().strip()
    words = text.split()

    if not text:
        return "ambiguous"

    ambiguous_phrases = [
        "help me", "help please", "need help", "need support", "i have a problem",
        "laptop issue", "laptop problem", "hilfe bitte", "ich brauche hilfe",
        "ich habe ein problem", "problem hier",
    ]

    refund_keywords = [
        "refund", "return", "money back", "wrong item", "charged twice", "duplicate payment",
        "rückerstattung", "geld zurück", "falschen artikel", "falscher artikel", "doppelt belastet", "zurückgeben",
    ]
    shipping_keywords = [
        "not arrived", "tracking", "lost package", "delivered but not received",
        "paket", "sendungsverfolgung", "zugestellt", "nicht erhalten", "nicht angekommen", "verloren",
    ]
    account_keywords = [
        "cannot login", "can't login", "password reset", "account hacked", "compromised", "locked account",
        "einloggen", "passwort", "konto gehackt", "gehackt", "konto gesperrt",
    ]
    warranty_keywords = [
        "warranty", "under warranty", "within warranty", "garantie", "garantiezeit", "innerhalb der garantie",
    ]
    technical_keywords = [
        "not working", "does not work", "doesn't work", "damaged", "broken", "defective", "error", "failed",
        "crash", "crashing", "battery drains", "charger stopped", "funktioniert nicht", "beschädigt", "kaputt",
        "defekt", "fehler", "stürzt", "akku", "ladegerät",
    ]
    sales_product_keywords = [
        "want to know", "tell me about", "information about", "details about", "price", "pricing", "cost",
        "available", "availability", "specs", "specification", "ryzen", "intel", "buy", "purchase",
        "want apple ipad", "want an ipad", "looking for an ipad",
        "want one more", "buy another", "interested in buying", "möchte wissen", "informationen über", "preis",
        "kosten", "verfügbarkeit", "verfügbar", "kaufen", "weiteres kaufen",
    ]
    positive_keywords = [
        "working well", "works great", "excellent", "excellently", "happy with", "satisfied", "thank you",
        "thanks", "great product", "funktioniert gut", "sehr gut", "zufrieden", "danke", "tolles produkt",
    ]

    if any(keyword in text for keyword in refund_keywords):
        return "refund_issue"
    if any(keyword in text for keyword in shipping_keywords):
        return "shipping_issue"
    if any(keyword in text for keyword in account_keywords):
        return "account_issue"
    if any(keyword in text for keyword in warranty_keywords):
        return "warranty_issue"
    if any(keyword in text for keyword in technical_keywords):
        return "technical_issue"
    if any(keyword in text for keyword in sales_product_keywords):
        return "sales_or_product_inquiry"
    if any(keyword in text for keyword in positive_keywords):
        return "positive_feedback"
    if any(phrase in text for phrase in ambiguous_phrases):
        return "ambiguous"
    if len(words) < 4:
        return "ambiguous"

    return "general_inquiry"


@lru_cache(maxsize=1)
def load_models() -> Dict[str, Any]:
    """Load trained ML models once and cache them."""
    required_model_paths = {
        "type_model": TYPE_MODEL_PATH,
        "queue_model": QUEUE_MODEL_PATH,
        "priority_model": PRIORITY_MODEL_PATH,
        "sentiment_model": SENTIMENT_MODEL_PATH,
    }

    missing = [path for path in required_model_paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "One or more required model files are missing:\n"
            + "\n".join(missing)
            + "\nTrain/save the models before running reply generation."
        )

    models = {name: joblib.load(path) for name, path in required_model_paths.items()}

    # Intent router is strongly recommended but loaded optionally so the API can still
    # run with the rule fallback if the .pkl has not been copied yet.
    if os.path.exists(INTENT_ROUTER_PATH):
        models["intent_router"] = joblib.load(INTENT_ROUTER_PATH)

    return models


def predict_workflow_intent(models: Dict[str, Any], ticket_text: str) -> Tuple[str, Optional[float], str]:
    """
    Predict workflow intent using models/intent_router.pkl when available.

    Returns:
        (detected_intent, confidence, source)
    """
    router = models.get("intent_router")
    if router is None:
        return detect_workflow_intent_rule_fallback(ticket_text), None, "rule_fallback"

    try:
        intent = str(router.predict([ticket_text])[0])
        confidence: Optional[float] = None

        if hasattr(router, "predict_proba"):
            probabilities = router.predict_proba([ticket_text])[0]
            confidence = float(max(probabilities))

        if intent not in ALL_INTENTS:
            fallback_intent = detect_workflow_intent_rule_fallback(ticket_text)
            return fallback_intent, confidence, "model_unexpected_label_fallback"

        return intent, confidence, "model"

    except Exception:
        return detect_workflow_intent_rule_fallback(ticket_text), None, "model_error_fallback"


def format_policy_context(retrieved_policy: List[Dict[str, str]]) -> str:
    """Format retrieved policy chunks for display, API metadata, or LLM prompting."""
    if not retrieved_policy:
        return "No relevant policy context found."

    context_parts = []
    for i, item in enumerate(retrieved_policy, start=1):
        source = item.get("source", "unknown_source")
        content = item.get("content", "").strip()
        context_parts.append(f"[Policy Source {i}: {source}]\n{content}")

    return "\n\n".join(context_parts)


def load_policy_file(policy_source: str) -> List[Dict[str, str]]:
    """
    Load a specific policy file directly from docs/company_policies.

    Used as fallback when the intent router knows the correct policy but FAISS
    retrieves a different policy first.
    """
    if not policy_source:
        return []

    policy_path = os.path.join(POLICY_DOC_DIR, policy_source)
    if not os.path.exists(policy_path):
        return []

    with open(policy_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    return [{"source": policy_source, "content": content}]


def prioritize_or_load_preferred_policy(
    retrieved_policy: List[Dict[str, str]],
    preferred_policy: Optional[str],
) -> List[Dict[str, str]]:
    """
    Move preferred policy to the top if FAISS retrieved it.
    If FAISS missed it, load it directly from docs/company_policies.
    """
    if not preferred_policy:
        return retrieved_policy

    preferred_items = [
        item for item in retrieved_policy
        if item.get("source") == preferred_policy
    ]
    other_items = [
        item for item in retrieved_policy
        if item.get("source") != preferred_policy
    ]

    if preferred_items:
        return preferred_items + other_items

    direct_policy = load_policy_file(preferred_policy)
    if direct_policy:
        return direct_policy + retrieved_policy

    return retrieved_policy


def generate_non_policy_reply(language: str, intent: str) -> Dict[str, str]:
    """Generate replies for messages that do not need policy retrieval."""
    if intent == "sales_or_product_inquiry":
        if language == "de":
            reply = """
Sehr geehrte Kundin, sehr geehrter Kunde,

vielen Dank für Ihre Nachricht.

Ihre Anfrage scheint eine Produkt- oder Verkaufsanfrage zu sein. Unser Sales- oder Customer-Service-Team kann Ihnen bei Produktdetails, Verfügbarkeit, Preisen und nächsten Schritten helfen.

Mit freundlichen Grüßen,
Customer Support Team
""".strip()
        else:
            reply = """
Dear Customer,

Thank you for reaching out.

Your message appears to be a product or sales inquiry rather than a technical support issue. Our Sales or Customer Service team can help with product details, availability, pricing, and next steps.

Best regards,
Customer Support Team
""".strip()

        return {
            "reply": reply,
            "policy_source": "No policy needed",
            "policy_context": "No policy context required for this sales/product inquiry.",
            "policy_suggested_queue": "Sales or Customer Service",
            "policy_suggested_action": "assist with product information, availability, pricing, or purchase guidance",
            "generation_mode": "template",
        }

    if intent == "positive_feedback":
        if language == "de":
            reply = """
Sehr geehrte Kundin, sehr geehrter Kunde,

vielen Dank für Ihre positive Rückmeldung.

Es freut uns zu hören, dass Sie mit dem Produkt zufrieden sind. Falls Sie weitere Unterstützung benötigen oder ein weiteres Produkt kaufen möchten, hilft Ihnen unser Customer-Service- oder Sales-Team gerne weiter.

Mit freundlichen Grüßen,
Customer Support Team
""".strip()
        else:
            reply = """
Dear Customer,

Thank you for your positive feedback.

We're glad to hear that your product is working well. If you need further assistance or would like to purchase another item, our Customer Service or Sales team will be happy to help.

Best regards,
Customer Support Team
""".strip()

        return {
            "reply": reply,
            "policy_source": "No policy needed",
            "policy_context": "No policy context required for positive feedback.",
            "policy_suggested_queue": "Customer Service or Sales",
            "policy_suggested_action": "acknowledge feedback and guide the customer if they need further help",
            "generation_mode": "template",
        }

    if intent == "ambiguous":
        if language == "de":
            reply = """
Sehr geehrte Kundin, sehr geehrter Kunde,

vielen Dank für Ihre Nachricht.

Damit unser Support-Team Ihnen gezielt helfen kann, teilen Sie bitte weitere Details zu Ihrem Anliegen mit, zum Beispiel Produktname, Problembeschreibung, Bestellnummer, Screenshots oder Fehlermeldungen.

Mit freundlichen Grüßen,
Customer Support Team
""".strip()
        else:
            reply = """
Dear Customer,

Thank you for contacting us.

To help our support team assist you properly, please share more details about your request, such as the product name, issue description, order number, screenshots, or any error messages.

Best regards,
Customer Support Team
""".strip()

        return {
            "reply": reply,
            "policy_source": "No policy needed",
            "policy_context": "No policy context retrieved because the ticket is ambiguous.",
            "policy_suggested_queue": "Customer Support",
            "policy_suggested_action": "ask the customer for more details before routing",
            "generation_mode": "template",
        }

    # general_inquiry fallback
    if language == "de":
        reply = """
Sehr geehrte Kundin, sehr geehrter Kunde,

vielen Dank für Ihre Nachricht.

Ihr Anliegen scheint eine allgemeine Anfrage zu sein. Unser Customer-Service-Team wird Ihre Anfrage prüfen und Sie bei den nächsten Schritten unterstützen.

Mit freundlichen Grüßen,
Customer Support Team
""".strip()
    else:
        reply = """
Dear Customer,

Thank you for reaching out.

Your message appears to be a general inquiry. Our Customer Service team will review your request and guide you with the next steps.

Best regards,
Customer Support Team
""".strip()

    return {
        "reply": reply,
        "policy_source": "No policy needed",
        "policy_context": "No policy context required for this general inquiry.",
        "policy_suggested_queue": "Customer Service",
        "policy_suggested_action": "review the request and guide the customer to the correct next step",
        "generation_mode": "template",
    }


def _get_opening_and_priority_lines(
    predicted_sentiment: str,
    predicted_priority: str,
    language: str,
) -> Dict[str, str]:
    """Create customer-safe tone lines using sentiment and priority predictions."""
    sentiment_value = str(predicted_sentiment).lower()
    priority_value = str(predicted_priority).lower()

    if language == "de":
        if sentiment_value in {"negative", "neg", "0"}:
            opening = "Es tut uns leid zu hören, dass Sie dieses Problem haben."
        elif sentiment_value in {"positive", "pos", "1"}:
            opening = "Vielen Dank für Ihre Nachricht."
        else:
            opening = "Vielen Dank, dass Sie den Kundensupport kontaktiert haben."

        if priority_value in {"high", "urgent", "critical"}:
            priority_line = (
                "Wir verstehen, dass dieses Anliegen dringend sein kann. "
                "Unser Support-Team wird Ihre Anfrage sorgfältig prüfen."
            )
        elif priority_value == "medium":
            priority_line = "Ihre Anfrage wurde erfasst und wird vom zuständigen Support-Team geprüft."
        else:
            priority_line = "Ihre Anfrage wurde aufgenommen und wird von unserem Support-Team geprüft."

        return {"opening": opening, "priority_line": priority_line}

    if sentiment_value in {"negative", "neg", "0"}:
        opening = "We're sorry to hear about the issue you're facing."
    elif sentiment_value in {"positive", "pos", "1"}:
        opening = "Thank you for reaching out to us."
    else:
        opening = "Thank you for contacting customer support."

    if priority_value in {"high", "urgent", "critical"}:
        priority_line = "We understand this may need prompt attention, so the support team will review it carefully."
    elif priority_value == "medium":
        priority_line = "Your request has been noted and will be reviewed by the appropriate support team."
    else:
        priority_line = "Your request has been recorded and will be reviewed by our support team."

    return {"opening": opening, "priority_line": priority_line}


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
    Generate a human-reviewable, policy-led template reply.

    Important:
    - The customer-facing reply does not expose raw ML labels.
    - ML predictions are returned separately as metadata by analyze_ticket_and_generate_reply().
    - Full policy context is returned separately for agent/API review.
    """
    policy_source = retrieved_policy[0]["source"] if retrieved_policy else "No policy source"
    full_policy_context = format_policy_context(retrieved_policy)
    guidance = get_policy_guidance(policy_source)

    tone = _get_opening_and_priority_lines(
        predicted_sentiment=predicted_sentiment,
        predicted_priority=predicted_priority,
        language=language,
    )

    if language == "de":
        reply = f"""
Sehr geehrte Kundin, sehr geehrter Kunde,

{tone["opening"]}

Basierend auf Ihrer Nachricht wurde eine relevante Unternehmensrichtlinie zu diesem Anliegen gefunden.

{tone["priority_line"]}

Richtlinienhinweis:
{guidance["customer_note_de"]}

Empfohlener nächster Schritt:
Bitte halten Sie Bestelldaten, Produktinformationen, Screenshots oder Kaufnachweise bereit, falls diese vom Support-Team benötigt werden.

Mit freundlichen Grüßen,
Customer Support Team
""".strip()

    else:
        reply = f"""
Dear Customer,

{tone["opening"]}

Based on your message, we found a relevant company policy related to this issue.

{tone["priority_line"]}

Policy-guided note:
{guidance["customer_note_en"]}

Recommended next step:
Please keep your order details, product information, screenshots, or proof of purchase available if required by the support team.

Best regards,
Customer Support Team
""".strip()

    return {
        "reply": reply,
        "policy_source": policy_source,
        "policy_context": full_policy_context,
        "policy_suggested_queue": guidance["queue"],
        "policy_suggested_action": guidance["action"],
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
    """Optional LLM-based reply generation. This function is not used unless use_llm=True."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Falling back to template reply is recommended.")

    client = OpenAI()
    selected_model = model or DEFAULT_OPENAI_MODEL
    policy_context = format_policy_context(retrieved_policy)
    reply_language = "German" if language == "de" else "English"

    instructions = f"""
You are a professional customer support assistant.

Generate a concise, polite, human-reviewable customer support reply.

Rules:
- Reply in {reply_language}.
- Use only the provided company policy context.
- Do not invent refund periods, warranty terms, shipping timelines, or escalation promises.
- Do not mention internal ML model names.
- Do not expose confidence scores or raw internal classification labels.
- If policy context is insufficient, say the support team will review the issue manually.
""".strip()

    user_input = f"""
Customer ticket:
{ticket_text}

Internal metadata for tone/routing only:
Ticket Type: {predicted_type}
Queue: {predicted_queue}
Priority: {predicted_priority}
Sentiment: {predicted_sentiment}

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
) -> Dict[str, Any]:
    """Generate final support-policy reply using optional LLM, with template fallback."""
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
            "generation_mode": "llm",
        }
    except Exception as exc:
        template_result["generation_mode"] = "template_fallback"
        template_result["llm_error"] = str(exc)
        return template_result


def _adjust_final_metadata_for_intent(
    ticket_text: str,
    detected_intent: str,
    predicted_priority: str,
    predicted_sentiment: str,
) -> Tuple[str, str, bool]:
    """Apply conservative workflow-level corrections to priority, sentiment, and review flag."""
    text = str(ticket_text).lower()
    final_priority = predicted_priority
    final_sentiment = predicted_sentiment
    requires_human_review = False

    if detected_intent in {"sales_or_product_inquiry", "positive_feedback"}:
        final_priority = "low"

    if detected_intent == "sales_or_product_inquiry":
        final_sentiment = "neutral"

    if detected_intent == "positive_feedback":
        final_sentiment = "positive"

    # Vague/non-specific messages should not inherit noisy high-priority or negative predictions.
    # They are sent to human review with a neutral, safer operational default.
    if detected_intent in {"ambiguous", "general_inquiry"}:
        final_priority = "medium"
        final_sentiment = "neutral"
        requires_human_review = True

    # Safety override for security/account compromise language.
    if detected_intent == "account_issue" and any(term in text for term in ["hacked", "compromised", "gehackt", "kompromittiert"]):
        final_priority = "high"

    return final_priority, final_sentiment, requires_human_review


def analyze_ticket_and_generate_reply(ticket_text: str, use_llm: bool = False) -> Dict[str, Any]:
    """
    Run ticket classification, workflow routing, optional multilingual RAG retrieval,
    language detection, and reply generation.
    """
    models = load_models()
    language = detect_language_safe(ticket_text)

    detected_intent, intent_confidence, intent_router_source = predict_workflow_intent(
        models=models,
        ticket_text=ticket_text,
    )

    # Abstain from policy routing when the trained router is uncertain.
    # A lightweight rule fallback can still preserve obvious support intents
    # such as refunds or hacked accounts, while vague/product requests skip RAG.
    if intent_confidence is not None and intent_confidence < LOW_INTENT_CONFIDENCE_THRESHOLD:
        fallback_intent = detect_workflow_intent_rule_fallback(ticket_text)
        if fallback_intent not in {"ambiguous", "general_inquiry"}:
            detected_intent = fallback_intent
            intent_router_source = "low_confidence_rule_fallback"

    # These models are assumed to be saved sklearn Pipelines, so they can accept raw text.
    # If your models were trained on pre-cleaned text outside a Pipeline, apply the exact
    # same preprocessing here before prediction.
    predicted_type = models["type_model"].predict([ticket_text])[0]
    predicted_queue = models["queue_model"].predict([ticket_text])[0]
    predicted_priority = models["priority_model"].predict([ticket_text])[0]
    sentiment_analysis = predict_sentiment_for_language(
        raw_text=ticket_text,
        model_text=ticket_text,
        language=language,
        sentiment_model=models["sentiment_model"],
    )
    predicted_sentiment = sentiment_analysis["sentiment"]

    final_priority, final_sentiment, requires_human_review = _adjust_final_metadata_for_intent(
        ticket_text=ticket_text,
        detected_intent=detected_intent,
        predicted_priority=predicted_priority,
        predicted_sentiment=predicted_sentiment,
    )

    rag_used = detected_intent not in NON_POLICY_INTENTS
    preferred_policy = INTENT_TO_POLICY.get(detected_intent)

    model_predictions = {
        "ticket_type": predicted_type,
        "queue": predicted_queue,
        "priority": predicted_priority,
        "sentiment": predicted_sentiment,
        "sentiment_method": sentiment_analysis["method"],
    }

    if not rag_used:
        reply_result = generate_non_policy_reply(
            language=language,
            intent=detected_intent,
        )
    else:
        retrieved_policy = retrieve_policy_context(ticket_text, top_k=5)
        retrieved_policy = prioritize_or_load_preferred_policy(
            retrieved_policy=retrieved_policy,
            preferred_policy=preferred_policy,
        )

        reply_result = generate_support_reply(
            ticket_text=ticket_text,
            predicted_type=predicted_type,
            predicted_queue=predicted_queue,
            predicted_priority=final_priority,
            predicted_sentiment=final_sentiment,
            retrieved_policy=retrieved_policy,
            language=language,
            use_llm=use_llm,
        )

        if reply_result.get("policy_source") in {"No policy source", "No policy needed"}:
            requires_human_review = True

    # Low router confidence means the result should be reviewed even if the pipeline continues.
    # This applies to both policy and non-policy workflows.
    if intent_confidence is not None and intent_confidence < LOW_INTENT_CONFIDENCE_THRESHOLD:
        requires_human_review = True

    return {
        "ticket": ticket_text,
        "language": language,
        "detected_intent": detected_intent,
        "intent_confidence": intent_confidence,
        "intent_router_source": intent_router_source,
        "rag_used": rag_used,
        "requires_human_review": requires_human_review,

        "model_predictions": model_predictions,

        # Backward-compatible fields for older notebooks/scripts.
        "predicted_type": predicted_type,
        "predicted_queue": predicted_queue,
        "predicted_priority": predicted_priority,
        "predicted_sentiment": predicted_sentiment,

        "final_priority": final_priority,
        "final_sentiment": final_sentiment,
        "final_queue_recommendation": reply_result.get("policy_suggested_queue"),

        "policy_source": reply_result["policy_source"],
        "policy_context": reply_result["policy_context"],
        "policy_suggested_queue": reply_result.get("policy_suggested_queue"),
        "policy_suggested_action": reply_result.get("policy_suggested_action"),

        "generation_mode": reply_result.get("generation_mode", "template"),
        "generated_reply": reply_result["reply"],
        "llm_error": reply_result.get("llm_error"),
    }


if __name__ == "__main__":
    sample_tickets = [
        "My laptop arrived damaged and I want a refund.",
        "Ich habe den falschen Artikel erhalten und möchte mein Geld zurück.",
        "Das Paket wird als zugestellt angezeigt, aber ich habe es nicht erhalten.",
        "Ich glaube, mein Konto wurde gehackt.",
        "Die App stürzt jedes Mal ab, wenn ich sie öffne.",
        "i want to know about laptop ryzen 5.",
        "My laptop is excellently working, I want one more.",
        "Help me.",
    ]

    for sample_ticket in sample_tickets:
        result = analyze_ticket_and_generate_reply(sample_ticket, use_llm=False)
        print("=" * 100)
        print("Ticket:", result["ticket"])
        print("Language:", result["language"])
        print("Detected Intent:", result["detected_intent"])
        print("Intent Confidence:", result["intent_confidence"])
        print("Intent Router Source:", result["intent_router_source"])
        print("RAG Used:", result["rag_used"])
        print("Requires Human Review:", result["requires_human_review"])
        print("Model Predictions:", result["model_predictions"])
        print("Final Priority:", result["final_priority"])
        print("Final Sentiment:", result["final_sentiment"])
        print("Final Queue Recommendation:", result["final_queue_recommendation"])
        print("Policy Source:", result["policy_source"])
        print("Policy Suggested Queue:", result["policy_suggested_queue"])
        print("Policy Suggested Action:", result["policy_suggested_action"])
        print("Generation Mode:", result["generation_mode"])
        print("\nGenerated Reply:\n")
        print(result["generated_reply"])
