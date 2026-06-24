"""Integration checks for the running FastAPI application.

Start the API first:
    uvicorn api.main:app --reload

Then run:
    python scripts/test_fastapi_outputs.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests


API_URL = os.getenv("COPILOT_API_URL", "http://127.0.0.1:8000/analyze-ticket")
REPORTS_DIR = Path("reports")
OUTPUT_PATH = REPORTS_DIR / "fastapi_test_outputs.json"

SUCCESS_CASES: List[Dict[str, Any]] = [
    {"case_id": 1, "category": "english_refund", "ticket_text": "My laptop arrived damaged and I want a refund.", "expected_language": "en", "expected_intent": "refund_issue", "expected_rag_used": True, "expected_policy_source": "refund_policy.txt", "expected_final_sentiment": "negative", "expected_final_queue": "Billing or Customer Support"},
    {"case_id": 2, "category": "english_shipping", "ticket_text": "My package has not arrived and tracking is not updating.", "expected_language": "en", "expected_intent": "shipping_issue", "expected_rag_used": True, "expected_policy_source": "shipping_policy.txt", "expected_final_queue": "Logistics or Customer Support"},
    {"case_id": 3, "category": "english_account", "ticket_text": "I think my account has been hacked.", "expected_language": "en", "expected_intent": "account_issue", "expected_rag_used": True, "expected_policy_source": "account_policy.txt", "expected_final_priority": "high", "expected_final_queue": "Account Support"},
    {"case_id": 4, "category": "english_technical", "ticket_text": "The app keeps crashing whenever I open it.", "expected_language": "en", "expected_intent": "technical_issue", "expected_rag_used": True, "expected_policy_source": "technical_support_policy.txt", "expected_final_queue": "Technical Support"},
    {"case_id": 5, "category": "english_warranty", "ticket_text": "My charger stopped working within the warranty period.", "expected_language": "en", "expected_intent": "warranty_issue", "expected_rag_used": True, "expected_policy_source": "warranty_policy.txt", "expected_final_queue": "Warranty Support or Technical Support"},
    {"case_id": 6, "category": "english_product_inquiry", "ticket_text": "I want to know about laptop Ryzen 5.", "expected_language": "en", "expected_intent": "sales_or_product_inquiry", "expected_rag_used": False, "expected_policy_source": "No policy needed", "expected_final_priority": "low", "expected_final_sentiment": "neutral"},
    {"case_id": 7, "category": "english_positive_feedback", "ticket_text": "My laptop is working excellently, thank you.", "expected_language": "en", "expected_intent": "positive_feedback", "expected_rag_used": False, "expected_policy_source": "No policy needed", "expected_final_priority": "low", "expected_final_sentiment": "positive"},
    {"case_id": 8, "category": "english_ambiguous", "ticket_text": "Help me.", "expected_language": "unknown", "expected_intent": "ambiguous", "expected_rag_used": False, "expected_policy_source": "No policy needed", "expected_final_priority": "medium", "expected_final_sentiment": "neutral", "expected_human_review": True},
    {"case_id": 9, "category": "german_refund", "ticket_text": "Ich habe den falschen Artikel erhalten und möchte mein Geld zurück.", "expected_language": "de", "expected_intent": "refund_issue", "expected_rag_used": True, "expected_policy_source": "refund_policy.txt", "expected_final_sentiment": "negative", "expected_sentiment_method": "German keyword-rule fallback", "expected_final_queue": "Billing or Customer Support"},
    {"case_id": 10, "category": "german_shipping", "ticket_text": "Das Paket wird als zugestellt angezeigt, aber ich habe es nicht erhalten.", "expected_language": "de", "expected_intent": "shipping_issue", "expected_rag_used": True, "expected_policy_source": "shipping_policy.txt", "expected_final_sentiment": "negative", "expected_sentiment_method": "German keyword-rule fallback", "expected_final_queue": "Logistics or Customer Support"},
    {"case_id": 11, "category": "german_account", "ticket_text": "Ich glaube, mein Konto wurde gehackt.", "expected_language": "de", "expected_intent": "account_issue", "expected_rag_used": True, "expected_policy_source": "account_policy.txt", "expected_final_priority": "high", "expected_final_sentiment": "negative", "expected_sentiment_method": "German keyword-rule fallback", "expected_final_queue": "Account Support"},
    {"case_id": 12, "category": "german_technical", "ticket_text": "Die App stürzt jedes Mal ab, wenn ich sie öffne.", "expected_language": "de", "expected_intent": "technical_issue", "expected_rag_used": True, "expected_policy_source": "technical_support_policy.txt", "expected_final_sentiment": "negative", "expected_sentiment_method": "German keyword-rule fallback", "expected_final_queue": "Technical Support"},
    {"case_id": 13, "category": "german_warranty", "ticket_text": "Mein Ladegerät funktioniert innerhalb der Garantiezeit nicht mehr.", "expected_language": "de", "expected_intent": "warranty_issue", "expected_rag_used": True, "expected_policy_source": "warranty_policy.txt", "expected_final_sentiment": "negative", "expected_sentiment_method": "German keyword-rule fallback", "expected_final_queue": "Warranty Support or Technical Support"},
    {"case_id": 14, "category": "german_product_inquiry", "ticket_text": "Ich möchte Informationen über einen Laptop mit Ryzen 5.", "expected_language": "de", "expected_intent": "sales_or_product_inquiry", "expected_rag_used": False, "expected_policy_source": "No policy needed", "expected_final_priority": "low", "expected_final_sentiment": "neutral", "expected_sentiment_method": "German keyword-rule fallback"},
    {"case_id": 15, "category": "german_positive_feedback", "ticket_text": "Mein Laptop funktioniert sehr gut, danke.", "expected_language": "de", "expected_intent": "positive_feedback", "expected_rag_used": False, "expected_policy_source": "No policy needed", "expected_final_priority": "low", "expected_final_sentiment": "positive", "expected_sentiment_method": "German keyword-rule fallback"},
    {"case_id": 16, "category": "low_confidence_product_inquiry", "ticket_text": "i want apple ipad", "expected_language": "unknown", "expected_intent": "sales_or_product_inquiry", "expected_rag_used": False, "expected_policy_source": "No policy needed", "expected_final_priority": "low", "expected_final_sentiment": "neutral", "expected_human_review": True},
]

INVALID_CASES = [
    {"case_id": 17, "category": "blank_whitespace", "ticket_text": "     ", "expected_status_code": 422},
    {"case_id": 18, "category": "too_short_after_trim", "ticket_text": "  hi  ", "expected_status_code": 422},
]

REQUIRED_RESPONSE_FIELDS = {
    "ticket", "language", "detected_intent", "rag_used", "requires_human_review",
    "model_predictions", "final_priority", "final_sentiment", "policy_source",
    "policy_context", "generation_mode", "generated_reply",
}


def _equals_if_expected(case: Dict[str, Any], result: Dict[str, Any], case_key: str, result_key: str):
    if case_key not in case:
        return None
    return result.get(result_key) == case[case_key]


def check_success_case(case: Dict[str, Any], result: Dict[str, Any], status_code: int) -> Dict[str, bool]:
    model_predictions = result.get("model_predictions") or {}
    checks: Dict[str, bool] = {
        "status_code_pass": status_code == 200,
        "required_fields_pass": REQUIRED_RESPONSE_FIELDS.issubset(result),
        "reply_nonempty_pass": bool(str(result.get("generated_reply", "")).strip()),
        "model_predictions_pass": {"ticket_type", "queue", "priority", "sentiment", "sentiment_method"}.issubset(model_predictions),
        "human_review_type_pass": isinstance(result.get("requires_human_review"), bool),
    }

    optional_checks = {
        "language_pass": _equals_if_expected(case, result, "expected_language", "language"),
        "intent_pass": _equals_if_expected(case, result, "expected_intent", "detected_intent"),
        "rag_used_pass": _equals_if_expected(case, result, "expected_rag_used", "rag_used"),
        "policy_source_pass": _equals_if_expected(case, result, "expected_policy_source", "policy_source"),
        "final_priority_pass": _equals_if_expected(case, result, "expected_final_priority", "final_priority"),
        "final_sentiment_pass": _equals_if_expected(case, result, "expected_final_sentiment", "final_sentiment"),
        "final_queue_pass": _equals_if_expected(case, result, "expected_final_queue", "final_queue_recommendation"),
        "human_review_pass": _equals_if_expected(case, result, "expected_human_review", "requires_human_review"),
    }
    checks.update({name: value for name, value in optional_checks.items() if value is not None})

    if "expected_sentiment_method" in case:
        checks["sentiment_method_pass"] = model_predictions.get("sentiment_method") == case["expected_sentiment_method"]

    return checks


def request_case(case: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(API_URL, json={"ticket_text": case["ticket_text"], "use_llm": False}, timeout=120)
    try:
        result = response.json()
    except ValueError:
        result = {"error": response.text}

    if "expected_status_code" in case:
        checks = {"status_code_pass": response.status_code == case["expected_status_code"]}
    else:
        checks = check_success_case(case, result, response.status_code)

    return {"case": case, "status_code": response.status_code, "checks": checks, "response": result}


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    outputs = []

    try:
        for case in [*SUCCESS_CASES, *INVALID_CASES]:
            item = request_case(case)
            outputs.append(item)
            failed = [name for name, passed in item["checks"].items() if not passed]
            print("=" * 100)
            print("Case:", case["case_id"], case["category"])
            print("Status:", item["status_code"])
            print("Result:", "PASS" if not failed else "FAIL")
            if failed:
                print("Failed checks:", ", ".join(failed))
    except requests.RequestException as exc:
        print(f"Could not complete API tests against {API_URL}: {exc}", file=sys.stderr)
        sys.exit(2)

    OUTPUT_PATH.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    total = len(outputs)
    passed = sum(all(item["checks"].values()) for item in outputs)
    print("\nSUMMARY")
    print("Passed:", passed, "/", total)
    print("Saved output to:", OUTPUT_PATH)
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
