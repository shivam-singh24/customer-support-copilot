import json
from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000/analyze-ticket"
REPORTS_DIR = Path("reports")
OUTPUT_PATH = REPORTS_DIR / "fastapi_test_outputs.json"

TEST_CASES = [
    {
        "case_id": 1,
        "category": "english_refund",
        "ticket_text": "My laptop arrived damaged and I want a refund.",
        "expected_intent": "refund_issue",
        "expected_rag_used": True,
        "expected_policy_source": "refund_policy.txt",
    },
    {
        "case_id": 2,
        "category": "english_shipping",
        "ticket_text": "My package has not arrived and tracking is not updating.",
        "expected_intent": "shipping_issue",
        "expected_rag_used": True,
        "expected_policy_source": "shipping_policy.txt",
    },
    {
        "case_id": 3,
        "category": "english_account",
        "ticket_text": "I think my account has been hacked.",
        "expected_intent": "account_issue",
        "expected_rag_used": True,
        "expected_policy_source": "account_policy.txt",
    },
    {
        "case_id": 4,
        "category": "english_technical",
        "ticket_text": "The app keeps crashing whenever I open it.",
        "expected_intent": "technical_issue",
        "expected_rag_used": True,
        "expected_policy_source": "technical_support_policy.txt",
    },
    {
        "case_id": 5,
        "category": "english_warranty",
        "ticket_text": "My charger stopped working within the warranty period.",
        "expected_intent": "warranty_issue",
        "expected_rag_used": True,
        "expected_policy_source": "warranty_policy.txt",
    },
    {
        "case_id": 6,
        "category": "english_product_inquiry",
        "ticket_text": "I want to know about laptop Ryzen 5.",
        "expected_intent": "sales_or_product_inquiry",
        "expected_rag_used": False,
        "expected_policy_source": "No policy needed",
    },
    {
        "case_id": 7,
        "category": "english_positive_feedback",
        "ticket_text": "My laptop is working excellently, thank you.",
        "expected_intent": "positive_feedback",
        "expected_rag_used": False,
        "expected_policy_source": "No policy needed",
    },
    {
        "case_id": 8,
        "category": "english_ambiguous",
        "ticket_text": "Help me.",
        "expected_intent": "ambiguous",
        "expected_rag_used": False,
        "expected_policy_source": "No policy needed",
    },
    {
        "case_id": 9,
        "category": "german_refund",
        "ticket_text": "Ich habe den falschen Artikel erhalten und möchte mein Geld zurück.",
        "expected_intent": "refund_issue",
        "expected_rag_used": True,
        "expected_policy_source": "refund_policy.txt",
    },
    {
        "case_id": 10,
        "category": "german_shipping",
        "ticket_text": "Das Paket wird als zugestellt angezeigt, aber ich habe es nicht erhalten.",
        "expected_intent": "shipping_issue",
        "expected_rag_used": True,
        "expected_policy_source": "shipping_policy.txt",
    },
    {
        "case_id": 11,
        "category": "german_account",
        "ticket_text": "Ich glaube, mein Konto wurde gehackt.",
        "expected_intent": "account_issue",
        "expected_rag_used": True,
        "expected_policy_source": "account_policy.txt",
    },
    {
        "case_id": 12,
        "category": "german_technical",
        "ticket_text": "Die App stürzt jedes Mal ab, wenn ich sie öffne.",
        "expected_intent": "technical_issue",
        "expected_rag_used": True,
        "expected_policy_source": "technical_support_policy.txt",
    },
    {
        "case_id": 13,
        "category": "german_warranty",
        "ticket_text": "Mein Ladegerät funktioniert innerhalb der Garantiezeit nicht mehr.",
        "expected_intent": "warranty_issue",
        "expected_rag_used": True,
        "expected_policy_source": "warranty_policy.txt",
    },
    {
        "case_id": 14,
        "category": "german_product_inquiry",
        "ticket_text": "Ich möchte Informationen über einen Laptop mit Ryzen 5.",
        "expected_intent": "sales_or_product_inquiry",
        "expected_rag_used": False,
        "expected_policy_source": "No policy needed",
    },
    {
        "case_id": 15,
        "category": "german_positive_feedback",
        "ticket_text": "Mein Laptop funktioniert sehr gut, danke.",
        "expected_intent": "positive_feedback",
        "expected_rag_used": False,
        "expected_policy_source": "No policy needed",
    },
]


def check_case(case, result):
    return {
        "intent_pass": result.get("detected_intent") == case["expected_intent"],
        "rag_used_pass": result.get("rag_used") == case["expected_rag_used"],
        "policy_source_pass": result.get("policy_source") == case["expected_policy_source"],
    }


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    outputs = []

    for case in TEST_CASES:
        payload = {
            "ticket_text": case["ticket_text"],
            "use_llm": False,
        }

        response = requests.post(API_URL, json=payload, timeout=120)

        try:
            result = response.json()
        except Exception:
            result = {
                "error": response.text
            }

        checks = check_case(case, result) if response.status_code == 200 else {}

        outputs.append({
            "case": case,
            "status_code": response.status_code,
            "checks": checks,
            "response": result,
        })

        print("=" * 100)
        print("Case:", case["case_id"], case["category"])
        print("Status:", response.status_code)
        print("Expected intent:", case["expected_intent"])
        print("Actual intent:", result.get("detected_intent"))
        print("Expected policy:", case["expected_policy_source"])
        print("Actual policy:", result.get("policy_source"))
        print("Checks:", checks)

    OUTPUT_PATH.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    total = len(outputs)
    passed = sum(
        all(item["checks"].values())
        for item in outputs
        if item["status_code"] == 200
    )

    print("\\nSUMMARY")
    print("Passed:", passed, "/", total)
    print("Saved output to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()