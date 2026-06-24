"""Convert the detailed FastAPI JSON test report into a compact CSV summary."""

import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("reports/fastapi_test_outputs.json")
OUTPUT_PATH = Path("reports/fastapi_test_summary.csv")


data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
rows = []

for item in data:
    case = item["case"]
    response = item.get("response") or {}
    checks = item.get("checks") or {}
    failed_checks = [name for name, passed in checks.items() if not passed]
    model_predictions = response.get("model_predictions") or {}

    rows.append({
        "case_id": case["case_id"],
        "category": case["category"],
        "ticket_text": case["ticket_text"],
        "status_code": item.get("status_code"),
        "expected_status_code": case.get("expected_status_code", 200),
        "expected_language": case.get("expected_language"),
        "actual_language": response.get("language"),
        "expected_intent": case.get("expected_intent"),
        "actual_intent": response.get("detected_intent"),
        "expected_rag_used": case.get("expected_rag_used"),
        "actual_rag_used": response.get("rag_used"),
        "expected_policy_source": case.get("expected_policy_source"),
        "actual_policy_source": response.get("policy_source"),
        "expected_final_priority": case.get("expected_final_priority"),
        "actual_final_priority": response.get("final_priority"),
        "expected_final_sentiment": case.get("expected_final_sentiment"),
        "actual_final_sentiment": response.get("final_sentiment"),
        "expected_sentiment_method": case.get("expected_sentiment_method"),
        "actual_sentiment_method": model_predictions.get("sentiment_method"),
        "expected_final_queue": case.get("expected_final_queue"),
        "actual_final_queue": response.get("final_queue_recommendation"),
        "requires_human_review": response.get("requires_human_review"),
        "generation_mode": response.get("generation_mode"),
        "overall_pass": bool(checks) and all(checks.values()),
        "failed_checks": ", ".join(failed_checks),
    })


df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)
print(df[["case_id", "category", "status_code", "overall_pass", "failed_checks"]].to_string(index=False))
