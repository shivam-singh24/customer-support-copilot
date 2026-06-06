import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("reports/fastapi_test_outputs.json")
OUTPUT_PATH = Path("reports/fastapi_test_summary.csv")


data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

rows = []

for item in data:
    case = item["case"]
    response = item["response"]
    checks = item["checks"]

    rows.append({
        "case_id": case["case_id"],
        "category": case["category"],
        "ticket_text": case["ticket_text"],
        "expected_intent": case["expected_intent"],
        "actual_intent": response.get("detected_intent"),
        "expected_rag_used": case["expected_rag_used"],
        "actual_rag_used": response.get("rag_used"),
        "expected_policy_source": case["expected_policy_source"],
        "actual_policy_source": response.get("policy_source"),
        "intent_pass": checks.get("intent_pass"),
        "rag_used_pass": checks.get("rag_used_pass"),
        "policy_source_pass": checks.get("policy_source_pass"),
        "overall_pass": all(checks.values()) if checks else False,
        "language": response.get("language"),
        "final_priority": response.get("final_priority"),
        "final_sentiment": response.get("final_sentiment"),
        "final_queue_recommendation": response.get("final_queue_recommendation"),
        "requires_human_review": response.get("requires_human_review"),
        "generation_mode": response.get("generation_mode"),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print(df[[
    "case_id",
    "category",
    "actual_intent",
    "actual_policy_source",
    "overall_pass"
]])