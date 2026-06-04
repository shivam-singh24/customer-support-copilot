import os
import joblib

from rag_pipeline import retrieve_policy_context


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_type_baseline.pkl")
QUEUE_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_queue_baseline.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_priority_baseline.pkl")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")


def load_models():
    """
    Loads all trained ML models.
    """

    models = {
        "type_model": joblib.load(TYPE_MODEL_PATH),
        "queue_model": joblib.load(QUEUE_MODEL_PATH),
        "priority_model": joblib.load(PRIORITY_MODEL_PATH),
        "sentiment_model": joblib.load(SENTIMENT_MODEL_PATH)
    }

    return models


def generate_support_reply(
    ticket_text,
    predicted_type,
    predicted_queue,
    predicted_priority,
    predicted_sentiment,
    retrieved_policy
):
    """
    Generates a human-reviewable support reply using
    ticket predictions and retrieved policy context.
    """

    policy_text = retrieved_policy[0]["content"] if retrieved_policy else "No relevant policy context found."
    policy_source = retrieved_policy[0]["source"] if retrieved_policy else "No policy source"

    sentiment_value = str(predicted_sentiment).lower()
    priority_value = str(predicted_priority).lower()

    if sentiment_value in ["negative", "neg", "0"]:
        opening = "We’re sorry to hear about the issue you’re facing."
    elif sentiment_value in ["positive", "pos", "1"]:
        opening = "Thank you for reaching out to us."
    else:
        opening = "Thank you for contacting customer support."

    if priority_value in ["high", "urgent", "critical"]:
        priority_line = "We understand this may be urgent, so your request should be reviewed with priority."
    elif priority_value == "medium":
        priority_line = "Your request has been noted and should be handled by the appropriate support team."
    else:
        priority_line = "Your request has been recorded and will be reviewed by our support team."

    reply = f"""
Dear Customer,

{opening}

Based on your message, this request appears to be related to {predicted_type} and should be handled by the {predicted_queue} team.

{priority_line}

Relevant policy reference:
{policy_text}

Recommended next step:
Please keep your order details, product information, screenshots, or proof of purchase available if required by the support team.

Best regards,
Customer Support Team
""".strip()

    return {
        "reply": reply,
        "policy_source": policy_source
    }


def analyze_ticket_and_generate_reply(ticket_text):
    """
    Runs ticket classification, queue prediction, priority prediction,
    sentiment analysis, RAG retrieval, and reply generation.
    """

    models = load_models()

    predicted_type = models["type_model"].predict([ticket_text])[0]
    predicted_queue = models["queue_model"].predict([ticket_text])[0]
    predicted_priority = models["priority_model"].predict([ticket_text])[0]
    predicted_sentiment = models["sentiment_model"].predict([ticket_text])[0]

    retrieved_policy = retrieve_policy_context(ticket_text, top_k=2)

    reply_result = generate_support_reply(
        ticket_text=ticket_text,
        predicted_type=predicted_type,
        predicted_queue=predicted_queue,
        predicted_priority=predicted_priority,
        predicted_sentiment=predicted_sentiment,
        retrieved_policy=retrieved_policy
    )

    return {
        "ticket": ticket_text,
        "predicted_type": predicted_type,
        "predicted_queue": predicted_queue,
        "predicted_priority": predicted_priority,
        "predicted_sentiment": predicted_sentiment,
        "policy_source": reply_result["policy_source"],
        "generated_reply": reply_result["reply"]
    }


if __name__ == "__main__":
    sample_ticket = "My laptop arrived damaged and I want a refund."

    result = analyze_ticket_and_generate_reply(sample_ticket)

    print("Ticket:", result["ticket"])
    print("Type:", result["predicted_type"])
    print("Queue:", result["predicted_queue"])
    print("Priority:", result["predicted_priority"])
    print("Sentiment:", result["predicted_sentiment"])
    print("Policy Source:", result["policy_source"])
    print("\nGenerated Reply:\n")
    print(result["generated_reply"])