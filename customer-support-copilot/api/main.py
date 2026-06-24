from typing import Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.generate_reply import analyze_ticket_and_generate_reply


app = FastAPI(
    title="AI Customer Support Copilot API",
    description=(
        "API for ticket classification, intent routing, multilingual RAG retrieval, "
        "and policy-led or intent-led reply generation."
    ),
    version="1.2.0",
)


class TicketRequest(BaseModel):
    ticket_text: str = Field(
        ...,
        min_length=5,
        description="Customer support ticket text in English or German.",
        examples=["My laptop arrived damaged and I want a refund."],
    )
    use_llm: bool = Field(
        default=False,
        description="Set true to use optional LLM reply generation for support-policy issues. Defaults to template mode.",
    )

    @field_validator("ticket_text", mode="before")
    @classmethod
    def strip_ticket_text(cls, value):
        """Trim surrounding whitespace before Pydantic applies min_length."""
        return value.strip() if isinstance(value, str) else value


class TicketResponse(BaseModel):
    ticket: str
    language: str
    detected_intent: str
    intent_confidence: Optional[float] = None
    intent_router_source: Optional[str] = None
    rag_used: bool
    requires_human_review: bool

    model_predictions: Dict[str, Any]

    final_priority: str
    final_sentiment: str
    final_queue_recommendation: Optional[str] = None

    policy_source: str
    policy_context: str
    policy_suggested_queue: Optional[str] = None
    policy_suggested_action: Optional[str] = None

    generation_mode: str
    generated_reply: str
    llm_error: Optional[str] = None


@app.get("/")
def root():
    return {
        "message": "AI Customer Support Copilot API is running.",
        "docs": "/docs",
        "main_endpoint": "/analyze-ticket",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "customer-support-copilot",
    }


@app.post("/analyze-ticket", response_model=TicketResponse)
def analyze_ticket(request: TicketRequest):
    result = analyze_ticket_and_generate_reply(
        ticket_text=request.ticket_text,
        use_llm=request.use_llm,
    )

    return result
