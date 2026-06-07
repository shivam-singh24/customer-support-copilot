import html
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"


def safe_text(value):
    """Convert value to safe display text."""
    if value is None:
        return "N/A"
    return html.escape(str(value))


def render_card(label, value):
    """Render compact dashboard card."""
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{safe_text(label)}</div>
            <div class="summary-value">{safe_text(value)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.set_page_config(
    page_title="AI Customer Support Copilot",
    page_icon="🎧",
    layout="wide"
)

st.markdown(
    """
    <style>
        .summary-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            min-height: 82px;
            margin-bottom: 0.8rem;
            background-color: rgba(250, 250, 250, 0.04);
        }

        .summary-label {
            font-size: 0.78rem;
            opacity: 0.75;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }

        .summary-value {
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        div[data-testid="stTextArea"] textarea {
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Customer Support Copilot")
st.write(
    "Analyze customer support tickets, retrieve policy context, "
    "and generate a human-reviewable reply."
)

st.divider()

ticket_text = st.text_area(
    "Enter customer ticket",
    height=180,
    placeholder="Example: My order arrived damaged and I want a refund."
)

submit = st.button("Analyze Ticket")

if submit:
    if not ticket_text.strip():
        st.warning("Please enter a ticket description.")
    else:
        with st.spinner("Analyzing ticket..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyze-ticket",
                    json={"ticket_text": ticket_text},
                    timeout=30
                )

                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code}")
                    st.text(response.text)

                else:
                    data = response.json()

                    model_predictions = data.get("model_predictions", {})

                    ticket_type = model_predictions.get("ticket_type", "N/A")
                    model_queue = model_predictions.get("queue", "N/A")
                    model_priority = model_predictions.get("priority", "N/A")
                    model_sentiment = model_predictions.get("sentiment", "N/A")

                    final_priority = data.get("final_priority", model_priority)
                    final_sentiment = data.get("final_sentiment", model_sentiment)
                    final_queue = data.get("final_queue_recommendation", model_queue)

                    detected_intent = data.get("detected_intent", "N/A")
                    intent_confidence = data.get("intent_confidence", None)
                    intent_source = data.get("intent_router_source", "N/A")

                    language = data.get("language", "N/A")
                    rag_used = data.get("rag_used", False)
                    requires_human_review = data.get("requires_human_review", False)

                    policy_source = data.get("policy_source", "N/A")
                    policy_context = data.get(
                        "policy_context",
                        "No policy context returned."
                    )
                    policy_suggested_queue = data.get(
                        "policy_suggested_queue",
                        "N/A"
                    )
                    policy_suggested_action = data.get(
                        "policy_suggested_action",
                        "N/A"
                    )

                    generation_mode = data.get("generation_mode", "N/A")
                    generated_reply = data.get(
                        "generated_reply",
                        "No reply generated."
                    )

                    confidence_display = (
                        f"{intent_confidence:.2f}"
                        if intent_confidence is not None
                        else "N/A"
                    )

                    st.success("Ticket analyzed successfully.")

                    st.subheader("Ticket Analysis Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        render_card("Ticket Type", ticket_type)

                    with col2:
                        render_card("Model Queue", model_queue)

                    with col3:
                        render_card("Priority", final_priority)

                    with col4:
                        render_card("Sentiment", final_sentiment)

                    col5, col6, col7, col8 = st.columns(4)

                    with col5:
                        render_card("Detected Issue", detected_intent)

                    with col6:
                        render_card("Intent Confidence", confidence_display)

                    with col7:
                        render_card("Language", language)

                    with col8:
                        render_card("Generation Mode", generation_mode)

                    st.divider()

                    st.subheader("Routing Decision")

                    st.write(f"**Intent router source:** {intent_source}")
                    st.write(f"**Final queue recommendation:** {final_queue}")

                    status_col1, status_col2 = st.columns(2)

                    with status_col1:
                        if rag_used:
                            st.success("RAG retrieval used")
                        else:
                            st.warning("RAG retrieval not used")

                    with status_col2:
                        if requires_human_review:
                            st.error("Human review required")
                        else:
                            st.success("Suitable for automated draft reply")

                    st.divider()

                    st.subheader("Policy Guidance")

                    col9, col10 = st.columns(2)

                    with col9:
                        st.write("**Policy source:**")
                        st.write(policy_source)

                    with col10:
                        st.write("**Policy suggested queue:**")
                        st.write(policy_suggested_queue)

                    st.write("**Policy suggested action:**")
                    st.info(policy_suggested_action)

                    with st.expander("Retrieved Policy Context"):
                        st.write(policy_context)

                    st.divider()

                    st.subheader("Suggested Reply")

                    st.text_area(
                        "Generated response",
                        value=generated_reply,
                        height=240
                    )

                    st.divider()

                    with st.expander("Raw API Response"):
                        st.json(data)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure backend is running on http://127.0.0.1:8000"
                )

            except requests.exceptions.Timeout:
                st.error("The API request timed out.")

            except Exception as e:
                st.error(f"Unexpected error: {e}")