"""Ticket -> ML prediction -> RAG retrieval -> LLM draft. Every step saves its result and fails soft."""
import db
import ml
import rag

FALLBACK = ("Thank you for contacting us. We have received your request and a member of our support "
            "team will review it and get back to you shortly.")


def run(tid):
    ticket, _ = db.get_ticket(tid)
    # 1. ML prediction
    try:
        solution, conf = ml.predict(ticket["subject"], ticket["description"])
        if conf < ml.MIN_CONFIDENCE:
            db.save_ai(tid, predicted_solution="", prediction_confidence=conf,
                       ai_error="ML model not confident enough; needs manual handling.", ai_response="")
            return
        db.save_ai(tid, predicted_solution=solution, prediction_confidence=conf, ai_error="")
    except Exception as e:  # missing/corrupt model, etc.
        db.save_ai(tid, ai_error=f"ML prediction failed: {e}. Run `uv run python ml.py` to train the model.")
        return
    # 2. RAG retrieval
    try:
        passages = rag.retrieve(f"{ticket['subject']}. {ticket['description']} {solution}")
    except Exception as e:
        db.save_ai(tid, ai_error=f"Knowledge base search failed: {e}")
        return
    if not passages:
        db.save_ai(tid, ai_response=FALLBACK,
                   ai_error="No relevant knowledge-base content found; generic reply used.")
        return
    # 3. LLM generation
    try:
        db.save_ai(tid, ai_response=rag.generate(ticket, solution, passages), ai_error="")
    except rag.LLMError as e:
        db.save_ai(tid, ai_error=str(e))
