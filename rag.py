"""RAG step: local embeddings + cosine search over data/kb, then a local Ollama LLM."""
import glob
import os
import re
import httpx
import numpy as np

KB_DIR = "data/kb"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")
TOP_K = 3
MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIM", "0.30"))

_embedder = None
_chunks = None   # list of (source, text)
_vecs = None


class LLMError(Exception):
    pass


def _embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return np.asarray(_embedder.encode(texts, normalize_embeddings=True))


def _load_kb():
    """Chunk = one paragraph, prefixed with the doc title so it keeps its context."""
    global _chunks, _vecs
    chunks = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
        first, _, rest = open(path, encoding="utf-8").read().partition("\n")
        title = first.lstrip("# ").strip()
        for p in rest.split("\n\n"):
            if p.strip():
                chunks.append((os.path.basename(path), f"{title}: {p.strip()}"))
    _chunks = chunks
    _vecs = _embed([c[1] for c in chunks]) if chunks else np.zeros((0, 1))


def retrieve(query):
    """Top-k chunks from the best-matching KB document above the threshold, as [(source, text, score)].

    Restricting to one document keeps loosely related topics out of the reply.
    """
    if _chunks is None:
        _load_kb()
    if not _chunks:
        return []
    scores = _vecs @ _embed([query])[0]
    ranked = np.argsort(-scores)
    top_doc = _chunks[ranked[0]][0]
    best = [i for i in ranked if _chunks[i][0] == top_doc and scores[i] >= MIN_SIMILARITY][:TOP_K]
    return [(_chunks[i][0], _chunks[i][1], float(scores[i])) for i in best]


SYSTEM = (
    "You are a customer support assistant. Write a clear, polite, customer-friendly reply to the ticket. "
    "Use ONLY the information in the KNOWLEDGE BASE excerpts. Do not invent steps, policies, time frames, "
    "links or contact details that are not in the excerpts. If the excerpts do not cover something, "
    "do not mention it. Do not ask the customer to send any information unless an excerpt says to. "
    "Start with a brief apology or acknowledgement and end by offering further help. "
    "Do not add a signature, a sign-off name, or placeholders in square brackets."
)


def clean(text):
    """Small models sometimes add a signature despite the prompt: drop it and any [placeholder] lines."""
    text = re.split(r"\n\s*(?:Best regards|Kind regards|Warm regards|Regards|Sincerely)\b", text, flags=re.I)[0]
    return "\n".join(l for l in text.split("\n") if not re.search(r"\[[^\]]*\]", l)).strip()


def generate(ticket, solution, passages):
    context = "\n".join(f"- {p[1]}" for p in passages)
    user = (f"TICKET SUBJECT: {ticket['subject']}\nTICKET DESCRIPTION: {ticket['description']}\n"
            f"SUGGESTED SOLUTION: {solution}\n\nKNOWLEDGE BASE:\n{context}\n\nWrite the reply to the customer.")
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=180, json={
            "model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
        r.raise_for_status()
        text = clean(r.json()["message"]["content"])
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise LLMError(f"Local LLM unavailable ({type(e).__name__}). Is Ollama running with model "
                       f"'{OLLAMA_MODEL}'?") from e
    if not text:
        raise LLMError("Local LLM returned an empty response.")
    return text
