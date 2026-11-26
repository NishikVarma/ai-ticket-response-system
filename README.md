# AI Customer Ticket Response System

Web app to manage support tickets. On creation, a local ML model predicts a one-line solution; a local LLM
with RAG over a local knowledge base expands it into a customer-friendly reply the agent can review and edit.
Everything runs locally: no cloud AI APIs.

## Architecture

```
Browser ── FastAPI (app.py, Jinja2 forms)
              │
              ├─ db.py        SQLite: tickets, field_definitions, ticket_field_values
              └─ pipeline.py  per ticket, each step saves its result and fails soft
                    ├─ ml.py   TF-IDF + LogisticRegression  → one-line solution + confidence
                    ├─ rag.py  MiniLM embeddings + cosine search over data/kb/*.md → best document, top-3 passages
                    └─ rag.py  Ollama (local LLM) → detailed reply
```

| File | Role |
|---|---|
| `app.py` | Routes: ticket CRUD, regenerate, custom-field admin, validation |
| `db.py` | Schema (created automatically on first run) and queries |
| `ml.py` | Training (`uv run python ml.py`) and prediction |
| `rag.py` | KB chunking, local embeddings, retrieval, Ollama call, prompt |
| `pipeline.py` | Orchestrates predict → retrieve → generate, saves results, handles failures |
| `data/tickets.csv` | Sample training data (80 tickets, 10 solution classes) |
| `data/kb/*.md` | Sample knowledge base (10 short documents) |

## Install and run

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
uv sync                               # installs Python 3.12 + dependencies
uv run python ml.py                   # trains the model -> model.joblib
ollama pull qwen2.5:3b-instruct       # local LLM (any chat model works; see config)
uv run uvicorn app:app                # http://127.0.0.1:8000
uv run pytest                         # tests (Ollama not needed; LLM is mocked)
```

The first run downloads the embedding model `all-MiniLM-L6-v2` (~90 MB) once; afterwards it works offline.

### Database
SQLite file `tickets.db`, created automatically. Schema is in `db.py` (`SCHEMA`). No migrations needed.

### Configuration (environment variables, all optional)
| Variable | Default | Meaning |
|---|---|---|
| `TICKET_DB` | `tickets.db` | SQLite path |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct` | LLM name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `ML_MIN_CONFIDENCE` | `0.25` | Below this, no solution is saved and the ticket is flagged |
| `RAG_MIN_SIM` | `0.30` | Minimum cosine similarity for a KB passage to be used |

## ML prediction
- **Data:** `data/tickets.csv` (subject, description, solution): 80 synthetic tickets, 8 per solution.
- **Model:** TF-IDF (1–2 grams) + Logistic Regression. Each distinct one-line solution is a class, so predictions are
  always well-formed, vetted sentences (a text-generation model could produce unvetted ones). With little data it
  trains in under a second, runs on CPU, and gives a probability used as a confidence score.
- **Process:** subject + description → TF-IDF → class probabilities → top class and its probability. If probability
  is below `ML_MIN_CONFIDENCE`, the ticket is flagged for manual handling instead of guessing.
- **Held-out check:** `uv run python ml.py` prints accuracy on a 20% stratified split (~0.94 on the sample data;
  the sample is small and synthetic, so treat as a sanity check, not a benchmark).
- **Adding a solution class:** add rows to the CSV, add a KB document covering it, re-run `ml.py`.

## LLM / RAG workflow
1. KB documents are split into paragraphs (each prefixed with its title) and embedded locally.
2. Query = ticket subject + description + predicted solution; embedded with the same model.
3. The best-matching document is chosen by cosine similarity; its top 3 paragraphs are used, dropping any below
   `RAG_MIN_SIM`. This keeps loosely related topics out of the reply.
4. The ticket, predicted solution and passages go to the local LLM with a strict system prompt: use only the
   excerpts; no invented steps, policies, time frames or contacts. Temperature 0.2. Signatures and
   `[placeholder]` lines are stripped from the output.
5. If nothing relevant is retrieved, the LLM is skipped and a generic acknowledgement is saved, so the model is never asked
   to answer without grounding.
6. The draft is saved as `ai_response`; the agent edits it in the "Final response" box, which is saved as `final_response`.

## Dynamic fields
Define fields at **Custom fields** (text, dropdown, number, date, checkbox; optional "required"). Definitions live in
`field_definitions`, values in `ticket_field_values` (ticket, field, value), so adding a field never changes the schema.
The ticket form is rendered from the definitions and validated server-side (required, number, ISO date, dropdown
option). "Removing" a field hides it; stored values are kept.

## Failure handling
| Situation | Behaviour |
|---|---|
| Missing required/invalid fields | Form re-shown with errors, nothing saved |
| Model not trained / prediction error | Ticket saved; warning shown with the command to train |
| Low ML confidence | Ticket saved, no solution, flagged for manual handling |
| KB empty or nothing relevant | Generic reply, warning shown |
| Ollama down, model missing, timeout | Ticket and prediction saved; warning shown; **Regenerate** button |
| Unknown ticket id / deleted ticket | Redirect/message, no crash |

## Demo script
1. `New ticket`: "Unable to log in" / "I forgot my password and cannot access my account." → predicted solution
   "Reset the password using the account recovery process." and a reply based on `password_reset.md`.
2. Try a billing, shipping and app-crash ticket to see different predictions.
3. `Custom fields`: add a required dropdown; it shows on the next ticket form with no code changes.
4. Edit the final response and save; stop Ollama and create a ticket to see the graceful failure, then Regenerate.

## Limitations
Sample data is synthetic and small; one solution per ticket; replies draw on one KB document; a small LLM can still paraphrase loosely, so review
every draft; synchronous generation (the create request waits
for the LLM); no authentication.
