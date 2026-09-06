# Archivist

An internal knowledge agent that answers from your own documents — built as a **stateful LangGraph agent** rather than a single LLM call. It classifies what you are asking for, decides whether to search your documents, asks which document you meant when the request is ambiguous, plans a sequence of tool calls, executes them, and returns the full reasoning trace alongside the answer.

[Live demo](https://internal-ai-agents-ui.onrender.com) · FastAPI · LangGraph · FAISS · JWT auth · Streamlit console

**Measured, not asserted:** an evaluation harness scores the intent router against the keyword baseline it falls back to — currently **96.3% vs 37.0%** over 54 labelled cases — and retrieval at **100% hit@4 / 0.93 MRR**. See [evals/](evals/).

```bash
git clone https://github.com/vaibhavagarwal19/internalaiagents.git
cd internalaiagents
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
alembic upgrade head
uvicorn app.main:app --reload
```

Then `streamlit run frontend/streamlit_app.py` and open <http://127.0.0.1:8501>.

---

## 1. What makes this more than an API wrapper

Most "chat with your docs" projects are one prompt and one retrieval call. This one routes each request through a compiled state graph where every decision is explicit, recorded, and independently testable.

```
                   ┌──────────────┐
                   │ detect_intent│  LLM classification + keyword fallback
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │plan_retrieval│  should we search at all?
                   └──────┬───────┘
                  search? │ no ──────────────────┐
                      yes ▼                      │
                   ┌──────────────┐              │
                   │   retrieve   │  FAISS, scoped to the caller
                   └──────┬───────┘              │
                          ▼                      │
                   ┌──────────────┐              │
                   │clarify_request│ ambiguous?  │
                   └──┬────────┬──┘              │
              ask user│        │clear            │
                      │        ▼                 │
                      │ ┌──────────────────┐     │
                      │ │understand_document│ compress context
                      │ └────────┬─────────┘     │
                      │          ▼               │
                      │   ┌─────────────┐ ◄──────┘
                      │   │ plan_actions│  which tools, in what order
                      │   └──┬───────┬──┘
                      │ tools│       │none
                      │      ▼       │
                      │ ┌──────────────┐
                      │ │execute_actions│
                      │ └──────────────┘
                      ▼          ▼
                     ┌────────────┐
                     │  respond   │
                     └────────────┘
```

**Design decisions worth calling out:**

- **Every LLM decision has a deterministic fallback.** Intent classification, the retrieval decision, and tool planning each ask the model for structured JSON — and each falls back to keyword heuristics when the model is unavailable or returns unparseable output. The agent degrades instead of failing. These fallbacks are pure functions and are covered by tests that never touch the network.
- **It asks instead of guessing.** When a request says "summarize the uploaded document" and several documents match, `clarify_request` returns the candidate list and asks which one, rather than silently picking the highest-scoring chunk.
- **Retrieval is scoped per user.** Chunks carry `owner_id` metadata and results are filtered to the caller, so one account's documents can never surface in another's answers.
- **Retrieved documents are treated as untrusted input.** Document text reaches a planner that can send email and file tickets, so it is delimited and labelled as data, scanned for injection phrasing, and — if flagged — the side-effecting tools are stripped from the plan. Email recipients are validated against addresses the *user* supplied, so an address that appears only inside a document is never contacted. Verified against a live poisoned document.
- **Conversations have memory.** Turns are persisted per user and thread, so "and who owns that?" resolves against what was actually said.
- **Every request is costed.** Token counts, an estimated price, and per-step latency are captured across all model calls in a request and surfaced in the UI.
- **The reasoning is visible.** Responses include the detected intent, confidence, the planner's stated reasoning, the planned tool steps, the retrieved context, and the executed calls. The Streamlit UI renders all of it, which makes the agent debuggable rather than a black box.

## 2. Architecture

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async, typed, generates OpenAPI docs for free |
| Agent | LangGraph | Explicit state machine beats an opaque agent loop for debuggability |
| Chat model | Groq (`openai/gpt-oss-120b`) | Fast inference, generous free tier. Groq retired the Llama 3.1 chat models; check `/v1/models` for what your key can use |
| Embeddings | fastembed (`bge-small-en-v1.5`) | Runs in-process: no API key, no network, no separate service |
| Vector store | FAISS | Rebuildable from Postgres, so the index is never the source of truth |
| Persistence | SQLAlchemy + Alembic | Versioned schema; SQLite locally, Postgres deployed |
| UI | Streamlit | Operator console for exercising every endpoint |

**Chat and embedding providers are configured independently.** This matters: Groq serves no embeddings endpoint, so a single shared `LLM_BASE_URL` for both could never work.

## 3. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check (public) |
| `POST` | `/api/v1/auth/register` | Create an account |
| `POST` | `/api/v1/auth/login` | Exchange credentials for a JWT |
| `GET` | `/api/v1/auth/me` | Current user |
| `POST` | `/api/v1/chat` | Ask the agent (pass `thread_id` to continue a conversation) |
| `DELETE` | `/api/v1/chat/{thread_id}` | Forget a conversation |
| `POST` | `/api/v1/generate-report` | Generate and store a report |
| `POST` | `/api/v1/summarize` | Summarize meeting notes |
| `POST` | `/api/v1/upload-doc` | Upload a text document for retrieval |
| `GET` | `/api/v1/upload-doc` | List your documents |
| `DELETE` | `/api/v1/upload-doc/{id}` | Delete a document, its file, and its vectors |
| `POST` | `/api/v1/workflow` | Run a multi-step tool workflow |

Everything except `/health` and the two auth routes requires `Authorization: Bearer <token>`. Interactive docs at `/docs`.

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"StrongPass123"}' | jq -r .access_token)

curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"What does the ops playbook say about payroll incidents?","use_rag":true}'
```

## 4. Configuration

Copy `.env.example` to `.env`. The only value you must supply is `GROQ_API_KEY`.

Embeddings default to `fastembed`, which needs no key and no running service. Switch to `openai` or `ollama` via `EMBEDDING_PROVIDER` if you prefer. **Changing the embedding model changes the vector dimensions**, so rebuild the index afterwards:

```bash
python -m scripts.reindex          # all users
python -m scripts.reindex --user 3 # one user
```

Because document text lives in the database, the FAISS index is always reproducible and never needs backing up — which is also how deployment on an ephemeral filesystem works.

Redis and Celery are optional. `ENABLE_BACKGROUND_TASKS` defaults to `false`; uploads work without a broker.

## 5. Development

```bash
pip install -r requirements-dev.txt
pytest                                    # 124 tests, ~12s, no network
python -m evals.run_eval                  # routing + retrieval scores
ruff check app frontend scripts tests
alembic upgrade head                      # apply migrations
alembic revision --autogenerate -m "..."  # after changing a model
```

The suite runs against throwaway SQLite and stubs the model and vector store, so it needs no API key and no network. The Streamlit console is covered too, via Streamlit's headless `AppTest` harness with the HTTP layer stubbed. CI runs lint, tests, and a from-scratch migration on every push.

## 6. Running with Docker

```bash
cp .env.example .env   # add GROQ_API_KEY
docker compose up --build
```

Brings up Postgres, the API on `:8000`, and the UI on `:8501`. The embedding model is baked into the image so the first request after a deploy doesn't pay a download.

## 7. Known limitations

Stated plainly, because they are deliberate scope choices rather than oversights:

- **The workflow tools are simulated.** `create_ticket` and `send_email` record a row in `workflow_actions` and report success; no ticketing system or SMTP is wired up. The planning and execution machinery around them is real.
- **Only text-like uploads are handled properly.** Files are decoded as UTF-8 with errors ignored, so a PDF or `.docx` will be accepted and indexed as garbage rather than rejected.
- **FAISS is single-node.** Fine for one container; a horizontally scaled deployment wants pgvector or a managed vector database.
- **No refresh tokens or RBAC.** Access tokens expire in an hour and every account has identical permissions.
- **Injection detection is pattern-based.** It catches the common phrasings and is backed by
  recipient validation that does not depend on detection working, but a novel phrasing could
  still slip past the scanner. The enforcement layer is the one to rely on.

## 8. Project layout

```text
app/
├── api/routes/     FastAPI routers, one per capability
├── api/deps.py     Bearer-token authentication
├── core/           Settings, logging, password + JWT handling
├── db/             Engine, session, startup checks
├── models/         SQLAlchemy ORM models
├── schemas/        Pydantic request/response contracts
├── services/       Agent, retrieval, guardrails, telemetry, conversations
└── tasks/          Optional Celery jobs
evals/              Labelled datasets + scoring harness
alembic/            Migrations
frontend/           Streamlit UI
samples/            Example documents to upload
scripts/reindex.py  Rebuild the vector index from the database
tests/              Auth, routing, retrieval, and document tests
```
