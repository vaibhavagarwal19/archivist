# Internal AI Agents

Production-oriented FastAPI backend for internal AI agents with JWT auth, LangChain + LangGraph orchestration, Groq-hosted open models for chat, PostgreSQL persistence, FAISS-based RAG, Celery background jobs, and a Streamlit operator UI.

## 1. Project Structure

```text
InternalAiAgents/
├── app/
│   ├── api/
│   │   ├── deps.py
│   │   └── routes/
│   │       ├── auth.py
│   │       ├── chat.py
│   │       ├── documents.py
│   │       ├── health.py
│   │       ├── reports.py
│   │       ├── summarize.py
│   │       └── workflow.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── security.py
│   ├── db/
│   │   ├── base.py
│   │   ├── init_db.py
│   │   └── session.py
│   ├── models/
│   │   ├── document.py
│   │   ├── report.py
│   │   └── user.py
│   ├── schemas/
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── document.py
│   │   └── report.py
│   ├── services/
│   │   ├── document_service.py
│   │   ├── langchain_factory.py
│   │   ├── langgraph_agent.py
│   │   ├── llm_client.py
│   │   ├── report_service.py
│   │   ├── summary_service.py
│   │   ├── vector_store.py
│   │   └── workflow_service.py
│   ├── tasks/
│   │   ├── celery_app.py
│   │   └── document_tasks.py
│   ├── tools/
│   │   └── internal_tools.py
│   ├── utils/
│   │   └── text.py
│   └── main.py
├── .env.example
├── .gitignore
├── requirements.txt
└── worker.py
```

## 2. Core Capabilities

- FastAPI backend with modular route and service layers
- JWT-based auth for protected APIs
- Hosted internal agent using LangChain + LangGraph with Groq-hosted open models
- RAG pipeline using FAISS with Ollama embeddings
- Report generation API
- Meeting summarization API
- Internal chatbot API
- Workflow automation endpoint for multi-step tool execution
- Celery + Redis background task integration
- PostgreSQL persistence via SQLAlchemy ORM

## 3. Environment Setup

### Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create your environment file

```bash
copy .env.example .env
```

Then update `.env` with your PostgreSQL connection string, Redis URL, and Groq API key.

## 4. LLM Setup With Groq

Create a Groq API key and place it in either `LLM_API_KEY` or `GROQ_API_KEY` inside `.env`.

By default the project is configured to use:

```text
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

This setup moves chat generation to Groq for much faster inference while keeping embeddings on local Ollama.

## 5. Infrastructure Setup

### PostgreSQL

Create a database named `internal_ai_agents` and ensure the user in `DATABASE_URL` can connect.

### Redis

Run Redis locally on port `6379` or update `REDIS_URL`.

## 6. Run the Application

### Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

On Windows, if you have both a global Python and a project virtual environment, prefer running the server through the virtual environment interpreter:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

or use:
### Start the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

### Start the Celery worker

```bash
celery -A worker.celery_app worker --loglevel=info
```

The API will be available at:

- API root: `http://127.0.0.1:8000`
- Streamlit dashboard: `http://127.0.0.1:8501`
- Swagger docs: `http://127.0.0.1:8000/docs`

## 7. Frontend Dashboard

The project now includes a Streamlit dashboard that talks to the FastAPI backend over HTTP.

- Open `http://127.0.0.1:8501` to access the UI
- Authenticate with register/login from the Streamlit access panel
- Use the dashboard sections to access chat, report generation, meeting summaries, workflow execution, and document upload/listing
- The frontend stores the JWT in Streamlit session state and attaches it to protected API requests
- The chat panel shows the detected intent, confidence, planner reasoning, planned tool steps, retrieved context, and executed tool calls

## 8. API Flow

1. Register a user
2. Log in to get a JWT token
3. Use `Authorization: Bearer <token>` for protected endpoints
4. Upload documents for RAG
5. Query the internal chatbot or generate reports/summaries

## 9. Example API Requests

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Register

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@example.com\",\"full_name\":\"Admin User\",\"password\":\"StrongPass123\"}"
```

### Login

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@example.com\",\"password\":\"StrongPass123\"}"
```

### Chat with the internal AI agent

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"query\":\"Summarize sales blockers and email ops team\",\"use_rag\":true}"
```

### Generate report

```bash
curl -X POST http://127.0.0.1:8000/api/v1/generate-report \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"title\":\"Weekly Ops Report\",\"prompt\":\"Create a weekly operational report for leadership.\"}"
```

### Summarize meeting notes

```bash
curl -X POST http://127.0.0.1:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"meeting_notes\":\"We reviewed Q2 hiring goals, decided to pause contractor spend, and assigned next steps to finance.\"}"
```

### Upload document

```bash
curl -X POST http://127.0.0.1:8000/api/v1/upload-doc \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@employee-handbook.txt"
```

### Execute workflow automation

```bash
curl -X POST http://127.0.0.1:8000/api/v1/workflow \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"objective\":\"Investigate failed payroll sync and notify stakeholders\",\"recipient_email\":\"ops@example.com\"}"
```

## 10. Implementation Notes

- The chat agent is orchestrated with LangGraph and uses LangChain chat models plus prompt templates.
- The graph routes requests across intent detection, retrieval, workflow execution, and response generation.
- The RAG flow stores uploaded document chunks in FAISS and retrieves matching context for chat requests.
- Uploaded documents are persisted in PostgreSQL and saved under `storage/uploads`.
- A Celery task hook is triggered after document upload for additional async processing.
- Database tables are created automatically on startup for convenience.
- FastAPI root `/` now returns a small API status payload instead of serving a browser app.
- Chat generation is configured for Groq by default, while embeddings use local Ollama.

## 11. Production Hardening Suggestions

- Add Alembic migrations instead of relying on startup table creation
- Replace mock tool handlers with real internal APIs
- Add stronger RBAC and refresh-token flows
- Store files in object storage like S3
- Add observability with Prometheus, OpenTelemetry, and structured JSON logging
- Add rate limiting and API gateway controls
- Consider moving secrets into a vault and adding per-environment model selection for cost and latency control
