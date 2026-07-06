# DocMind

A multi-tenant RAG (retrieval-augmented generation) document Q&A service.
Upload documents, then chat with an AI that answers grounded in them and
cites its sources.

## Run it

```bash
docker compose up --build
make migrate          # applies the users table migration
```

Then check:

- http://localhost:8000/        — basic liveness check
- http://localhost:8000/health  — confirms Postgres and Redis are reachable
- http://localhost:8000/docs    — interactive API docs (Swagger UI)

The auth endpoints are live at `/auth/signup`, `/auth/login`, `/auth/me`.

## Useful commands

```bash
make up           # start everything
make down         # stop everything
make test         # full suite inside docker (unit + integration)
make test-unit    # standalone suite, no Docker needed
make migrate      # alembic upgrade head
make migration msg="add documents table"   # autogenerate a new migration
make lint         # ruff check
make logs         # tail all container logs
```

## Project layout

```
api/
├── core/        # settings, db session, logging
├── auth/        # Phase 1 — signup / login / JWT
├── documents/   # Phase 2
└── chat/        # Phase 4
alembic/         # DB migrations
worker/          # Celery tasks (Phase 3)
frontend/        # React + Vite chat UI (Phase 5)
tests/
```

## Roadmap

- [x] Phase 0 — repo skeleton, Postgres + Redis, CI, tests
- [x] Phase 1 — auth (signup/login, JWT, Alembic migrations)
- [ ] Phase 2 — document upload + object storage (MinIO)
- [ ] Phase 3 — ingestion pipeline (Celery: extract, chunk, embed → pgvector)
- [ ] Phase 4 — retrieval + chat (similarity search, LLM call, SSE streaming)
- [ ] Phase 5 — frontend (React + Vite)
- [ ] Phase 6 — hardening (rate limiting, structured logging, test coverage)
- [ ] Phase 7 — deploy
