# DocMind

A multi-tenant RAG (retrieval-augmented generation) document Q&A service.
Upload documents, then chat with an AI that answers grounded in them and
cites its sources.

This repo is being built in phases — see the roadmap below. **This commit
is Phase 0**: the skeleton boots, Postgres and Redis are wired up, and the
test/lint tooling works. No real features yet.

## Run it

```bash
docker compose up --build
```

Then check:

- http://localhost:8000/        — basic liveness check
- http://localhost:8000/health  — confirms the API can reach Postgres and Redis
- http://localhost:8000/docs    — interactive API docs (Swagger UI)

## Useful commands

```bash
make up      # start everything (postgres, redis, api, worker)
make down    # stop everything
make test    # run the test suite inside the api container
make lint    # run ruff
make logs    # tail logs from all services
```

## Project layout

```
api/
├── core/        # settings, db session, logging
├── auth/        # Phase 1
├── documents/   # Phase 2
└── chat/        # Phase 4
worker/          # Celery tasks (Phase 3)
frontend/        # React + Vite chat UI (Phase 5)
tests/
```

## Roadmap

- [x] Phase 0 — repo skeleton, Postgres + Redis, CI, tests
- [ ] Phase 1 — auth (signup/login, JWT)
- [ ] Phase 2 — document upload + object storage
- [ ] Phase 3 — ingestion pipeline (Celery: extract, chunk, embed into pgvector)
- [ ] Phase 4 — retrieval + chat (similarity search, LLM call, SSE streaming)
- [ ] Phase 5 — frontend
- [ ] Phase 6 — hardening (rate limiting, structured logging, test coverage)
- [ ] Phase 7 — deploy
