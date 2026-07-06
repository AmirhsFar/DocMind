.PHONY: up down logs test test-unit lint fmt shell migrate migration

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	docker compose exec api pytest -v

test-unit:
	pytest -v -m "not integration"

lint:
	docker compose exec api ruff check .

fmt:
	docker compose exec api ruff format .

shell:
	docker compose exec api bash

# Apply all pending Alembic migrations (run after docker compose up).
migrate:
	docker compose exec api alembic upgrade head

# Generate a new migration from ORM model changes.
# Usage: make migration msg="add documents table"
migration:
	docker compose exec api alembic revision --autogenerate -m "$(msg)"
