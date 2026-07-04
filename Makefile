.PHONY: up down logs test lint fmt shell

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	docker compose exec api pytest -v

lint:
	docker compose exec api ruff check .

fmt:
	docker compose exec api ruff format .

shell:
	docker compose exec api bash
