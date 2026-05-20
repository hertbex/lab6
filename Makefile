.PHONY: run lint type-check check

run:
	uv run streamlit run st_demo.py

lint:
	uv run ruff check . --fix
	uv run ruff format .

type-check:
	uv run mypy .

check: lint type-check
