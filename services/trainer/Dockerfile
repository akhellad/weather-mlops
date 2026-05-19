FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY src ./src

ENV PATH="/app/.venv/bin:$PATH"

ENV GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["python", "-m", "src.train"]
