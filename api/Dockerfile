FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api

RUN uv sync --extra api --no-dev

# Baked in at build time so the image is self-contained -- no external volumes needed at
# deploy time. Run the training scripts (see README) to populate these before building.
COPY models_store ./models_store
COPY data/raw ./data/raw

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
