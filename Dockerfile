# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

# Hugging Face Spaces runs the container as UID 1000.
RUN useradd --create-home --uid 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    FLEET_RUNTIME_DIR=/home/user/runtime \
    WEB_HOST=0.0.0.0 \
    PORT=7860

WORKDIR /home/user/app

# Dependencies first so this layer is cached until the lock file changes.
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user:user . .

# Pre-warm: train the delay model, build the ChromaDB index and download the
# ~80 MB all-MiniLM-L6-v2 ONNX model into $HOME/.cache/chroma (owned by user).
RUN mkdir -p "$FLEET_RUNTIME_DIR" "$HOME/.cache" \
    && python -c "from src.mcp.tools.delay_predictor import forecast; from src.mcp.tools.chroma_rag import corpus_stats; print(forecast(10, 0.8, 0, 12, 10, 'Van', 'Zone A', 'Zone B')); print(corpus_stats())"

EXPOSE 7860

CMD ["python", "start.py"]
