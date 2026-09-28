FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md constraints.txt ./
COPY src ./src

RUN pip install --no-cache-dir -c constraints.txt .

ENV PYTHONUNBUFFERED=1 \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    CHROMA_DIR=/app/data/chroma

EXPOSE 8000

CMD ["python", "-m", "src.server"]
