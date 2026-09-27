"""Конфигурация приложения. Все параметры читаются из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Иммутабельная конфигурация; создаётся один раз при старте сервера."""

    chroma_dir: str = "./data/chroma"
    embedding_provider: str = "default"  # "default" | "ollama"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct"
    ollama_timeout: float = 180.0
    ollama_retries: int = 2

    chunk_size: int = 1000
    chunk_overlap: int = 150

    top_k: int = 5
    min_relevant_chunks: int = 2
    max_retrieval_loops: int = 2
    rrf_k: int = 60

    log_level: str = "INFO"
    mcp_transport: str = "stdio"  # "stdio" | "http"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    supported_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}
        )
    )

    @property
    def embedding_is_ollama(self) -> bool:
        return self.embedding_provider.lower() == "ollama"


def load_settings() -> Settings:
    """Собирает Settings из окружения (env > значения по умолчанию)."""
    return Settings(
        chroma_dir=os.getenv("CHROMA_DIR", "./data/chroma"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "default").lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b-instruct"),
        ollama_timeout=_env_float("OLLAMA_TIMEOUT", 180.0),
        ollama_retries=_env_int("OLLAMA_RETRIES", 2),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 5),
        min_relevant_chunks=_env_int("MIN_RELEVANT_CHUNKS", 2),
        max_retrieval_loops=_env_int("MAX_RETRIEVAL_LOOPS", 2),
        rrf_k=_env_int("RRF_K", 60),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        mcp_transport=os.getenv("MCP_TRANSPORT", "stdio").lower(),
        mcp_host=os.getenv("MCP_HOST", "0.0.0.0"),
        mcp_port=_env_int("MCP_PORT", 8000),
    )
