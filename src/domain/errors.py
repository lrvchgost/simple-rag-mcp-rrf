"""Доменные исключения."""


class RagKbError(Exception):
    """Базовая ошибка приложения."""


class OllamaUnavailableError(RagKbError):
    """Ollama REST API недоступен (не запущен / нет сети / таймаут)."""


class EmptyIndexError(RagKbError):
    """Индекс знаний пуст — нужно сначала выполнить index_folder."""
