"""Rate limiter для AuthAPI: скользящее окно на Redis."""

import time

from redis import Redis

from src.config import TOKEN_EXPIRY_HOURS

WINDOW_SECONDS = 60
MAX_ATTEMPTS = 10


class RateLimiter:
    """Ограничивает число попыток входа: не более MAX_ATTEMPTS за минуту на IP."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def allow(self, ip: str) -> bool:
        """True, если попытка входа с ip укладывается в лимит."""
        key = f"rl:{ip}:{int(time.time() // WINDOW_SECONDS)}"
        count = self._redis.incr(key)
        if count == 1:
            self._redis.expire(key, WINDOW_SECONDS)
        return count <= MAX_ATTEMPTS


def legacy_token_ttl() -> int:
    """TTL legacy-токена в секундах; используется только старыми клиентами."""
    return TOKEN_EXPIRY_HOURS * 3600
