"""
api/rate_limiter.py
-------------------
IP-based sliding-window rate limiter for FastAPI.

- Pure stdlib: no Redis, no external deps
- Thread-safe via threading.Lock
- Sliding window algorithm: more accurate than fixed buckets
- Pluggable: works as a FastAPI dependency or standalone
"""

import time
import threading
from collections import defaultdict, deque

from fastapi import Request, HTTPException


# ── Config dataclass ───────────────────────────────────────────────────────────

class RateLimitConfig:
    """
    Encapsulates all rate-limit policy for one endpoint group.

    Args:
        max_requests:   Maximum allowed calls in the window.
        window_seconds: Rolling window size in seconds.
        message:        Human-readable message returned on HTTP 429.
    """
    def __init__(
        self,
        max_requests:   int = 5,
        window_seconds: int = 60,
        message:        str = (
            "Rate limit exceeded. "
            "You may send {max_requests} requests per {window_seconds}s. "
            "Retry after {retry_after}s."
        ),
    ) -> None:
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self.message        = message

    def format_message(self, retry_after: int) -> str:
        return self.message.format(
            max_requests   = self.max_requests,
            window_seconds = self.window_seconds,
            retry_after    = retry_after,
        )


# ── In-memory sliding-window store ────────────────────────────────────────────

class RateLimitStore:
    """
    Tracks per-IP request timestamps using a deque per client.

    Sliding-window logic:
      - Each request appends its timestamp.
      - On every check, timestamps older than `window_seconds` are discarded.
      - If the remaining count >= max_requests → deny.

    Thread-safety: a single reentrant lock guards all mutations.
    """

    def __init__(self) -> None:
        # ip → deque of float timestamps
        self._buckets: dict = defaultdict(deque)
        self._lock = threading.RLock()

    # ── Core check ────────────────────────────────────────────────────────────

    def is_allowed(self, ip: str, config: RateLimitConfig) -> tuple:
        """
        Check whether `ip` is within its rate limit.

        Returns:
            (allowed: bool, retry_after: int)
            retry_after is 0 when allowed, otherwise seconds until the
            oldest request falls outside the window.
        """
        now    = time.monotonic()
        cutoff = now - config.window_seconds

        with self._lock:
            bucket = self._buckets[ip]

            # Evict expired timestamps (sliding window)
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= config.max_requests:
                # Seconds until the oldest request exits the window
                retry_after = int(bucket[0] - cutoff) + 1
                return False, retry_after

            # Admit the request and record its timestamp
            bucket.append(now)
            return True, 0

    # ── Introspection ─────────────────────────────────────────────────────────

    def remaining(self, ip: str, config: RateLimitConfig) -> int:
        """Return how many requests the IP has left in the current window."""
        now    = time.monotonic()
        cutoff = now - config.window_seconds
        with self._lock:
            bucket = self._buckets[ip]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            return max(0, config.max_requests - len(bucket))

    def usage_snapshot(self) -> dict:
        """
        Return a dict of {ip: request_count} for all active IPs.
        Useful for a /admin/rate-limits diagnostic endpoint.
        """
        with self._lock:
            return {ip: len(dq) for ip, dq in self._buckets.items() if dq}

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear(self, ip: str = None) -> None:
        """Clear one IP or the entire store (useful in tests)."""
        with self._lock:
            if ip:
                self._buckets.pop(ip, None)
            else:
                self._buckets.clear()


# ── Shared singleton store (one per process) ──────────────────────────────────

_default_store = RateLimitStore()


# ── FastAPI dependency factory ─────────────────────────────────────────────────

def rate_limit(
    config: RateLimitConfig = None,
    store:  RateLimitStore  = None,
):
    """
    FastAPI dependency factory.  Returns an async callable that FastAPI
    injects before the route handler runs.

    Usage:
        from api.rate_limiter import rate_limit, RateLimitConfig

        RUN_LIMIT = RateLimitConfig(max_requests=5, window_seconds=60)

        @app.get("/run", dependencies=[Depends(rate_limit(RUN_LIMIT))])
        def run_pipeline(): ...

    The dependency:
      1. Extracts the real client IP (honours X-Forwarded-For for proxies).
      2. Checks the sliding-window store.
      3. Attaches rate-limit headers to the response.
      4. Raises HTTP 429 with structured JSON body if over the limit.
    """
    _config = config or RateLimitConfig()
    _store  = store  or _default_store

    async def _check(request: Request) -> None:
        ip = _get_client_ip(request)
        allowed, retry_after = _store.is_allowed(ip, _config)

        # Stash remaining count for the header middleware
        remaining = _store.remaining(ip, _config)
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_ip        = ip
        request.state.rate_limit_limit     = _config.max_requests
        request.state.rate_limit_window    = _config.window_seconds

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error":       "rate_limit_exceeded",
                    "message":     _config.format_message(retry_after),
                    "limit":       _config.max_requests,
                    "window_sec":  _config.window_seconds,
                    "retry_after": retry_after,
                    "client_ip":   ip,
                },
                headers={
                    "Retry-After":           str(retry_after),
                    "X-RateLimit-Limit":     str(_config.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Window":    str(_config.window_seconds),
                },
            )

    return _check


# ── Response header middleware ─────────────────────────────────────────────────

async def add_rate_limit_headers(request: Request, call_next):
    """
    Starlette middleware that appends X-RateLimit-* headers to every response
    so the client always knows its current quota.

    Mount with:
        app.middleware("http")(add_rate_limit_headers)
    """
    response = await call_next(request)

    remaining = getattr(request.state, "rate_limit_remaining", None)
    if remaining is not None:
        response.headers["X-RateLimit-Limit"]     = str(getattr(request.state, "rate_limit_limit",  ""))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"]    = str(getattr(request.state, "rate_limit_window", ""))
        response.headers["X-RateLimit-IP"]        = getattr(request.state, "rate_limit_ip", "")

    return response


# ── IP extraction helper ───────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """
    Extract the real client IP, respecting reverse-proxy headers.
    Priority: X-Forwarded-For → X-Real-IP → direct connection.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # May be comma-separated; the first entry is the original client
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    return request.client.host if request.client else "unknown"
