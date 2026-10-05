import asyncio

class ActiveRequestCounter:
    def __init__(self):
        self.value = 0
        self._lock = asyncio.Lock()

    async def inc(self):
        async with self._lock: self.value += 1

    async def dec(self):
        async with self._lock: self.value -= 1

def choose_engine(counter: ActiveRequestCounter, cap: int, hysteresis: int = 2):
    """Returns True if fast engine should be used."""
    load = counter.value
    # add hysteresis to avoid flipping
    if load > cap:
        return True
    if hasattr(choose_engine, 'last_using_fast') and choose_engine.last_using_fast:
        return load > cap - hysteresis
    choose_engine.last_using_fast = load > cap
    return choose_engine.last_using_fast
retry_backoff.py – Exponential Backoff with Jitter
python
"""Retry wrapper with exponential backoff and jitter (Equation 4th, page 19)."""
import asyncio, random, time
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")

async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    **kwargs
) -> T:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt == max_retries: raise
            status = getattr(e, 'status_code', None)
            if status not in (429, 503): raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter_amount = delay * jitter * (random.random() * 2 - 1)
            wait = delay + jitter_amount
            if hasattr(e, 'headers') and e.headers.get('Retry-After'):
                try: wait = float(e.headers['Retry-After'])
                except: pass
            print(f"[retry] sleeping {wait:.2f}s (attempt {attempt+1})")
            await asyncio.sleep(wait)
    raise last_exc if last_exc else RuntimeError("all retries failed")